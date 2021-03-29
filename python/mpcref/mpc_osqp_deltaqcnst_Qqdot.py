import numpy as np
import osqp
import scipy.sparse as sparse


class MPCController(object):
    def __init__(self, A, B, s_d, Np, Q1, Q2, Q3, Q4, Q5, QDg, x_min, x_max, delta_x_min, delta_x_max,
                 x_zero=None, x_minus1=None, s_minus1=None):

        # system matrices
        self.A = A
        self.A_dense = np.copy(A) #np.array(self.A.todense())
        self.B = B

        # system dimensions
        self.n_x = A.shape[0]
        self.n_s = B.shape[1]
        self.n_q = self.n_x - self.n_s

        # reference
        self.s_d = s_d

        # MPC cost
        self.Np = Np  # prediction/control horizon
        self.Q1 = Q1  # penalty on s - s_d
        self.Q2 = Q2  # penalty on s_star - s_d
        self.Q3 = Q3  # unused
        self.Q4 = Q4  # penalty on q(k) - 2q(k=1) + q(k-2) (second-order derivative approx)
        self.Q5 = Q5  # penalty on slack variables
        self.QDg = QDg  # penalty on s_star(k) - s_star(k-1)

        # MPC constraints
        self.x_min = x_min
        self.x_max = x_max

        self.delta_x_min = delta_x_min
        self.delta_x_max = delta_x_max

        self.s_star_min = -1e6*np.ones(self.n_s)
        self.s_star_max = 1e6*np.ones(self.n_s)

        if s_minus1 is not None:
            self.s_minus1 = np.copy(s_minus1)
        else:
            self.s_minus1 = np.zeros(self.n_s)
        if x_zero is not None:
            self.x_zero = np.copy(x_zero)
        else:
            self.x_zero = np.zeros(self.n_x)

        if x_minus1 is not None:
            self.x_minus1 = np.copy(x_minus1)
        else:
            self.x_minus1 = np.copy(self.x_zero)

        # OSQP problem
        self.problem = None
        self.res = None

        # QP problem components
        self.P_QP = None
        self.p_QP = None
        self.A_QP = None
        self.lower_QP = None
        self.upper_QP = None

        self.Q4_x = None
        self.Q1_x = None
        self.p1_x = None
        self.p2_sst = None

        # options
        self.SOFT_ON = True  # soft constraints with slack variables on

        self.DELTA = sparse.kron(np.eye(self.Np), sparse.eye(self.n_x)) - sparse.kron(np.eye(self.Np, k=-1), sparse.eye(self.n_x))

    def setup(self, solve=True):
        # From term Q1
        self.Q1_x = sparse.block_diag([self.Q1, sparse.csc_matrix((self.n_q, self.n_q))], format="csc")
        P1_x = sparse.kron(sparse.eye(self.Np), self.Q1_x)  # x1...x_N
        self.p1_x = np.kron(np.ones(self.Np), np.r_[-self.Q1.dot(self.s_d), np.zeros(self.n_q)])  # x1... x_Ns

        # From term Q2
        P2_sst = sparse.kron(sparse.eye(self.Np), self.Q2)  # s*_0 ... s*_N-1
        self.p2_sst = np.kron(np.ones(self.Np), -self.Q2.dot(self.s_d))

        # From term QDg
        iDu = 2 * sparse.eye(self.Np) - sparse.eye(self.Np, k=1) - sparse.eye(self.Np, k=-1)
        iDu[self.Np - 1, self.Np - 1] = 1
        PDg_sst = sparse.kron(iDu, self.QDg)
        pDg_sst = np.hstack([-self.QDg.dot(self.s_minus1), np.zeros((self.Np - 1) * self.n_s)])  # u1..uN-1

        # From term Q4
        self.Q4_x = sparse.block_diag([sparse.csc_matrix((self.n_s, self.n_s)), self.Q4], format="csc")

        iDq = 2 * sparse.eye(self.Np) - 1 * sparse.eye(self.Np, k=1) - 1 * sparse.eye(self.Np, k=-1)
        iDq[self.Np - 1, self.Np - 1] = 1.0
        P4_x = sparse.kron(iDq, self.Q4_x)
        p4_x = np.r_[- self.Q4_x.dot(self.x_zero), np.zeros((self.Np - 1) * self.n_x)]

        # From all terms
        P_x = P1_x + P4_x
        P_sst = P2_sst + PDg_sst
        p_x = self.p1_x + p4_x
        p_sst = self.p2_sst + pDg_sst

        # From soft term Q_eps
        if self.SOFT_ON:
            P_eps = self.Q5 * sparse.eye(2 * self.Np * self.n_x)
            p_eps = np.zeros(2 * self.Np * self.n_x)

        # Linear dynamics constraints
        A_cal = sparse.kron(sparse.eye(self.Np, k=-1), self.A)
        B_cal = sparse.kron(sparse.eye(self.Np, k=0), self.B)

        Aeq_dyn = sparse.hstack([A_cal - sparse.eye(self.Np * self.n_x), B_cal])
        if self.SOFT_ON:
            Aeq_dyn = sparse.hstack([Aeq_dyn, sparse.csc_matrix((Aeq_dyn.shape[0], 2 * self.Np * self.n_x))])  # For soft constraints slack variables

        c0 = -np.matmul(np.array(self.A_dense), self.x_zero)
        leq_dyn = np.hstack([c0, np.zeros((self.Np - 1) * self.n_x)])
        ueq_dyn = leq_dyn  # for equality constraints -> upper bound  = lower bound!

        # Inequality constraints in u, x, deltax
        Aineq_u = sparse.hstack([sparse.csc_matrix((self.Np * self.n_s, self.Np * self.n_x)),
                                sparse.eye(self.Np * self.n_s)])
        if self.SOFT_ON:
            Aineq_u = sparse.hstack([Aineq_u, sparse.csc_matrix((self.Np * self.n_s, 2 * self.Np * self.n_x))])
        lineq_u = np.kron(np.ones(self.Np), self.s_star_min)
        uineq_u = np.kron(np.ones(self.Np), self.s_star_max)

        Aineq_x = sparse.hstack([sparse.eye(self.Np * self.n_x), sparse.csc_matrix((self.Np * self.n_x, self.Np * self.n_s))])
        if self.SOFT_ON:
            Aineq_x = sparse.hstack([Aineq_x, sparse.eye(self.Np * self.n_x), sparse.csc_matrix((self.Np * self.n_x, self.Np * self.n_x))])
        lineq_x = np.kron(np.ones(self.Np), self.x_min)
        uineq_x = np.kron(np.ones(self.Np), self.x_max)

        Aineq_Dx = sparse.hstack([self.DELTA, sparse.csc_matrix((self.Np * self.n_x, self.Np * self.n_s))])
        if self.SOFT_ON:
            Aineq_Dx = sparse.hstack([Aineq_Dx, sparse.csc_matrix((self.Np * self.n_x, self.Np * self.n_x)), sparse.eye(self.Np * self.n_x)])
        lineq_Dx = np.kron(np.ones(self.Np), self.delta_x_min) + np.r_[self.x_zero, np.zeros((self.Np - 1) * self.n_x)]
        uineq_Dx = np.kron(np.ones(self.Np), self.delta_x_max) + np.r_[self.x_zero, np.zeros((self.Np - 1) * self.n_x)]

        # Stack all inequalities
        Aineq = sparse.vstack([Aineq_x, Aineq_u, Aineq_Dx])
        lineq = np.r_[lineq_x, lineq_u, lineq_Dx]
        uineq = np.r_[uineq_x, uineq_u, uineq_Dx]

        self.A_QP = sparse.vstack([Aeq_dyn, Aineq], format="csc")
        self.lower_QP = np.hstack([leq_dyn, lineq])
        self.upper_QP = np.hstack([ueq_dyn, uineq])

        self.P_QP = sparse.block_diag([P_x, P_sst], format="csc")
        self.p_QP = np.r_[p_x, p_sst]

        if self.SOFT_ON:
            self.P_QP = sparse.block_diag([self.P_QP, P_eps], format="csc")
            self.p_QP = np.r_[self.p_QP, p_eps]

        # Setup problem
        self.problem = osqp.OSQP()
        self.problem.setup(P=self.P_QP, q=self.p_QP, A=self.A_QP, l=self.lower_QP, u=self.upper_QP,
                           warm_start=True, verbose=False, eps_abs=1e-3, eps_rel=1e-3)

        if solve:
            self.solve()

    def update(self, x_zero, x_minus1=None, s_minus1=None, s_d=None, A=None, B=None, solve=True):

        if s_minus1 is not None:
            self.s_minus1 = np.copy(s_minus1)

        if x_minus1 is not None:
            self.x_minus1 = np.copy(x_minus1)
        else:
            self.x_minus1 = np.copy(self.x_zero)

        self.x_zero = np.copy(x_zero)

        if A is not None and B is not None:
            self.A = A
            self.B = B
            A_cal = sparse.kron(sparse.eye(self.Np, k=-1), self.A)
            B_cal = sparse.kron(sparse.eye(self.Np, k=0), self.B)

            # new dynamics constraints
            Aeq_dyn = sparse.hstack([A_cal - sparse.eye(self.Np * self.n_x), B_cal])
            self.A_QP[:self.Np * self.n_x, :self.Np * (self.n_x + self.n_s)] = Aeq_dyn

        if A is not None:
            self.A_dense = np.copy(A)  #np.array(A.todense())

        # minimal update
        # Initial state for dynamics constraint
        c0 = -np.matmul(self.A_dense, self.x_zero)
        self.lower_QP[:self.n_x] = c0
        self.upper_QP[:self.n_x] = c0

        # Initial state for delta_q constraint
        idx_delta = (self.n_s + 2 * self.n_x) * self.Np
        self.lower_QP[idx_delta:idx_delta + self.n_x] = self.delta_x_min + self.x_zero
        self.upper_QP[idx_delta:idx_delta + self.n_x] = self.delta_x_max + self.x_zero

        # From term Q1
        if s_d is not None:
            self.p1_x = np.kron(np.ones(self.Np), np.r_[-self.Q1.dot(s_d), np.zeros(self.n_q)])  # x1... x_Ns

        # From term Q2
        if s_d is not None:
            self.p2_sst = np.kron(np.ones(self.Np), -self.Q2.dot(s_d))

        # From term Q4
        p4_x = np.r_[- self.Q4_x.dot(self.x_zero), np.zeros((self.Np - 1) * self.n_x)]

        p_x = self.p1_x + p4_x

        # From term QDg
        pDg_sst = np.hstack([-self.QDg.dot(self.s_minus1), np.zeros((self.Np - 1) * self.n_s)])  # u1..uN-1
        p_sst = self.p2_sst + pDg_sst

        # minimal update
        self.p_QP[:(self.n_x*self.Np)] = p_x
        self.p_QP[self.n_x*self.Np:(self.n_x+self.n_s)*self.Np] = p_sst

        self.problem.update(l=self.lower_QP, u=self.upper_QP, q=self.p_QP, Ax=self.A_QP.data)

        if solve:
            self.solve()

    def solve(self):
        self.res = self.problem.solve()

    def output(self, update_s_star_old=True):
        s_star_MPC = self.res.x[(self.Np * self.n_x): (self.Np * self.n_x) + self.n_s]

        if update_s_star_old:
            self.s_minus1 = np.copy(s_star_MPC)
        return np.copy(s_star_MPC)

    def full_output(self):
        x_seq = self.res.x[:(self.Np * self.n_x)].reshape((self.Np, self.n_x))
        s_star_seq = self.res.x[(self.Np * self.n_x): self.Np * (self.n_x + self.n_s)].reshape((self.Np, self.n_s))
        gamma_x_seq = self.res.x[self.Np * (self.n_x + self.n_s):self.Np * (2*self.n_x + self.n_s)].reshape((self.Np, self.n_x))
        gamma_deltax_seq = self.res.x[self.Np * (2 * self.n_x + self.n_s):self.Np * (3*self.n_x + self.n_s)].reshape((self.Np, self.n_x))
        return s_star_seq, x_seq, gamma_x_seq, gamma_deltax_seq
