#!/usr/bin/env python
from __future__ import division
import rospy
import xbot_interface.config_options as xbot_opt
import xbot_interface.xbot_interface as xbot
from xbot_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
from cartesian_interface.pyci_all import *
from opensot_visual_servoing.msg import VisualFeature, VisualFeatures
from sensor_msgs.msg import CameraInfo
import numpy as np
import pandas as pd
import time

def createRobot():
    opt = xbot_opt.ConfigOptions()

    urdf = rospy.get_param('xbotcore/robot_description')
    srdf = rospy.get_param('xbotcore/robot_description_semantic')

    opt = xbot_opt.ConfigOptions()
    opt.set_urdf(urdf)
    opt.set_srdf(srdf)
    opt.generate_jidmap()
    opt.set_bool_parameter('is_model_floating_base', True)
    opt.set_string_parameter('model_type', 'RBDL')
    opt.set_string_parameter('framework', 'ROS')
    return xbot.RobotInterface(opt)


def extract_base_data(data):
    id = data.name.index("coman")
    base_pose_gazebo = data.pose[id]
    base_twist_gazebo = data.twist[id]

    base_pose = Affine3(pos=[base_pose_gazebo.position.x, base_pose_gazebo.position.y, base_pose_gazebo.position.z],
                        rot=[base_pose_gazebo.orientation.w, base_pose_gazebo.orientation.x, base_pose_gazebo.orientation.y, base_pose_gazebo.orientation.z])
    base_twist = [base_twist_gazebo.linear.x, base_twist_gazebo.linear.y, base_twist_gazebo.linear.z, base_twist_gazebo.angular.x, base_twist_gazebo.angular.y, base_twist_gazebo.angular.z]

    return base_pose, base_twist

# TODO: to be checked
def extract_joint_data(data):

    joint_positions = data.link_position
    joint_velocities = data.link_velocity

    return joint_positions, joint_velocities    

def get_intrinsic_param(msg):
        
    # Extract the camera intrinsic parameters from the matrix K 
    # (Ref.: http://docs.ros.org/melodic/api/sensor_msgs/html/msg/CameraInfo.html)

    print("Got camera info!")

    intrinsic = {'fx' : 0, 'fy': 0, 'cx': 0, 'cy' : 0}
    
    intrinsic['fx'] = msg.K[0]
    intrinsic['fy'] = msg.K[4]
    intrinsic['cx'] = msg.K[2]
    intrinsic['cy'] = msg.K[5]

    return intrinsic

def getFeaturesAndDepths(msg):
    features = np.array([])
    depths = np.array([])
    for feature in msg.features:
        features = np.append(features, np.array([feature.x, feature.y]))
        depths = np.append(depths, feature.Z)

    return features, depths

def visjac_p(intrinsic, feat_vec, depths):

    # Taken and adapted from RCV Matlab toolbox
    # Reference:
    # [1] P. Corke, "Robotics, Vision & Control: Fundamental algorithms in
    # MATLAB," Springer, 2011

    L = np.zeros((feat_vec.shape[0], 6))

    for i in np.arange(0,feat_vec.shape[0],2): # iterate over the feature vector
 	       
        uv = np.array(feat_vec[i:i+2])

        # Take the depth
        Z = depths[max(0,i//2)]
        
        # Convert to normalized image-plane coordinates
        x = (uv[0] - intrinsic['cx']) / intrinsic['fx']
        y = (uv[1] - intrinsic['cy']) / intrinsic['fy']

        L_i = np.array([
            [1/Z, 0,   -x/Z, -x*y,     (1+x*x), -y],
            [0,   1/Z, -y/Z, -(1+y*y), x*y,      x]
        ])

        L_i = - np.matmul(np.diag([intrinsic['fx'], intrinsic['fy']]), L_i)

        L[i:i+2,:] = L_i

    return L

def discretizeMPC(A_bar,B_bar,C_bar,D_bar,c):
    
    # Function to go from discretized model (with sampling time T) to matrices 
    # of the discrete-time model used by the MPC to close the loop at a sampling 
    # time T_MPC = c *T
    # Inputs:
    # - A_bar, B_bar, C_bar, D_bar: matrices of the SS model already discretized 
    #   with time sample $T
    # - c: integer such that T_MPC = c * T
    # Outputs:
    # - A, B , C, D: matrices of the SS discrete-time model used by the MPC

    nx,nu = np.shape(B_bar)
    ny,_ = np.shape(C_bar)

    A = A_bar
    B_list = [B_bar]
    B = B_bar
    
    for ind in range(1,c): # from 1 to c-1
        B_list.append(np.matmul(A_bar,B_list[-1]))
        B = B_list[-1] + B
        A = np.matmul(A,A_bar)
      
    C = C_bar
    D = D_bar
          
    ## This is another option:
    ## For j = 0    
    #A = A_bar 
    #B = B_bar 
    #t = time.time()
    ## For j from 1 to c-1
    #for j in range(1,c):
    #    B = B + np.matmul(A,B_bar)
    #    A = np.matmul(A,A_bar)
    
    #For comparison
    #print(np.linalg.matrix_power(A_bar,c))
    
    return A, B, C, D  

def compute_system_matrices(J,J_const,vs_gain,T,c):

    # Function used to compute the matrices of the system describing
    # the closed-loop behavior of the visual servoing with a humanoid 
    # robot:
    #     s_{h+1} = A_h * s_h + B_h * s_star_h
    #         y_h = C_h * s_h + D_h * s_star_h 
    # Inputs:
    # - J, the Jacobian of the VS task
    # - J_const, the Jacobian of the task constraint 
    # - vs_gain, control gain of the VS task
    # - T, sampling time discretizing the time-continuous system
    # - c, integer such that T_MPC = c * T, being T_MPC the sampling
    #   of the MPC
    # Outputs:
    # - A_h, B_h, C_h and D_h matices 
    
    # Dimension of the robot state
    _, n_state = np.shape(J)
        
    # Number of visual features
    n_features, _ = np.shape(L)

    # n x n identity matrix
    I_n = np.eye(n_state)
    
    # f x f identity matrix
    I_f = np.eye(n_features)

    # f x n zero matrix
    O_fn = np.zeros((n_features,n_state))
    
    # n x n zero matrix
    O_n = np.zeros((n_state,n_state))

    # Null prokector matrix of the equality constraints 
    J_const_pinv = np.linalg.pinv(J_const)
    P = I_n - np.matmul(J_const_pinv,J_const)

    # J * P
    JP = np.matmul(J,P)
    
    # (J*P)^+
    JP_pinv = np.linalg.pinv(JP)
    
    # lambda * T * J * (J*P)^+
    lambda_T_J_JP_pinv = vs_gain * T * np.matmul(J,JP_pinv)

    # lambda * (J*P)^+
    lambda_JP_pinv = vs_gain * JP_pinv

    # A_bar
    A11 = I_f - lambda_T_J_JP_pinv
    A12 = O_fn #np.zeros((n_features, n_state))
    A21 = - T * lambda_JP_pinv
    A22 = I_n #np.eye(n_state)

    A_bar = np.r_[np.c_[A11, A12],
                  np.c_[A21, A22]]   
    
    # B_bar
    B1 = lambda_T_J_JP_pinv
    B2 = T * lambda_JP_pinv
    B_bar = np.r_[B1, B2]
        
    # C_bar 
    C1 = np.eye(n_features + n_state)
    C2 = np.c_[-lambda_JP_pinv, O_n]
    C_bar = np.r_[C1, C2]

    # D_bar
    D1 = np.zeros((n_features + n_state, n_features))
    D2 = lambda_JP_pinv
    D_bar = np.r_[D1, D2]
    
    # Discretizing to get A, B, C and D matrices for the MPC
    A, B, C, D =  discretizeMPC(A_bar, B_bar, C_bar, D_bar, c) 

    return A, B, C, D

if __name__ == '__main__':

    rospy.init_node("reference_governor")

    # Create the robot 
    robot = createRobot()

    # Subscriber to the camera_info topic
    camera_info_msg = rospy.wait_for_message("/camera/rgb/camera_info", CameraInfo, timeout=None)
    intrinsic = get_intrinsic_param(camera_info_msg)

    # Publisher for the sequence of visual features
    vis_ref_seq = rospy.Publisher("/visual_reference_governor/visual_reference_sequence", VisualFeatures, queue_size=10, latch=True)
    
    # TODO: Time parameters: do we need to set these parameters somewhere else?
    # sampling time
    T = 0.001   
    
    # sampling time for the MPC
    TMPC = 0.1  
    # c parameter: it is such that TMPC = c * T
    c = int(TMPC/T) 

    rate = rospy.Rate(1/TMPC) 

    while not rospy.is_shutdown():
    
        # Sense robot state: FB pose and velocity
        data = rospy.wait_for_message("gazebo/model_states", ModelStates, timeout=None)
        base_pose, base_twist = extract_base_data(data)
    
        # TODO: what exactily this function does?
        robot.sense()
        
        # TODO: need to set the robot FB by using specific code (now not available)
        robot.model().setFloatingBaseState(base_pose, base_twist)
        
        robot.model().update()

        q = robot.model().getJointPosition() # 35 x 1
        q_dot = robot.model().getJointVelocity() # 35 x 1

        n_state = len(q)

        # TODO: should be taken from the cartesian interface
        vs_gain = 1.5 # lambda from the stack 
        
        # Compute the jacobian of the robot
        J_camera = robot.model().getJacobian('camera_link') 

        # Compute the interaction matrix (image Jacobian)# TODO DOUBLE CHECK
        visual_features = rospy.wait_for_message("cartesian/visual_servoing_camera_link/features", VisualFeatures, timeout=None) # TODO : /image_processing/visual_features
        features, depths = getFeaturesAndDepths(visual_features)
        L = visjac_p(intrinsic, features, depths)

        # Robot's camera frame to camera sensor twist transformation TODO to get from cartesian interface
        V = np.eye(6)

        # Compute the task Jacobian.
        J = np.matmul(L,V)
        J = np.matmul(J,J_camera)

        # Compute the Jacobian of the equality constraint (CMM in air, contacts on the ground)
        sim_on_earth = False
        if sim_on_earth:
            # TODO compute in case of in air sim. J_const =
            J_const = 0
        else:
            J_const, _ = robot.model().getCentroidalMomentumMatrix()
            
        # Build the matrices for the MPC # TODO check if the computation time is OK
        #t = time.time()
        A, B, C, D = compute_system_matrices(J,J_const,vs_gain,T,c)
        #print('Matrices computation time : ', time.time()-t)

        # Simulating a sequence of references (to be able to publish something) # TODO to be substituted with MPCpy function

        # Number of features TODO: to be better computed
        n_features, _ = np.shape(L)
        n_points = int(n_features/2)

        # State MPC
        x_MPC = np.r_[features.reshape(8,1),q.reshape(n_state,1),q_dot.reshape(n_state,1)] # TODO remove hard-coded numbers
        
        # Preview window size: TODO: to be properly set (as parameter?)
        Np = 20

        # This loop simulates a sequene of references, which will be computed by an MPC
        offset = 50
        dummy_des_feat = np.array([
                    intrinsic['cx']-offset, intrinsic['cy']-offset, intrinsic['cx']+offset, intrinsic['cy']-offset,
                    intrinsic['cx']+offset, intrinsic['cy']+offset, intrinsic['cx']-offset, intrinsic['cy']+offset
                    ])
        
        # Visual features messages (for the reference sequence)
        visual_reference_sequence_msg = VisualFeatures()

        # TODO MPC SETUP

        # TODO: MPC constraints: take (from Cartesio?) once, out of the loop
        #limits.q_dot_lower = limits.q_dot_lower
        #limits.q_dot_upper = limits.q_dot_upper
        #y_min = np.r_[-1e6*np.ones(ns), limits.q_lower, limits.q_dot_lower]
        #y_max = np.r_[1e6*np.ones(ns), limits.q_upper, limits.q_dot_upper]

        # TODO: this should be before the loop
        #K = MPCController(A=A_MPC, B=B_MPC, C=C_MPC, D=D_MPC, Np=Np, Q1=Q1, Q2=Q2, Q3=None, Q4=Q4, Q5=Q5, QDg=QDg, y_min=y_min, y_max=y_max)
        #K.setup()  # setup cvxpy problem

        # TODO fill with the proper values
        # K.update(x0_val=x_step, s_d_val=s_d_val, s_star_minus1_val=s_star_step, y_minus1_val=y_step)

        # TODO: get the ouptut
        s_star_step = K.output()


        '''print('here')
        df = pd.DataFrame(J)
        df.to_csv('~/Desktop/A.csv',index=False)
        df = pd.DataFrame(J_const)
        df.to_csv('~/Desktop/B.csv',index=False)
        df = pd.DataFrame(x_MPC)
        df.to_csv('~/Desktop/s_q_q_dot.csv',index=False)'''
        

        for n in range(0,Np):

            alpha = (n+1)/Np

            visualFeature_msg = VisualFeature()
            
            xx = features[0] * intrinsic['fx'] + intrinsic['cx'] 
            yy = features[1] * intrinsic['fy'] + intrinsic['cy']

            visualFeature_msg.x = (1-alpha) * xx + alpha * dummy_des_feat[0] 
            visualFeature_msg.y = (1-alpha) * yy + alpha * dummy_des_feat[1] 
            visualFeature_msg.Z = 0 # TODO what should we put here??? The one at the final destination?
            visualFeature_msg.type = visualFeature_msg.POINT

            visual_reference_sequence_msg.features.append(visualFeature_msg)

            visualFeature_msg = VisualFeature()
        
            xx = features[2] * intrinsic['fx'] + intrinsic['cx'] 
            yy = features[3] * intrinsic['fy'] + intrinsic['cy']

            visualFeature_msg.x = (1-alpha) * xx + alpha * dummy_des_feat[2]
            visualFeature_msg.y = (1-alpha) * yy + alpha * dummy_des_feat[3]
            visualFeature_msg.Z = 0 # TODO what should we put here??? The one at the final destination?
            visualFeature_msg.type = visualFeature_msg.POINT

            visual_reference_sequence_msg.features.append(visualFeature_msg)

            visualFeature_msg = VisualFeature()
            
            xx = features[4] * intrinsic['fx'] + intrinsic['cx'] 
            yy = features[5] * intrinsic['fy'] + intrinsic['cy']

            visualFeature_msg.x = (1-alpha) * xx + alpha * dummy_des_feat[4] 
            visualFeature_msg.y = (1-alpha) * yy + alpha * dummy_des_feat[5] 
            visualFeature_msg.Z = 0 # TODO what should we put here??? The one at the final destination?
            visualFeature_msg.type = visualFeature_msg.POINT

            visual_reference_sequence_msg.features.append(visualFeature_msg)

            visualFeature_msg = VisualFeature()

            xx = features[6] * intrinsic['fx'] + intrinsic['cx']
            yy = features[7] * intrinsic['fy'] + intrinsic['cy']

            visualFeature_msg.x = (1-alpha) * xx + alpha * dummy_des_feat[6]
            visualFeature_msg.y = (1-alpha) * yy + alpha * dummy_des_feat[7] 
            visualFeature_msg.Z = 0 # TODO what should we put here??? The one at the final destination?
            visualFeature_msg.type = visualFeature_msg.POINT

            visual_reference_sequence_msg.features.append(visualFeature_msg)

        visual_reference_sequence_msg.header.stamp = rospy.Time.now()

        vis_ref_seq.publish(visual_reference_sequence_msg)

        rate.sleep()
