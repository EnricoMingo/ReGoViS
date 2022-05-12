#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import tf
import sys
import time
import rospy
import scipy
import numpy as np
import pandas as pd

from std_msgs.msg import Float32
from sensor_msgs.msg import CameraInfo
from gazebo_msgs.msg import ModelStates
from franka_interface import ArmInterface
from mpcref.mpc_osqp_deltaqcnst_Qqdot import MPCController
from opensot_visual_servoing.msg import VisualFeature, VisualFeatures

# Global variable for the feedback
data = ModelStates()
visual_features = VisualFeatures()

# def createRobot():
#     opt = xbot_opt.ConfigOptions()

#     urdf = rospy.get_param('xbotcore/robot_description')
#     srdf = rospy.get_param('xbotcore/robot_description_semantic')

#     opt = xbot_opt.ConfigOptions()
#     opt.set_urdf(urdf)
#     opt.set_srdf(srdf)
#     opt.generate_jidmap()
#     opt.set_bool_parameter('is_model_floating_base', True)
#     opt.set_string_parameter('model_type', 'RBDL')
#     opt.set_string_parameter('framework', 'ROS')
#     return xbot.RobotInterface(opt)

# def extract_base_data(data):
#     id = data.name.index("coman")
#     base_pose_gazebo = data.pose[id]
#     base_twist_gazebo = data.twist[id]

#     base_pose = Affine3(pos=[base_pose_gazebo.position.x, base_pose_gazebo.position.y, base_pose_gazebo.position.z],
#                         rot=[base_pose_gazebo.orientation.x, base_pose_gazebo.orientation.y, base_pose_gazebo.orientation.z, base_pose_gazebo.orientation.w])
#     base_twist = [base_twist_gazebo.linear.x, base_twist_gazebo.linear.y, base_twist_gazebo.linear.z, base_twist_gazebo.angular.x, base_twist_gazebo.angular.y, base_twist_gazebo.angular.z]

#     return base_pose, base_twist

# def extract_joint_data(data):

#     # On real robot joint position should be motor position
#     joint_positions = data.link_position
#     joint_velocities = data.link_velocity

#     return joint_positions, joint_velocities    

def get_intrinsic_param(msg):
        
    # Extract the camera intrinsic parameters from the matrix K 
    # (Ref.: http://docs.ros.org/melodic/api/sensor_msgs/html/msg/CameraInfo.html)

    print("[RefGov] Got camera info!")

    intrinsic = {'fx' : 0, 'fy': 0, 'cx': 0, 'cy' : 0}
    
    intrinsic['fx'] = msg.K[0]
    intrinsic['fy'] = msg.K[4]
    intrinsic['cx'] = msg.K[2]
    intrinsic['cy'] = msg.K[5]

    # Size of camera is not really an intrinsic parameter, but I save it here for convenience
    intrinsic['height'] = msg.height
    intrinsic['width'] = msg.width

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
    # P. Corke, "Robotics, Vision & Control: Fundamental algorithms in MATLAB," Springer, 2011
    #
    # Visp links: 
    # https://visp-doc.inria.fr/doxygen/visp-daily/classvpFeaturePoint.html#afafc6dca7c571b8f2743defb1438fb44
    # https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-ibvs.html
    
    L = np.zeros((feat_vec.shape[0], 6))

    for i in np.arange(0,feat_vec.shape[0],2): # iterate over the feature vector
 	       
        #uv = np.array(feat_vec[i:i+2])
        xy = np.array(feat_vec[i:i+2])

        # Take the depth
        Z = depths[max(0,i//2)]
        
        # Convert to normalized image-plane coordinates
        x = xy[0] #(uv[0] - intrinsic['cx']) / intrinsic['fx']
        y = xy[1] #(uv[1] - intrinsic['cy']) / intrinsic['fy']

        L_i = np.array([
            [-1/Z, 0,   x/Z, x*y,     -(1+x*x),  y],
            [0,   -1/Z, y/Z, (1+y*y), -x*y,     -x]
        ])

        #L_i = - np.matmul(np.diag([intrinsic['fx'], intrinsic['fy']]), L_i)

        L[i:i+2,:] = L_i
        #L[i:i+2,:] = -L_i

    return L

def convert_to_normalized(features_in_pixel,intrinsic):

    # Ref.: https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-ibvs.html

    features_normalized = np.array([])

    for i in np.arange(0,len(features_in_pixel),2): # iterate over the feature vector
 	       
        uv = np.array(features_in_pixel[i:i+2])

        # Convert to normalized image-plane coordinates
        x = (uv[0] - intrinsic['cx']) / intrinsic['fx']
        y = (uv[1] - intrinsic['cy']) / intrinsic['fy']

        features_normalized = np.append(features_normalized, np.array([x, y]))

    return features_normalized

def convert_to_pixel(features_normalized,intrinsic):

    features_in_pixel = np.array([])

    for i in np.arange(0,len(features_normalized),2): # iterate over the feature vector

        xy = np.array(features_normalized[i:i+2])

        # Convert to normalized image-plane coordinates
        u = xy[0] * intrinsic['fx'] + intrinsic['cx']  
        v = xy[1] * intrinsic['fy'] + intrinsic['cy'] 

        features_in_pixel = np.append(features_in_pixel, np.array([u, v]))

    return features_in_pixel

def fill_visualFeatures_msg(data_in):
            
    visualFeatures_msg = VisualFeatures()
    
    # This assumes that we work with points
    for k in range(0,len(data_in),2):

        visualFeature_msg = VisualFeature()
    
        visualFeature_msg.x = data_in[k]  
        visualFeature_msg.y = data_in[k+1] 
        visualFeature_msg.Z = 0.5 # This depends on what the controller considers. It normally takes into account the Z of measured features
        visualFeature_msg.type = visualFeature_msg.POINT
        
        visualFeatures_msg.features.append(visualFeature_msg)

    return visualFeatures_msg

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

def compute_system_matrices(J,vs_gain,T,c):

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
    
    # Dimension of the visual features and robot state
    n_features, n_state = np.shape(J)
    
    # n x n identity matrix
    I_n = np.eye(n_state)
    
    # f x f identity matrix
    I_f = np.eye(n_features)

    # f x n zero matrix
    O_fn = np.zeros((n_features,n_state))
    
    # n x n zero matrix
    O_n = np.zeros((n_state,n_state))

    # Null prokector matrix of the equality constraints 
    #J_const_pinv = np.linalg.pinv(J_const)
    #P = I_n - np.matmul(J_const_pinv,J_const)

    # J * P
    #JP = np.matmul(J,P) ######################################## TODO: removed projector
    
    # (J*P)^+
    #JP_pinv = np.linalg.pinv(JP)
    J_pinv = np.linalg.pinv(J)
    
    # lambda * T * J * (J*P)^+
    lambda_T_J_J_pinv = vs_gain * T * np.matmul(J,J_pinv)#JP_pinv)

    # lambda * (J*P)^+
    lambda_J_pinv = vs_gain * J_pinv

    # A_bar
    A11 = I_f - lambda_T_J_J_pinv
    A12 = O_fn #np.zeros((n_features, n_state))
    A21 = - T * lambda_J_pinv
    A22 = I_n #np.eye(n_state)

    A_bar = np.r_[np.c_[A11, A12],
                  np.c_[A21, A22]]   
    
    # B_bar
    B1 = lambda_T_J_J_pinv
    B2 = T * lambda_J_pinv
    B_bar = np.r_[B1, B2]
        
    # C_bar 
    C1 = np.eye(n_features + n_state)
    C2 = np.c_[-lambda_J_pinv, O_n]
    C_bar = np.r_[C1, C2]

    # D_bar
    D1 = np.zeros((n_features + n_state, n_features))
    D2 = lambda_J_pinv
    D_bar = np.r_[D1, D2]
    
    # Discretizing to get A, B, C and D matrices for the MPC
    A, B, C, D =  discretizeMPC(A_bar, B_bar, C_bar, D_bar, c) 

    return A, B, C, D

def model_states_cb(msg):
    global data
    data = msg 

def visual_features_cb(msg):
    global visual_features
    visual_features = msg 

def skew_mat(p):
    return np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])

if __name__ == '__main__':

    rospy.init_node("reference_governor")

    # Create the robot 
    robot = ArmInterface()
    
    vals_pos = robot.joint_angles()
    vals_vel = robot.joint_velocities()
    joint_names = robot.joint_names()

    rospy.loginfo('RefGov: Moving to neutral')
    robot.move_to_neutral()
    
    # Subscriber to the camera_info topic
    camera_info_msg = rospy.wait_for_message("/d435_color/camera_info", CameraInfo, timeout=None)
    intrinsic = get_intrinsic_param(camera_info_msg)
    
    # Transformations

    tf_listener = tf.TransformListener()
    tf_broadcaster = tf.TransformBroadcaster()

    # wait 5s for the robot arm and camera to be spawned in
    try:
        tf_listener.waitForTransform('world', 'd435_color_optical_frame',rospy.Time(), rospy.Duration(5))
    except tf.Exception:
        rospy.logerr('Cannot establish pose of the camera')
        rospy.signal_shutdown('Cannot establish pose of the camera')

    # note: this is constant camera-end-effector transformation needed in the control
    c_t_n, c_q_n = tf_listener.lookupTransform( 'd435_color_optical_frame', 'panda_hand',rospy.Time() ) # 1sr: wrt, 2nd: frame
    c_R_n = tf.transformations.quaternion_matrix(c_q_n)[0:3, 0:3]
    c_t_n_skew = skew_mat(c_t_n)

    c_V_n = np.zeros((6, 6))
    c_V_n[0:3, 0:3] = c_R_n
    c_V_n[3:6, 3:6] = c_R_n
    c_V_n[0:3, 3:6] = np.matmul(c_t_n_skew, c_R_n)

    V = c_V_n

    # Publisher for the sequence of visual features. Indeed it is just the current value of the sequence
    vis_ref_seq = rospy.Publisher("/reference_governor/reference_features", VisualFeatures, queue_size=10, latch=True) #/visual_reference_governor/visual_reference_sequence
    
    # Publisher for the sequence of visual features
    des_feat_pub = rospy.Publisher("/reference_governor/desired_features", VisualFeatures, queue_size=10, latch=True) #/visual_reference_governor/visual_reference_sequence
    
    # Publisher for computation time
    comp_time_pub = rospy.Publisher("/reference_governor/time", Float32, queue_size=10, latch=True)

    # Discretization sampling time  -> NO NEED THIS ANYMORE: DONE WITH qp_control_rate ######################################## TODO
    T = rospy.get_param('regovis_system_sampling_time', default=0.001) # s
    #qp_control_rate = rospy.get_param('/ros_server_node/rate')
    #T = 1./qp_control_rate # Assumption: the QP control period match with the discretization time of the system

    # Sampling time for the MPC (according to real-time)
    TMPC = rospy.get_param('regovis_MPC_sampling_time', default=0.125)
    print('TMP: ', TMPC)

    # MPC prediction horizon
    Np = rospy.get_param('regovis_preview_window_size', default=15)
    print('Np: ', Np)

    # c parameter: it is such that TMPC = c * T
    c = int(TMPC/T) 

    rate = rospy.Rate(1./TMPC) 

    first_time = True

    # Get the VS lambda parameter
    qp_gain = 0.0001 ######################################### TODO
    print('VS gain: ', qp_gain)
    
    vs_gain = qp_gain / T # lambda from the stack / qp_control period

    # Time log
    time_log_file = '/tmp/reference_governor_panda_times.csv'
    df = pd.DataFrame({'t_signals': [], 't_mpc': [], 't_loop': [], 't_matrixes':[], 'T_MPC':[],'gamma_x': [], 'gamma_deltax':[], 'x_seq' : [], 's_star_seq' : [], 'time':[]})
    df.to_csv(time_log_file,index=True)

    # Wait for sensors once than subscribe it with a callback############################### TODO do we need this?
    # global data 
    # rospy.Subscriber("gazebo/model_states", ModelStates, model_states_cb)
    # data = rospy.wait_for_message("gazebo/model_states", ModelStates, timeout=None)
    # rospy.loginfo("RefGov: Got model_states message")
    
    # Get the current visual feature in normalized coordinates
    global visual_features #################################################################### TODO # why here?
    rospy.Subscriber("cartesian/visual_servoing_d435_color_optical_frame/features", VisualFeatures, visual_features_cb)
    visual_features = rospy.wait_for_message("cartesian/visual_servoing_d435_color_optical_frame/features", VisualFeatures, timeout=None)
    rospy.loginfo("RefGov: Got visual_features message")

    # To allow to properly log data
    np.set_printoptions(threshold=sys.maxsize)

    q = np.zeros((7,))

    # RG LOOP
    while not rospy.is_shutdown():
        
        # Get initial time
        time_cycle_start = rospy.get_time() # in seconds # 
        time_cycle_start_wall = time.time() # in seconds # 
        
        # Get joint values from robot
        vals_pos = robot.joint_angles()
        for i, j in enumerate(joint_names):
                q[i] = vals_pos[j]
                
        # End of the SENSING
        t_robot_sense = time.time() - time_cycle_start_wall

        # Start of MATRIX COMPUTATION 
        t_matrix_start = time.time()
            
        # Compute the jacobian of the robot
        o_J_n = robot.zero_jacobian()

        # Note that, to be fully formally correct, I would need the Jacobian in the
        # end-effector frame, see e.g. https://hal.inria.fr/inria-00351899/document
        # However ArmInterface gives me the Jacobian in the robot base frame,
        # thus I need to properly handle this with a proper transformation matrix

        _, n_q_o = tf_listener.lookupTransform('panda_hand', 'base',rospy.Time())
        n_R_o = tf.transformations.quaternion_matrix(n_q_o)[0:3, 0:3]

        n_V_o = np.zeros((6, 6))
        n_V_o[0:3, 0:3] = n_R_o
        n_V_o[3:6, 3:6] = n_R_o

        Jn = np.matmul(n_V_o, o_J_n)
        VJn = np.matmul(V, Jn)
        
        # Fill the feature vector (stay in normalized coordinates)
        features, depths = getFeaturesAndDepths(visual_features)
        if not len(features)>0:
            rate.sleep()
            rospy.loginfo('RefGov: No features')
            continue
        
        # Compute the interaction matrix (image Jacobian)
        L = visjac_p(intrinsic, features, depths)
        
        # Compute the task Jacobian: L * V * J_camera
        J = np.matmul(L, VJn)
        
        # NO NEED FOR THE ROBOT MANIPULATOR ################################################################ TODO : to be checked
        #J_const = np.zeros((7,7))

        # Build the matrices for the MPC 
        # print(VJn)
        # print(L)
        # print(J)
        # print(J_const)
        # print('##########')
        A, B, C, D = compute_system_matrices(J,vs_gain,T,c) # TODO: removed J_const

        # print(A)
        # print(B)
        # print(C)
        # print(D)
        
        # End of MATRIX COMPUTATION
        t_matrix = time.time() - t_matrix_start
        #print('Matrices computation time : ', time.time()-t)

        # Useful dimensions
        if first_time:
            nx = A.shape[0]
            ns = B.shape[1]
            nq = nx - ns

        # MPC state read from the sensors
        #y_step = np.r_[features.reshape(ns,1),q.reshape(nq,1),q_dot.reshape(nq,1)] 
        x_step = np.r_[features.reshape(ns,1), q.reshape(nq,1)] 
        
        if first_time: # MCP initialization 
            
            print('Initializing MPC...')

            first_time = False
            
            # Set (and publish) the desired features as the first detected ones
            des_features = features
            desired_features_msg = fill_visualFeatures_msg(des_features)
            desired_features_msg.header.stamp = rospy.Time.now()
            des_feat_pub.publish(desired_features_msg)

            # At the beginning send the desired features as reference
            reference_features_msg = fill_visualFeatures_msg(des_features)
            reference_features_msg.header.stamp = rospy.Time.now()
            vis_ref_seq.publish(reference_features_msg)
            
            # Initialization of the solution
            s_star_step = des_features
            
            # MPC weights
            sparse = True
            if sparse: 
                
                low_lambda = True

                if low_lambda:
                    # low lambda
                    Q1 = 100*scipy.sparse.eye(ns)  # s_d - s
                    Q2 = 1*scipy.sparse.eye(ns)   # s_d - s_star
                else:                
                    # high lambda 
                    Q1 = 0.1*scipy.sparse.eye(ns)  # s_d - s
                    Q2 = 10.0*scipy.sparse.eye(ns)   # s_d - s_star
                    
                QDg = 0.0*scipy.sparse.eye(ns)   # s_star_i - s_star_i-1
                Q4 = 1*scipy.sparse.eye(nq)    # q_dot_i 
                Q5 = 1e4 #1e4    
            else:
                Q1 = 1.0*np.eye(ns)  # s_d - s
                Q2 = 1.0*np.eye(ns)   # s_d - s_star
                Q4 = 0.0001*np.eye(nq)    # q_dot_i - q_dot_i-1
                QDg = 0.0001*np.eye(ns)   # s_star_i - s_star_i-1
                Q5 = 1e2 #1e4    
            
            # MPC constraints on q_dot -> to be handled as delta q
            # https://frankaemika.github.io/docs/control_parameters.html

            q_dot_upper = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
            q_dot_lower = -q_dot_upper
            q_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
            q_upper = np.array([2.8973,	1.7628,	2.8973,	-0.0698, 2.8973, 3.7525, 2.8973])
            
            # Feature position limits
            s_safety_margin = 25 # pixels
            s_min = convert_to_normalized(s_safety_margin*np.ones(ns),intrinsic)
            s_max = convert_to_normalized(np.array([intrinsic['width']-s_safety_margin, intrinsic['height']-s_safety_margin,
                                                    intrinsic['width']-s_safety_margin, intrinsic['height']-s_safety_margin,
                                                    intrinsic['width']-s_safety_margin, intrinsic['height']-s_safety_margin,
                                                    intrinsic['width']-s_safety_margin, intrinsic['height']-s_safety_margin]), 
                                                    intrinsic)

            #print('S_bound_min: ', s_min)
            #print('S_bound_max: ', s_max)
            #y_min = np.r_[s_min, q_lower, q_dot_lower]
            #y_max = np.r_[s_max, q_upper, q_dot_upper]
            x_min = np.r_[s_min, q_lower]
            x_max = np.r_[s_max, q_upper]
            
            # Bound on s_dot -> to be transformed in delta s
            s_dot_limit = 570 # pixel per seconds 
            s_dot_bound = np.abs( convert_to_normalized(s_dot_limit*np.ones(ns),intrinsic)) 
            #print('S_dot_bound: ', s_dot_bound)
            #delta_x_min = np.r_[-1e6*np.ones(ns), TMPC * q_dot_lower]
            #delta_x_max = np.r_[ 1e6*np.ones(ns), TMPC * q_dot_upper]
            delta_x_min = np.r_[-TMPC*s_dot_bound, TMPC * q_dot_lower]
            delta_x_max = np.r_[ TMPC*s_dot_bound, TMPC * q_dot_upper]
            
            # print(x_min)
            # print(x_max)
            # print(delta_x_min)
            # print(delta_x_max)
            # Setup Reference Governor
            #print('Setup')
            K = MPCController(
                A=A, B=B, 
                s_d=features, 
                Np=Np, 
                x_min=x_min, x_max=x_max, delta_x_min=delta_x_min, delta_x_max=delta_x_max,
                Q1=Q1, Q2=Q2, Q3=None, Q4=Q4, Q5=Q5, QDg=QDg,
                x_zero=x_step.flatten(), x_minus1=x_step.flatten(), s_minus1=features)

            #print('Setup2')                        
            K.setup() 
            
            print('MPC initialized.')

        # Compute the new reference 
        t_mpc_start = rospy.get_time()
        t_mpc_start_wall = time.time()
        
        # Update the MPC with the current values
        K.update(x_zero = x_step.flatten(), A=A, B=B, s_d=des_features, solve=True)
        print('MPC updated')

        #print('MPC time:', rospy.get_time() - t_mpc_start)
        #print('MPC time wall:', time.time() - t_mpc_start_wall)
        #x_step_minus1 = x_step
        #features_minus1 = features

        #Get the MPC ouptut
        s_star_step = K.output()
        s_star_seq, x_seq, gamma_x_seq, gamma_deltax_seq = K.full_output()

        t_mpc_wall = time.time() - t_mpc_start_wall
        
        #print("New reference computed: ", s_star_step)

        # Publish the new reference
        # End of the SENSING
        t_msg_start = time.time()

        reference_features_msg = fill_visualFeatures_msg(s_star_step)
        reference_features_msg.header.stamp = rospy.Time.now()

        vis_ref_seq.publish(reference_features_msg)
        
        t_msg = time.time() - t_msg_start

        elapsed_time =   rospy.get_time() - time_cycle_start
        elapsed_time_wall =   time.time() - time_cycle_start_wall
        #print('Elapsed time: ', elapsed_time)
        #print('Elapsed time wall: ', elapsed_time_wall)
        # Log time info

        df = pd.DataFrame({
            't_signals':[t_robot_sense+t_msg],
            't_mpc': [t_mpc_wall], 
            't_loop': [elapsed_time_wall], 
            't_matrixes': [t_matrix], 
            'T_MPC': [TMPC],
            'gamma_x': [np.max(np.abs(gamma_x_seq[0,:]))], # I actually save the sum of the current gamma_x
            'gamma_deltax': [np.max(np.abs(gamma_deltax_seq[0,:]))], # I actually save the sum of the current gamma_x
            'x_seq' : [ np.array_repr(x_seq[:,:]) ],
            's_star_seq' : [ np.array_repr(s_star_seq[:,:]) ],
            'time' : [rospy.get_rostime().secs]
             })

        df.to_csv(time_log_file,mode='a',index=True,header=False)    
        
        #comp_time_pub.publish(time_msg)

        rate.sleep()