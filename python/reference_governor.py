#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
import rospy
import xbot_interface.config_options as xbot_opt
import xbot_interface.xbot_interface as xbot
from xbot_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
from cartesian_interface.pyci_all import *
from opensot_visual_servoing.msg import VisualFeature, VisualFeatures
from sensor_msgs.msg import CameraInfo
import numpy as np
import scipy
import pandas as pd
import time
from mpcref.mpc import MPCController

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
 	       
        #uv = np.array(feat_vec[i:i+2])
        xy = np.array(feat_vec[i:i+2])

        # Take the depth
        Z = depths[max(0,i//2)]
        
        # Convert to normalized image-plane coordinates
        x = xy[0] #(uv[0] - intrinsic['cx']) / intrinsic['fx']
        y = xy[1] #(uv[1] - intrinsic['cy']) / intrinsic['fy']

        L_i = np.array([
            [1/Z, 0,   -x/Z, -x*y,     (1+x*x), -y],
            [0,   1/Z, -y/Z, -(1+y*y), x*y,      x]
        ])

        #L_i = - np.matmul(np.diag([intrinsic['fx'], intrinsic['fy']]), L_i)

        #L[i:i+2,:] = L_i
        L[i:i+2,:] = -L_i

    return L

# TODO: TO BE CHECKED: https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-ibvs.html
def convert_to_normalized(features_in_pixel,intrinsic):

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
    
    # TODO : this assumes that we work with points
    for k in range(0,len(data_in),2):

        visualFeature_msg = VisualFeature()
    
        visualFeature_msg.x = data_in[k]  
        visualFeature_msg.y = data_in[k+1] 
        visualFeature_msg.Z = 0.5 # TODO what should we put here??? The one at the final destination?
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

    # Publisher for the sequence of visual features # TODO: it is not a sequence (or should it be?)
    vis_ref_seq = rospy.Publisher("/reference_governor/reference_features", VisualFeatures, queue_size=10, latch=True) #/visual_reference_governor/visual_reference_sequence
    
    # Publisher for the sequence of visual features
    des_feat_pub = rospy.Publisher("/reference_governor/desired_features", VisualFeatures, queue_size=10, latch=True) #/visual_reference_governor/visual_reference_sequence
    
    #### TODO: set from rosparam 

    # Discretization sampling time
    T = 0.001 # s
    
    # Sampling time for the MPC
    TMPC = 0.3 # s

    # c parameter: it is such that TMPC = c * T
    c = int(TMPC/T) 

    # MPC prediction horizon
    Np = 15 #20

    ##### ##### #####
    
    rate = rospy.Rate(1./TMPC) 

    first_time = True

    while not rospy.is_shutdown():
    
        # Sense robot state: FB pose and velocity
        data = rospy.wait_for_message("gazebo/model_states", ModelStates, timeout=None)
        base_pose, base_twist = extract_base_data(data)
    
        # TODO: what exactily this function does?
        robot.sense()
        
        # TODO: need to set the robot FB by using specific code (now not available)
        robot.model().setFloatingBaseState(base_pose, base_twist)
        
        robot.model().update()

        q = robot.model().getJointPosition() 
        q_dot = robot.model().getJointVelocity()

        # TODO: should be taken from the cartesian interface, NEED TO BE CONSISTENT WITH THE STACK OF TASK 
        qp_control_period = 0.001 # TODO CAN WE TAKE FROM SOMEWHERE
            
        # Compute the jacobian of the robot
        J_camera = robot.model().getJacobian('camera_link') 

        # Compute the interaction matrix (image Jacobian)# TODO DOUBLE CHECK THAT IS IDENTICAL TO THE ONE USED IN OPENSOT. IT WOULD BE PERFERCT TO TAKE IT DIRECTLY FROM robot.model()
        #print('Waiting for visual_features topic...')
        visual_features = rospy.wait_for_message("cartesian/visual_servoing_camera_link/features", VisualFeatures, timeout=None) # TODO : /image_processing/visual_features
        #visual_features = rospy.wait_for_message("/image_processing/visual_features", VisualFeatures, timeout=None) # TODO : the topic name has to be a parameter
        features, depths = getFeaturesAndDepths(visual_features)
        
        L = visjac_p(intrinsic, features, depths)

        # Robot's camera frame to camera sensor twist transformation TODO to get from cartesian interface
        V = np.eye(6)

        # Compute the task Jacobian: L * V * J_camera
        J = np.matmul(L,V)
        J = np.matmul(J,J_camera)

        # Compute the Jacobian of the equality constraint (CMM in space, contacts Jacobian on earth)
        sim_on_earth = True # TODO it has to be a parameter, e.g. ROS parameter
        if sim_on_earth:
            print('SIM ON EARTH')
            # TODO compute in case of on earth sim.
            J_rf = robot.model().getJacobian('r_sole')
            J_lf = robot.model().getJacobian('l_sole')
            J_contacts = np.block([ [J_rf],[J_lf] ])
            J_com = robot.model().getCOMJacobian()
            J_const = np.block([
                [J_contacts],
                [J_com]
                ])
            vs_gain = 0.0005 / qp_control_period # lambda from the stack / qp_control period
        else: # in space
            print('SIM IN SPACE')
            J_const, _ = robot.model().getCentroidalMomentumMatrix()
            vs_gain = 0.001 / qp_control_period # lambda from the stack / qp_control period

        #rosservice call /cartesian/visual_servoing_camera_link/get_task_properties
        #rospy.wait_for_service('add_two_ints')
        
        # Build the matrices for the MPC # TODO check if the computation time is OK, use MARCO's function
        #t = time.time()
        A, B, C, D = compute_system_matrices(J,J_const,vs_gain,T,c)
        #print('Matrices computation time : ', time.time()-t)

        # Simulating a sequence of references (to be able to publish something) # TODO to be substituted with MPCpy function
        #offset = 50
        #dummy_des_feat_pixel = np.array([
        #            intrinsic['cx']-offset, intrinsic['cy']-offset, intrinsic['cx']+offset, intrinsic['cy']-offset,
        #            intrinsic['cx']+offset, intrinsic['cy']+offset, intrinsic['cx']-offset, intrinsic['cy']+offset
        #            ])
        #dummy_des_feat = convert_to_normalized(dummy_des_feat_pixel,intrinsic)
        #des_features = dummy_des_feat
        #print("desired features 2: ", des_features)
        
        if first_time: # MCP initialization 
            
            print('Initializing MPC...')

            first_time = False
            
            # Set (and publish) the desired features as the first detected ones
            des_features = features
            des_feat_pixel = convert_to_pixel(des_features,intrinsic) # TODO: probably, no need anymore
            desired_features_msg = fill_visualFeatures_msg(des_features) # TODO: in this case no need to convert in pixel!
            desired_features_msg.header.stamp = rospy.Time.now()
            des_feat_pub.publish(desired_features_msg)

            # At the beginning send the desired features as reference
            reference_features_msg = fill_visualFeatures_msg(des_features) # TODO: in this case also no need to convert in pixel!
            reference_features_msg.header.stamp = rospy.Time.now()
            vis_ref_seq.publish(reference_features_msg)

            # Useful dimensions
            nx = A.shape[0]
            ns = B.shape[1]
            nq = nx - ns
            
            # Initialization of the solution
            s_star_step = des_features
            
            # MPC weights
            Q1 = 100*np.eye(ns)  # s_d - s
            Q2 = 10*np.eye(ns)  # s_d - s_star
            Q4 = scipy.linalg.block_diag(np.zeros((ns+nq, ns+nq)), np.eye(nq))
            QDg = np.eye(ns)
            Q5 = 1e4    
            
            # MPC constraints 
            q_dot_lower = -robot.model().getVelocityLimits() 
            q_dot_upper = robot.model().getVelocityLimits() 
            q_lower, q_upper = robot.model().getJointLimits() 
          
            s_min = convert_to_normalized(0*np.ones(ns),intrinsic)
            s_max = convert_to_normalized(np.array([640,480,640,480,640,480,640,480]),intrinsic) # TODO: can I get the size of the image from somewhere?
            
            y_min = np.r_[s_min, q_lower, q_dot_lower]
            y_max = np.r_[s_max, q_upper, q_dot_upper]

            K = MPCController(A=A, B=B, C=C, D=D, Np=Np, 
                              Q1=Q1, Q2=Q2, Q3=None, Q4=Q4, Q5=Q5, QDg=QDg, 
                              y_min=y_min, y_max=y_max)
    
            K.setup()  # setup cvxpy problem

            print('MPC initialized.')

        # Update the MPC with the current values
        print("Computing a new reference...")
        
        # State MPC read from the sensors
        y_step = np.r_[features.reshape(ns,1),q.reshape(nq,1),q_dot.reshape(nq,1)] 
        x_step = np.r_[features.reshape(ns,1),q.reshape(nq,1)] 
        
        K.update(x0_val = x_step.flatten(),
                 s_d_val = des_features, 
                 s_star_minus1_val = s_star_step, 
                 y_minus1_val = y_step.flatten(), 
                 A=A, B=B, C=C, D=D)

        #Get the MPC ouptut
        s_star_step = K.output()
        
        print("New reference computed: ", s_star_step)

        # Publish the new reference
        s_star_pixel = convert_to_pixel(s_star_step,intrinsic) # TODO: probably, no need anymore
        reference_features_msg = fill_visualFeatures_msg(s_star_step)
        reference_features_msg.header.stamp = rospy.Time.now()

        vis_ref_seq.publish(reference_features_msg)

        rate.sleep()

        '''print('here')
        df = pd.DataFrame(J)
        df.to_csv('~/Desktop/A.csv',index=False)
        df = pd.DataFrame(J_const)
        df.to_csv('~/Desktop/B.csv',index=False)
        df = pd.DataFrame(x_MPC)
        df.to_csv('~/Desktop/s_q_q_dot.csv',index=False)'''
        
        '''
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
        '''
        
