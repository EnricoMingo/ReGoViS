#!/usr/bin/env python
from __future__ import division
import rospy
import xbot_interface.config_options as xbot_opt
import xbot_interface.xbot_interface as xbot
from gazebo_msgs.msg import ModelStates
from cartesian_interface.pyci_all import *
from opensot_visual_servoing.msg import VisualFeature, VisualFeatures
from sensor_msgs.msg import CameraInfo
import numpy as np
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
    
    t = time.time()
    for ind in range(1,c):

        B_list.append(np.matmul(A_bar,B_list[-1]))
        B = B_list[-1] + B
        A = np.matmul(A,A_bar)
      
    print(time.time() - t)
    C = C_bar
    D = D_bar

    print(A)
    print('---')
    print(B)
    print('---')
          
    # Maybe it is faster:

    t = time.time()
    A = np.linalg.matrix_power(A_bar,c)

    B = np.zeros((n_features,n_features))
    for j in range(0,c-1):
        A_bar_j = np.linalg.matrix_power(A_bar,j) 
        B = B + np.matmul(A_bar_j,B_bar)

    print(time.time()-t)
    print(A)
    print('---')
    print(B)
    print('---')
    
    print('---')
    
    print('---')
    
    return A, B, C, D  

if __name__ == '__main__':
    rospy.init_node("reference_governor")

    robot = createRobot()

    # Subscriber to the camera_info topic
    camera_info_msg = rospy.wait_for_message("/camera/rgb/camera_info", CameraInfo, timeout=None)
    intrinsic = get_intrinsic_param(camera_info_msg)

    # Publisher for the sequence of visual features
    vis_ref_seq = rospy.Publisher("/visual_reference_governor/visual_reference_sequence", VisualFeatures, queue_size=10, latch=True)

    # TODO: we need to set these parameters somewhere
    # sampling time
    T = 0.001   
    # sampling time for the MPC
    TMPC = 0.1  
    # c parameter: it is such that TMPC = c * T
    c = int(TMPC/T) 

    rate = rospy.Rate(1/TMPC) 

    while not rospy.is_shutdown():

        # Sense robot state
        data = rospy.wait_for_message("gazebo/model_states", ModelStates, timeout=None)
        base_pose, base_twist = extract_base_data(data)

        robot.sense()
        # TODO: need to get the robot FB by using specific code (now not available)
        #robot.model().setFloatingBaseState(base_pose, base_twist)

        # TODO: need to get (and not set) the following parameters
        n_state = 34 
        n_features = 8 
        vs_gain = 1.5 # lambda from the stack 
        
        # TODO: compute the jacobian of the robot (nees specific code for this?). Now set as random matrix
        J = np.random.rand(n_features,n_state) # HARD CODED JACOBIAN, TO DEBUG THE CODE

        # Compute interaction matrix # TODO DOUBLE CHECK
        visual_features = rospy.wait_for_message("/image_processing/visual_features", VisualFeatures, timeout=None)
        features, depths = getFeaturesAndDepths(visual_features)
        L = visjac_p(intrinsic, features, depths)

        # Build the matrices for the MPC
        
        # Identity matrix
        I = np.eye(n_features)

        # Null prokector matrix of the equality constraints # TODO: now identity, to be get from robot code?
        P = np.eye(n_state)

        # J * P
        JP = np.matmul(J,P)
        
        # (J*P)^+
        JP_pinv = np.linalg.pinv(JP)
        
        # lambda * T * (J*P)^+
        lambda_T_J_JP_pinv = vs_gain * T * np.matmul(J,JP_pinv)  
        
        A_bar = I - lambda_T_J_JP_pinv
        B_bar = lambda_T_J_JP_pinv
		        
        lambda_JP_pinv = vs_gain * JP_pinv
        C_bar = np.block([
            [ np.eye(n_features) ],
            [-lambda_JP_pinv     ]
            ])

        D_bar = np.block([
            [ np.zeros((n_features,n_features)) ],
            [ lambda_JP_pinv                    ]
        ])
        
        # Discretizing to get A, B, C and D matrices for the MPC
        A, B, C, D =  discretizeMPC(A_bar,B_bar,C_bar,D_bar,c) 
        
        # Simulating a sequence of references (to be able to publish something)

        # Publish the visual features in x-y format
        visualFeatures_msg = VisualFeatures()

        # Number of features TODO: to be better computed
        n_points = 4

        # Preview window size: TODO: to be properly fixed
        Np = 10
        print(features)
        for n in range(0,Np):

            dummy_offset = ((n+1)/Np) * 50
            

            visualFeature_msg = VisualFeature()
            
            xx = features[0] * intrinsic['fx'] + intrinsic['cx'] 
            yy = features[1] * intrinsic['fy'] + intrinsic['cy']


            visualFeature_msg.x = xx + dummy_offset
            visualFeature_msg.y = yy + dummy_offset 
            visualFeature_msg.Z = 0 # TODO what should we put here??? The one at the final destination?
            visualFeature_msg.type = visualFeature_msg.POINT

            visualFeatures_msg.features.append(visualFeature_msg)

            visualFeature_msg = VisualFeature()
        
            xx = features[2] * intrinsic['fx'] + intrinsic['cx'] 
            yy = features[3] * intrinsic['fy'] + intrinsic['cy']

            visualFeature_msg.x = xx + dummy_offset
            visualFeature_msg.y = yy + dummy_offset 
            visualFeature_msg.Z = 0 # TODO what should we put here??? The one at the final destination?
            visualFeature_msg.type = visualFeature_msg.POINT

            visualFeatures_msg.features.append(visualFeature_msg)

            visualFeature_msg = VisualFeature()
            
            xx = features[4] * intrinsic['fx'] + intrinsic['cx'] 
            yy = features[5] * intrinsic['fy'] + intrinsic['cy']

            visualFeature_msg.x = xx + dummy_offset
            visualFeature_msg.y = yy + dummy_offset 
            visualFeature_msg.Z = 0 # TODO what should we put here??? The one at the final destination?
            visualFeature_msg.type = visualFeature_msg.POINT

            visualFeatures_msg.features.append(visualFeature_msg)

            visualFeature_msg = VisualFeature()

            xx = features[6] * intrinsic['fx'] + intrinsic['cx']
            yy = features[7] * intrinsic['fy'] + intrinsic['cy']

            visualFeature_msg.x = xx + dummy_offset
            visualFeature_msg.y = yy + dummy_offset 
            visualFeature_msg.Z = 0 # TODO what should we put here??? The one at the final destination?
            visualFeature_msg.type = visualFeature_msg.POINT

            visualFeatures_msg.features.append(visualFeature_msg)

        visualFeatures_msg.header.stamp = rospy.Time.now()

        vis_ref_seq.publish(visualFeatures_msg)

        rate.sleep()




