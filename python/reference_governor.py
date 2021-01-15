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
    
    for ind in range(1,c): # from 1 to c-1
        B_list.append(np.matmul(A_bar,B_list[-1])) # TODO Shouldn't A_bar be elevated to ind?
        B = B_list[-1] + B
        A = np.matmul(A,A_bar)
      
    print(time.time() - t)
    C = C_bar
    D = D_bar

    print(A)
    print('---')
    print(B)
    print('---')
          
    # This is another option:

    # For j = 0    
    A = A_bar 
    B = B_bar 
    t = time.time()
    # For j from 1 to c-1
    for j in range(1,c):
        B = B + np.matmul(A,B_bar)
        A = np.matmul(A,A_bar)
    
    print(time.time()-t)
   
    print(A)
    print('---')
    print(B)
    print('---')
    
    print('---')
    print(np.linalg.matrix_power(A_bar,c))
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

    # TODO: Time parameters: do we need to set these parameters somewhere else?
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
        # TODO: need to set the robot FB by using specific code (now not available)
        #robot.model().setFloatingBaseState(base_pose, base_twist)

        # TODO: should be taken from the cartesian interface
        vs_gain = 1.5 # lambda from the stack 
        
        # Compute the jacobian of the robot
        J_camera = robot.model().getJacobian('camera_link') 

        # Compute the interaction matrix (image Jacobian)# TODO DOUBLE CHECK
        visual_features = rospy.wait_for_message("/image_processing/visual_features", VisualFeatures, timeout=None)
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
            
        # Build the matrices for the MPC
        
        _, n_state = np.shape(J_camera)
        n_features, _ = np.shape(L)

        # Identity matrix
        I_n = np.eye(n_state)

        # Null prokector matrix of the equality constraints 
        J_const_pinv = np.linalg.pinv(J_const)
        P = I_n - np.matmul(J_const_pinv,J_const)

        # J * P
        JP = np.matmul(J,P)
        
        # (J*P)^+
        JP_pinv = np.linalg.pinv(JP)
        
        # lambda * T * (J*P)^+
        lambda_T_J_JP_pinv = vs_gain * T * np.matmul(J,JP_pinv)  
        
        I_f = np.eye(n_features)

        A_bar = I_f - lambda_T_J_JP_pinv
        B_bar = lambda_T_J_JP_pinv
		        
        lambda_JP_pinv = vs_gain * JP_pinv
        C_bar = np.block([
            [ I_f ],
            [-lambda_JP_pinv     ]
            ])

        D_bar = np.block([
            [ np.zeros((n_features,n_features)) ],
            [ lambda_JP_pinv                    ]
        ])
        
        # Discretizing to get A, B, C and D matrices for the MPC
        A, B, C, D =  discretizeMPC(A_bar,B_bar,C_bar,D_bar,c) 
        
        # Simulating a sequence of references (to be able to publish something)

        # Prepare the visual features messages (for the reference sequence)
        visualFeatures_msg = VisualFeatures()

        # Number of features TODO: to be better computed
        n_points = int(n_features/2)

        # Preview window size: TODO: to be properly fixed
        Np = 20
        print(features)

        # This loop simulates a sequene of references, which will be computed by an MPC
        offset = 50
        dummy_des_feat = np.array([
                    intrinsic['cx']-offset, intrinsic['cy']-offset, intrinsic['cx']+offset, intrinsic['cy']-offset,
                    intrinsic['cx']+offset, intrinsic['cy']+offset, intrinsic['cx']-offset, intrinsic['cy']+offset])
        

        for n in range(0,Np):

            alpha = (n+1)/Np

            visualFeature_msg = VisualFeature()
            
            xx = features[0] * intrinsic['fx'] + intrinsic['cx'] 
            yy = features[1] * intrinsic['fy'] + intrinsic['cy']

            visualFeature_msg.x = (1-alpha) * xx + alpha * dummy_des_feat[0] 
            visualFeature_msg.y = (1-alpha) * yy + alpha * dummy_des_feat[1] 
            visualFeature_msg.Z = 0 # TODO what should we put here??? The one at the final destination?
            visualFeature_msg.type = visualFeature_msg.POINT

            visualFeatures_msg.features.append(visualFeature_msg)

            visualFeature_msg = VisualFeature()
        
            xx = features[2] * intrinsic['fx'] + intrinsic['cx'] 
            yy = features[3] * intrinsic['fy'] + intrinsic['cy']

            visualFeature_msg.x = (1-alpha) * xx + alpha * dummy_des_feat[2]
            visualFeature_msg.y = (1-alpha) * yy + alpha * dummy_des_feat[3]
            visualFeature_msg.Z = 0 # TODO what should we put here??? The one at the final destination?
            visualFeature_msg.type = visualFeature_msg.POINT

            visualFeatures_msg.features.append(visualFeature_msg)

            visualFeature_msg = VisualFeature()
            
            xx = features[4] * intrinsic['fx'] + intrinsic['cx'] 
            yy = features[5] * intrinsic['fy'] + intrinsic['cy']

            visualFeature_msg.x = (1-alpha) * xx + alpha * dummy_des_feat[4] 
            visualFeature_msg.y = (1-alpha) * yy + alpha * dummy_des_feat[5] 
            visualFeature_msg.Z = 0 # TODO what should we put here??? The one at the final destination?
            visualFeature_msg.type = visualFeature_msg.POINT

            visualFeatures_msg.features.append(visualFeature_msg)

            visualFeature_msg = VisualFeature()

            xx = features[6] * intrinsic['fx'] + intrinsic['cx']
            yy = features[7] * intrinsic['fy'] + intrinsic['cy']

            visualFeature_msg.x = (1-alpha) * xx + alpha * dummy_des_feat[6]
            visualFeature_msg.y = (1-alpha) * yy + alpha * dummy_des_feat[7] 
            visualFeature_msg.Z = 0 # TODO what should we put here??? The one at the final destination?
            visualFeature_msg.type = visualFeature_msg.POINT

            visualFeatures_msg.features.append(visualFeature_msg)

        visualFeatures_msg.header.stamp = rospy.Time.now()

        vis_ref_seq.publish(visualFeatures_msg)

        rate.sleep()




