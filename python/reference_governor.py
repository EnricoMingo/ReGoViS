#!/usr/bin/env python
import rospy
import xbot_interface.config_options as xbot_opt
import xbot_interface.xbot_interface as xbot
from gazebo_msgs.msg import ModelStates
from cartesian_interface.pyci_all import *
from opensot_visual_servoing.msg import VisualFeature, VisualFeatures
from sensor_msgs.msg import CameraInfo
import numpy as np

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
# TO HERE####################################################################

if __name__ == '__main__':
    rospy.init_node("reference_governor")

    robot = createRobot()

    camera_info_msg = rospy.wait_for_message("/camera/rgb/camera_info", CameraInfo, timeout=None)
    intrinsic = get_intrinsic_param(camera_info_msg)

    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        #1. sense robot state
        data = rospy.wait_for_message("gazebo/model_states", ModelStates, timeout=None)
        base_pose, base_twist = extract_base_data(data)

        robot.sense()
        robot.model().setFloatingBaseState(base_pose, base_twist)

        #2 compute interaction matrix
        visual_features = rospy.wait_for_message("/image_processing/visual_features", VisualFeatures, timeout=None)
        features, depths = getFeaturesAndDepths(visual_features)
        L = visjac_p(intrinsic, features, depths)
 


        # Build the matrix for the MPC

		# A_bar = I - lambda * T * J * (J*P)^+ 

		# B_bar = lambda * T * J * (J*P)^+ 

		# A = A_bar^c

		# B = Sum_{j=0}^{c-1} A^j * B_bar

		# C = [I; -lambda *(J*P)^+]

		# D = [0; lambda *(J*P)^+]

		# For matrix exponential:

        #A = np.random.rand(3,3)
        #A_pow2 = np.linalg.matrix_power(A,2)
        #A_pow2_ = np.matmul(A,A)
		
		# A_pow2 = A_pow2_
        rate.sleep()




