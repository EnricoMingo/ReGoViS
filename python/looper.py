#!/usr/bin/env python

import rospy
import xbot_interface.config_options as xbot_opt
import xbot_interface.xbot_interface as xbot
import tf2_ros
import geometry_msgs.msg
from cartesian_interface.pyci_all import *
import os

from sensor_msgs.msg import JointState

def createRobot():
    opt = xbot_opt.ConfigOptions()

    urdf = rospy.get_param('robot_description')
    srdf = rospy.get_param('robot_description_semantic')

    opt = xbot_opt.ConfigOptions()
    opt.set_urdf(urdf)
    opt.set_srdf(srdf)
    opt.generate_jidmap()
    opt.set_bool_parameter('is_model_floating_base', True)
    opt.set_string_parameter('model_type', 'RBDL')
    opt.set_string_parameter('framework', 'ROS')
    return xbot.RobotInterface(opt)

if __name__ == '__main__':

    rospy.init_node("looper")

    ## LAUNCH coman.launch in coman_urdf
    os.system("roslaunch coman_urdf coman.launch")

    # Create the robot
    robot = createRobot()
    model = robot.model()

    # LOAD CONFIGURATION INTO A LIST
    data_time = pd.read_csv('reference_governor_times_for_looper.csv')                                   
    x_seq = data_time['x_seq']
    x_seq_np = eval("np."+x_seq[3]) # Take the last (in this case fourth element of the sequence)
    q_traj = x_seq_np[:,8:8+35] # Take the part related to the joint positions. Size: Np x 35

    #q_traj = []

    #######
    joint_states = JointState
    joint_states.names = robot.getEnabledJointNames()
    joint_pub = rospy.Publisher('joint_states', JointState, queue_size=10)

    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    t.header.frame_id = "world"
    t.child_frame_id = "base_link"

    rate = rospy.Rate(10)  # 10hz
 
    for q in q_traj:
    
        print(q)
        now = rospy.get_rostime()

        model.setJointPositions(q)
        model.update()

        joint_states.position = q
        joint_states.header.stamp = now

        T = model.getFloatingBasePose() # floating base pose in world frame
        t.header.stamp = now
        t.transform.translation.x = T.translation[0]
        t.transform.translation.y = T.translation[1]
        t.transform.translation.z = T.translation[2]
        q = T.quaternion
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        br.sendTransform(t)
        joint_pub.publish(joint_states)

        rospy.spin()
        rate.sleep()





