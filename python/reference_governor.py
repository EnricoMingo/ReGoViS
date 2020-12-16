#!/usr/bin/env python
import rospy
import xbot_interface.config_options as xbot_opt
import xbot_interface.xbot_interface as xbot

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

if __name__ == '__main__':
    rospy.init_node("reference_governor")

    robot = createRobot()

    while not rospy.is_shutdown():
        #1. sense robot state
        robot.sense()



        rospy.spin()




