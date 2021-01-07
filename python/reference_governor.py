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

		# Build the matrix for the MPC

		# A_bar = I - lambda * T * J * (J*P)^+ 

		# B_bar = lambda * T * J * (J*P)^+ 

		# A = A_bar^c

		# B = Sum_{j=0}^{c-1} A^j * B_bar

		# C = [I; -lambda *(J*P)^+]

		# D = [0; lambda *(J*P)^+]

		# For matrix exponential:

		A = np.random.rand(3,3)
		A_pow2 = np.linalg.matrix_power(A,2)
		A_pow2_ = np.matmul(A,A)
		
		# A_pow2 = A_pow2_

        rospy.spin()




