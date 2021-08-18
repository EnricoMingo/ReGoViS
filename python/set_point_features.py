#!/usr/bin/env python

import rospy
from copy import copy
from opensot_visual_servoing.msg import VisualFeatures

desired_features_x = [-0.14306034865516204, -0.04182184615596208, -0.04345472522852982, -0.14306034865516204]
desired_features_y = [-0.11145350962968413, -0.10818626599060409, -0.011802578637742468, -0.015069822276822522]

def callback(data):
    refs = copy(data)

    i = 0
    for point in refs.features:
        point.x = desired_features_x[i]
        point.y = desired_features_y[i]
        i += 1

    pub = rospy.Publisher("/reference_governor/desired_features_input", VisualFeatures, queue_size=10)
    #pub = rospy.Publisher("/cartesian/visual_servoing_D435i_camera_color_optical_frame/desired_features", VisualFeatures, queue_size=10)
    rate = rospy.Rate(10)  # 10hz
    for i in range(5):
        pub.publish(refs)
        rate.sleep()

    rospy.signal_shutdown("new reference were published")

def publisher():
    rospy.init_node('reference_points_pub', anonymous=True)

    rospy.Subscriber("/cartesian/visual_servoing_D435i_camera_color_optical_frame/reference_features", VisualFeatures, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    publisher()
