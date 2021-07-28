#!/usr/bin/env python

import rospy
from copy import copy
from opensot_visual_servoing.msg import VisualFeatures

desired_features_x = [0.0345, 0.174, 0.201, 0.060]
desired_features_y = [-0.330, -0.352, -0.218, -0.195]

def callback(data):
    refs = copy(data)

    i = 0
    for point in refs.features:
        point.x = desired_features_x[i]
        point.y = desired_features_y[i]
        i += 1

    pub = rospy.Publisher("/reference_governor/desired_features", VisualFeatures, queue_size=10)
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