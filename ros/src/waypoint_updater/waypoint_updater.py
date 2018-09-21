#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Member variables
        self.pose = None
        self.base_waypoints = None
        self.base_waypoints_2d = None
        self.base_waypoints_kdtree = None
        self.stop_waypoint_id = None

        self.loop()

    def loop(self):
        # Set loop rate at 50Hz
        rate = rospy.Rate(50)
        # Run until node is shutted down
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints_kdtree:
                # Find closest id waypoint to current pose using KDTree
                closest_id = self.get_closest_waypoint_id()
                # Publish {LOOKAHEAD_WPS} waypoints from this id
                self.publish_waypoints(closest_id)
            rate.sleep()

    def get_closest_waypoint_id(self):
        # Get current pose coordinates
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        # Query KDTree to retrieve closest waypoint from current pose
        closest_id = self.base_waypoints_kdtree.query([[x, y]], 1)[1]

        # Check if point is ahead of car by computing dot product between
        # vector {closest_wp-->prev_wp} and vector {closest_wp-->current_car_wp}
        closest_wp = np.array(self.base_waypoints_2d[closest_id])
        prev_wp = np.array(self.base_waypoints_2d[closest_id - 1])
        current_car_wp = np.array([x, y])

        if np.dot(closest_wp - prev_wp, current_car_wp - closest_wp) > 0:
            # Retrieve the next waypoint from the previous closest id
            # This is one should be the closest one in front of the car
            return (closest_id + 1) % len(self.base_waypoints_2d)

        return closest_id

    def publish_waypoints(self, closest_id):
        # Create message to publish
        lane = Lane()
        # Add header to lane message
        lane.header = self.base_waypoints.header
        # Add waypoints to lane message
        lane.waypoints = self.get_final_waypoints(closest_id)
        # Publish lane message
        self.final_waypoints_pub.publish(lane)

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # Base waypoints are only received once, save them
        self.base_waypoints = waypoints
        # Only keep 2D data
        self.base_waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
        # Use KDTree for quick nearest-neighbor lookup
        self.base_waypoints_kdtree = KDTree(self.base_waypoints_2d)

    def traffic_cb(self, msg):
        self.stop_waypoint_id = msg

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_final_waypoints(self, closest_id):
        red_ligh_ahead = (self.stop_waypoint_id > closest_id) and (self.stop_waypoint_id <= closest_id + LOOKAHEAD_WPS)

        # Check if red light detected in the next {LOOKAHEAD_WPS} waypoints
        if not red_ligh_ahead:
            return self.base_waypoints.waypoints[closest_id:closest_id + LOOKAHEAD_WPS]

        # Get first waypoint target velocity
        vel = self.base_waypoints.waypoints[closest_id].twist.linear.x

        # Compute decreasing velocity step
        decrease_vel_step = vel / float(LOOKAHEAD_WPS)

        # Update velocity for each waypoint
        waypoints = []
        for i in range(LOOKAHEAD_WPS):
            wp = self.base_waypoints.waypoints[closest_id+i]
            # Stop at red light
            if closest_id+i == self.stop_waypoint_id:
                wp.twist.linear.x = 0
            # Decrease target velocity
            else:
                wp.twist.linear.x -= decrease_vel_step
            # Add updated waypoint to the list
            waypoints.append(wp)

        return waypoints

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
