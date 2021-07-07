#!/usr/bin/env python
# -*- coding: utf-8 -*-

# visualizer.py: rviz visualizer
# Author: Ravi Joshi
# Date: 2019/10/01

# import modules
import math
import rospy
from ros_openpose.msg import Frame
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3, Point
from visualization_msgs.msg import Marker, MarkerArray
from people_msgs.msg import PositionMeasurement, PositionMeasurementArray
from copy import deepcopy
import random


class RealtimeVisualization():
    def __init__(self, ns, frame_topic, skeleton_frame, id_text_size, id_text_offset, skeleton_hands, skeleton_line_width):
        self.ns = ns
        self.skeleton_frame = skeleton_frame
        self.id_text_size = id_text_size
        self.id_text_offset = id_text_offset
        self.skeleton_hands = skeleton_hands
        self.skeleton_line_width = skeleton_line_width

        # define a few colors we are going to use later on
        self.colors = [ColorRGBA(0.12, 0.63, 0.42, 1.00),
                       ColorRGBA(0.98, 0.30, 0.30, 1.00),
                       ColorRGBA(0.26, 0.09, 0.91, 1.00),
                       ColorRGBA(0.77, 0.44, 0.14, 1.00),
                       ColorRGBA(0.92, 0.73, 0.14, 1.00),
                       ColorRGBA(0.00, 0.61, 0.88, 1.00),
                       ColorRGBA(1.00, 0.65, 0.60, 1.00),
                       ColorRGBA(0.59, 0.00, 0.56, 1.00)]

        '''
        The skeleton is considered as a combination of line strips.
        Hence, the skeleton is decomposed into 3 LINE_STRIP as following:
            1) upper_body : from nose to mid hip
            2) hands : from left-hand wrist to right-hand wrist
            3) legs : from left foot toe to right foot toe

        See the link below to get the id of each joint as defined in Kinect v2
        src: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#keypoint-ordering
        Result for BODY_25 (25 body parts consisting of COCO + foot)
        const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
            { 0,      "Nose"},    {13,      "LKnee"}
            { 1,      "Neck"},    {14,     "LAnkle"}
            { 2, "RShoulder"},    {15,       "REye"}
            { 3,    "RElbow"},    {16,       "LEye"}
            { 4,    "RWrist"},    {17,       "REar"}
            { 5, "LShoulder"},    {18,       "LEar"}
            { 6,    "LElbow"},    {19,    "LBigToe"}
            { 7,    "LWrist"},    {20,  "LSmallToe"}
            { 8,    "MidHip"},    {21,      "LHeel"}
            { 9,      "RHip"},    {22,    "RBigToe"}
            {10,     "RKnee"},    {23,  "RSmallToe"}
            {11,    "RAnkle"},    {24,      "RHeel"}
            {12,      "LHip"},    {25, "Background"}


        hand output ordering
        src: https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/doc/media/keypoints_hand.png
        We are using 5 LINE_STRIP to draw a hand
        '''

        self.upper_body_ids = [0, 1, 8]
        self.hands_ids = [4, 3, 2, 1, 5, 6, 7]
        self.legs_ids = [22, 11, 10, 9, 8, 12, 13, 14, 19]
        self.body_parts = [self.upper_body_ids, self.hands_ids, self.legs_ids]

        # number of fingers in a hand
        self.fingers = 5

        # number of keypoints to denote a finger
        self.count_keypoints_one_finger = 5

        self.total_finger_kepoints = self.fingers * self.count_keypoints_one_finger

        # write person id on the top of his head
        self.nose_id = 1
        self.leftH_id = 7
        self.rightH_id = 4

        # define a publisher to publish the 3D skeleton of multiple people
        self.skeleton_pub = rospy.Publisher(self.ns, MarkerArray, queue_size=1)

        self.position_pub = rospy.Publisher("/position",PositionMeasurementArray, queue_size=10)
        #self.circle_pub = rospy.Publisher("/circle",PositionMeasurementArray, queue_size=10)
        self.human_pub = rospy.Publisher("/human_localization", Marker, queue_size=10)
        self.group_pub = rospy.Publisher("/group_localization", MarkerArray, queue_size=10)
        self.fire_pub = rospy.Publisher("/fire", MarkerArray, queue_size=10)
        # define a subscriber to retrive tracked bodies
        rospy.Subscriber(frame_topic, Frame, self.frame_callback)

    def spin(self):
        '''
        We enter in a loop and wait for exit whenever `Ctrl + C` is pressed
        '''
        rospy.spin()


    def create_marker(self, index, color, marker_type, size, time):
        '''
        Function to create a visualization marker which is used inside RViz
        '''
        marker = Marker()
        marker.id = index
        marker.ns = self.ns
        marker.color = color
        marker.action = Marker.ADD
        marker.type = marker_type
        marker.scale = Vector3(size, size, size)
        marker.header.stamp = time
        marker.header.frame_id = self.skeleton_frame
        marker.lifetime = rospy.Duration(1)  # 1 second
        return marker

    def create_position(self, x, y, z, time):
        
        pos = PositionMeasurement()
        
        pos.header.stamp = time
        pos.name = "person"
        pos.pos.x = x
        pos.pos.y = y
        pos.pos.z = z
        pos.header.frame_id = self.skeleton_frame
        pos.reliability = 0.0
        pos.initialization = 1
        pos.covariance[0] = 0.04
        pos.covariance[1] = 0.0
        pos.covariance[2] = 0.0
        pos.covariance[3] = 0.0
        pos.covariance[4] = 0.04
        pos.covariance[5] = 0.0
        pos.covariance[6] = 0.0
        pos.covariance[7] = 0.0
        pos.covariance[8] = 0.04
        
        return pos

    def create_human(self, x, y, z, time, counter):
        marker = Marker()
        marker.header.frame_id = self.skeleton_frame
        marker.header.stamp = time
        marker.ns = "my_namespace"
        marker.type = marker.MESH_RESOURCE
        marker.id = counter
        marker.action = marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.x = 1.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.mesh_use_embedded_materials = True
        marker.mesh_resource = "file:///home/kwan/catkin_ws/src/visualization_tutorials/rviz_plugin_tutorials/media/stand.dae"
        #marker.color.a = 1.0 # Don't forget to set the alpha!
        #marker.color.r = 1.0
        #marker.color.g = 0.0
        #marker.color.b = 0.0
        # only if using a MESH_RESOURCE marker type:
        mark = deepcopy(marker)
        mark.color.a = 1.0
        mark.color.r = random.random()
        #mark.color.g = random.random()
        #mark.color.b = random.random()
        return mark

    def create_fire(self, x, y, z, time, counter):
        marker = Marker()
        marker.header.frame_id = self.skeleton_frame
        marker.header.stamp = time
        marker.ns = "my_namespace"
        marker.type = marker.SPHERE
        marker.id = counter
        marker.action = marker.ADD
        marker.pose.position.x = x/10
        marker.pose.position.y = y/10
        marker.pose.position.z = z/10
        marker.pose.orientation.x = 1.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = random.random()
        marker.color.g = random.uniform(0.0,0.2)
        marker.mesh_use_embedded_materials = True
        return marker

    def isValid(self, bodyPart):
        '''
        When should we consider a body part as a valid entity?
        We make sure that the score and z coordinate is a positive number.
        Notice that the z coordinate denotes the distance of the object located
        in front of the camera. Therefore it must be a positive number always.
        '''
        return bodyPart.score > 0 and not math.isnan(bodyPart.point.x) and not math.isnan(bodyPart.point.y) and not math.isnan(bodyPart.point.z) and bodyPart.point.z > 0


    def frame_callback(self, data):
        '''
        This function will be called everytime whenever a message is received by the subscriber
        '''
        marker_counter = 0
        person_counter = 0
        marker_array = MarkerArray()
        Group_Array = MarkerArray()
        fire_array = MarkerArray()
        pos_array = PositionMeasurementArray()
        pos_array.header.frame_id = self.skeleton_frame

        '''

        image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")

        datum = op.Datum()
        datum.cvInputData = image
        self.op_wrapper.emplaceAndPop([datum])

        pose_kp = datum.poseKeypoints
        pose = ""
        # Set number of people detected
        if pose_kp.shape == ():
            num_persons = 0
            body_part_count = 0
        else:
            num_persons = pose_kp.shape[0]
            body_part_count = pose_kp.shape[1]
            kp = pose_kp[0][1:9]
            kp = np.delete(kp, 2, 1)
            kp = kp.flatten()
            kp = kp.reshape(1, -1)

            pose = mlp.predict(kp)[0]
            print(pose)
            pub.publish(pose)

        '''


        for person in data.persons:
            now = rospy.Time.now()
            pos_array.header.stamp = rospy.Time.now()
            marker_color = self.colors[person_counter % len(self.colors)]

            # the body_marker contains three markers as mentioned already
            # 1. upper body 2. hands 3. legs
            body_marker = [self.create_marker(marker_counter + idx, marker_color, Marker.LINE_STRIP, self.skeleton_line_width, now) for idx in range(len(self.body_parts))]
            marker_counter += len(self.body_parts)

            # assign 3D positions to each body part
            # make sure to consider only valid body parts
            for index, body_part in enumerate(self.body_parts):
                body_marker[index].points = [person.bodyParts[idx].point for idx in body_part if self.isValid(person.bodyParts[idx])]

            marker_array.markers.extend(body_marker)

            if self.skeleton_hands:
                left_hand = [self.create_marker(marker_counter + idx, marker_color, Marker.LINE_STRIP, self.skeleton_line_width, now) for idx in range(self.fingers)]
                marker_counter += self.fingers

                right_hand = [self.create_marker(marker_counter + idx, marker_color, Marker.LINE_STRIP, self.skeleton_line_width, now) for idx in range(self.fingers)]
                marker_counter += self.fingers

                keypoint_counter = 0
                for idx in range(self.total_finger_kepoints):
                    strip_id = idx / self.count_keypoints_one_finger
                    temp_id = idx % self.count_keypoints_one_finger
                    if temp_id == 0:
                        point_id = temp_id
                    else:
                        keypoint_counter += 1
                        point_id = keypoint_counter

                    leftHandPart = person.leftHandParts[point_id]
                    rightHandPart = person.rightHandParts[point_id]
                    if self.isValid(leftHandPart):
                        left_hand[strip_id].points.append(leftHandPart.point)

                    if self.isValid(rightHandPart):
                        right_hand[strip_id].points.append(rightHandPart.point)
                marker_array.markers.extend(left_hand)
                marker_array.markers.extend(right_hand)

            person_id = self.create_marker(marker_counter, marker_color, Marker.TEXT_VIEW_FACING, self.id_text_size, now)
            marker_counter += 1
            # assign person id and 3D position
            person_id.text = str(person_counter)
            nose = person.bodyParts[self.nose_id]
            leftHand = person.bodyParts[self.leftH_id]
            rightHand = person.bodyParts[self.rightH_id]
            if self.isValid(nose):
                person_id.pose.position = Point(nose.point.x, nose.point.y + self.id_text_offset, nose.point.z)
                marker_array.markers.append(person_id)
            PP = self.create_position(nose.point.x, nose.point.y, nose.point.z, now)
            HH = self.create_human(nose.point.x, nose.point.y, nose.point.z, now, person_counter)
            
            for i in range(10):
                for j in range(10):
                    for k in range(10):
                        fire_array.markers.append(self.create_fire((nose.point.x)+1+float(i), (nose.point.y)+float(k), (nose.point.z)+float(j), now,(i*100+j*10+k)))
            

            #self.fire_pub.publish(KK)
            pos_array.people.append(PP)
            Group_Array.markers.append(HH)
            #fire_array.markers.append(KK)
            
            #self.human_pub.publish(HH)

            # update the counter
            person_counter += 1

        # publish the markers
        self.skeleton_pub.publish(marker_array)
        self.position_pub.publish(pos_array)
        self.group_pub.publish(Group_Array)
        self.fire_pub.publish(fire_array)
        Group_Array.markers = []
        fire_array.markers= []


if __name__ == '__main__':
    # define some constants
    ns = 'visualization'

    # initialize ros node
    rospy.init_node('visualizer_node', anonymous=False)

    # read the parameters from ROS parameter server
    frame_topic = rospy.get_param('~pub_topic')
    skeleton_frame = rospy.get_param('~frame_id')
    id_text_size = rospy.get_param('~id_text_size')
    id_text_offset = rospy.get_param('~id_text_offset')
    skeleton_hands = rospy.get_param('~skeleton_hands')
    skeleton_line_width = rospy.get_param('~skeleton_line_width')

    # instantiate the RealtimeVisualization class
    visualization = RealtimeVisualization(ns, frame_topic, skeleton_frame, id_text_size, id_text_offset, skeleton_hands, skeleton_line_width)
    visualization.spin()
