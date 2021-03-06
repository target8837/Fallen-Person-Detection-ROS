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
from geometry_msgs.msg import PoseStamped


from copy import deepcopy
import random
#import cv_bridge

from keras.models import model_from_yaml
import numpy
import os
import tensorflow as tf
import pandas as pd
from imutils.video import VideoStream
from imutils.video import FPS
import time


import sys

#tf.compat.v1.disable_eager_execution()
#tf.__version__
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
 
# load YAML and create model
yaml_file = open('/home/kwan/PoseTrain/model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("/home/kwan/PoseTrain/model.h5")
print("Loaded model from disk")

tf.__version__
time.sleep(2.0)
pastx = 0
pasty = 0
pastz = 0
spawn_cnt = 0
fire_cnt = 0
fire_sw = 0
fire_scale = 10
global_x = 0
global_y = 0
global_z = 0
pos_cnt = 0

drone_pub = rospy.Publisher("/drone/position", MarkerArray, queue_size=10000)
map_pub = rospy.Publisher("/create/map", MarkerArray, queue_size=100)
all = rospy.Publisher("/all", MarkerArray, queue_size=1000)

def create_depot(time):
    marker = Marker()
    marker.header.frame_id = "/map"
    marker.header.stamp = time
    marker.ns = "my_namespace"
    marker.type = marker.MESH_RESOURCE
    marker.id = 124
    marker.action = marker.ADD
    marker.pose.position.x = 1
    marker.pose.position.y = -7.5
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = -0.9
    marker.pose.orientation.w = 1.0
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.mesh_use_embedded_materials = True
    marker.mesh_resource = "file:///home/kwan/Downloads/Depot/meshes/Depot.dae"
    #marker.mesh_resource = "file:///home/kwan/sub_catkin/visualization_tutorials/rviz_plugin_tutorials/media/fired.dae"
    return marker

def create_cage(time):
    marker = Marker()
    marker.header.frame_id = "/map"
    marker.header.stamp = time
    marker.ns = "my_namespace"
    marker.type = marker.MESH_RESOURCE
    marker.id = 127
    marker.action = marker.ADD
    marker.pose.position.x = 1
    marker.pose.position.y = -2.5
    marker.pose.position.z = 10
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = -0.9
    marker.pose.orientation.w = 1.0
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.mesh_use_embedded_materials = True
    marker.mesh_resource = "file:///home/kwan/Downloads/cage.dae"
    #marker.mesh_resource = "file:///home/kwan/sub_catkin/visualization_tutorials/rviz_plugin_tutorials/media/fired.dae"
    return marker
    
def pos_callback(msg):
    global global_x, global_y, global_z, pos_cnt
    #global_x = msg.pose.position.y * -1
    #global_y = msg.pose.position.x + 2
    global_x = msg.pose.position.x
    global_y = msg.pose.position.y
    global_z = msg.pose.position.z
    pos_cnt = pos_cnt + 1
    if pos_cnt % 1 == 0:
        marker = Marker()
        ma = MarkerArray()
        map_array = MarkerArray()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "my_namespace"
        #marker.type = marker.MESH_RESOURCE
        marker.type = marker.SPHERE
        marker.id = int(global_x*global_y*10000)
        marker.action = marker.ADD
        marker.pose.position.x = global_x
        marker.pose.position.y = global_y
        marker.pose.position.z = global_z+5
        marker.pose.orientation.x = 1.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.mesh_use_embedded_materials = True
        ma.markers.append(marker)
        drone_pub.publish(ma)
        all.publish(ma)
        map_array.markers.append(create_depot(rospy.Time.now()))
        map_array.markers.append(create_cage(rospy.Time.now()))
        map_pub.publish(map_array)

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
        self.nose_id = 0
        self.neck_id = 1
        self.leftH_id = 7
        self.rightH_id = 4

        # define a publisher to publish the 3D skeleton of multiple people
        self.skeleton_pub = rospy.Publisher(self.ns, MarkerArray, queue_size=1)
        
        self.position_pub = rospy.Publisher("/position",PositionMeasurementArray, queue_size=10)
        #self.circle_pub = rospy.Publisher("/circle",PositionMeasurementArray, queue_size=10)
        self.human_pub = rospy.Publisher("/human_localization", Marker, queue_size=10)
        self.group_pub = rospy.Publisher("/group_localization", MarkerArray, queue_size=10)
        #self.fire_pub = rospy.Publisher("/fire", MarkerArray, queue_size=10)
        self.path_pub = rospy.Publisher("/path", MarkerArray, queue_size=10)
        self.all_pub = rospy.Publisher("/all", MarkerArray, queue_size=100)
        # define a subscriber to retrive tracked bodies
        rospy.Subscriber(frame_topic, Frame, self.frame_callback)
        rospy.Subscriber('mavros/local_position/pose', PoseStamped, pos_callback)

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

    def create_human(self, danger, x, y, z, time, counter):
        marker = Marker()
        marker.header.frame_id = self.skeleton_frame
        marker.header.stamp = time
        marker.ns = "my_namespace"
        marker.type = marker.MESH_RESOURCE
        #marker.type = marker.SPHERE
        marker.id = counter
        marker.action = marker.ADD
        marker.pose.position.x = global_x + z - 0.9
        marker.pose.position.y = global_y - x
        marker.pose.position.z = 20
        marker.scale.x = 0.75
        marker.scale.y = 0.75
        marker.scale.z = 0.75
        marker.mesh_use_embedded_materials = True
        marker.mesh_resource = "file:///home/kwan/Downloads/manicon.dae"
        #marker.mesh_resource = "file:///home/kwan/catkin_ws/src/visualization_tutorials/rviz_plugin_tutorials/media/fired.dae"
        mark = deepcopy(marker)
        mark.color.a = 1
        if danger == 1 :
            mark.color.r = 1.0
            mark.color.g = 1.0
            mark.color.b = 0.0
            #mark.text = "FALL"
            mark.pose.position.x = global_x + z + 0.1 - 0.9
            mark.pose.position.y = global_y - x - 0.45
            mark.pose.orientation.x = -0.53
            mark.pose.orientation.y = 0.46
            mark.pose.orientation.z = 0.46
            mark.pose.orientation.w = 0.53
        else :
            mark.color.r = 0.0
            mark.color.g = 1.0
            mark.color.b = 0.0
            #mark.text = "STAND"
            mark.pose.orientation.x = -0.53
            mark.pose.orientation.y = 0.46
            mark.pose.orientation.z = -0.46
            mark.pose.orientation.w = 0.53
        #mark.color.b = random.random()
        return mark
    


    def create_point(self, x, y, time):
        marker = Marker()
        marker.header.frame_id = self.skeleton_frame
        marker.header.stamp = time
        marker.ns = "my_namespace"
        #marker.type = marker.MESH_RESOURCE
        marker.type = marker.SPHERE
        marker.id = int(x* y)
        marker.action = marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 1
        marker.pose.orientation.x = 1.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.mesh_use_embedded_materials = True
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1
        return marker

    def create_path(self, sx, sy, fx, fy, time):
        marker = Marker()
        marker.header.frame_id = self.skeleton_frame
        marker.header.stamp = time
        marker.ns = "my_namespace"
        marker.type = marker.LINE_STRIP
        marker.id = int(fy * fx) + 25
        marker.action = marker.ADD
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 1
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = 0.1
        marker.scale.z = 0.05

        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = 0
        marker.color.a = 1
        marker.points.append(Point(sx,sy,1))
        marker.points.append(Point(fx,fy,1))
        return marker

    def isValid(self, bodyPart):
        return bodyPart.score > 0 and not math.isnan(bodyPart.point.x) and not math.isnan(bodyPart.point.y) and not math.isnan(bodyPart.point.z) and bodyPart.point.z > 0

    def fire_callback(self, data):
        FF = self.create_fire(data.px, data.py, data.pz,rospy.Time.now())
        self.fire_array.markers.append(FF)
        self.all_array.markers.append(FF)

    def frame_callback(self, data):
        '''
        This function will be called everytime whenever a message is received by the subscriber
        '''
        marker_counter = 0
        person_counter = 0
        marker_array = MarkerArray()
        Group_Array = MarkerArray()
        pos_array = PositionMeasurementArray()
        pos_array.header.frame_id = self.skeleton_frame
        fire_array = MarkerArray()
        path_array = MarkerArray()
        all_array = MarkerArray()

        for person in data.persons:
            #print(person)
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

            person_id = self.create_marker(marker_counter+12, ColorRGBA(0, 0, 0, 1.00), Marker.TEXT_VIEW_FACING, 0.35, now)
            marker_counter += 1
            # assign person id and 3D position
            person_id.text = str(person_counter)
            #nose = person.bodyParts[self.nose_id]
            neck = person.bodyParts[self.neck_id]
            leftHand = person.bodyParts[self.leftH_id]
            rightHand = person.bodyParts[self.rightH_id]
            pixellist = []
            xmin = 10000
            xmax = 0
            ymin = 10000
            ymax = 0
            for i in range(0,25):
                if person.bodyParts[i].pixel.x != 0 :
                    if xmin > person.bodyParts[i].pixel.x :
                        xmin = person.bodyParts[i].pixel.x
                    if xmax <= person.bodyParts[i].pixel.x :
                        xmax = person.bodyParts[i].pixel.x
                    if ymin > person.bodyParts[i].pixel.y :
                        ymin = person.bodyParts[i].pixel.y
                    if ymax <= person.bodyParts[i].pixel.y :
                        ymax = person.bodyParts[i].pixel.y
            xmin = xmin - 1
            xmax = xmax + 1
            ymin = ymin - 1
            ymax = ymax + 1

            xlen = xmax - xmin
            ylen = ymax - ymin

            for i in range(0,25):
                if person.bodyParts[i].pixel.x == 0 :
                    pixellist.append(0)
                else :
                    pixellist.append(((person.bodyParts[i].pixel.x)-xmin)/xlen)
                if person.bodyParts[i].pixel.y == 0 :
                    pixellist.append(0)
                else :
                    pixellist.append(((person.bodyParts[i].pixel.y)-ymin)/ylen)

            
            numpy_points = numpy.asarray(pixellist)
            loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            test_X = numpy_points[0:50]
            if (test_X.ndim == 1):
                test_X = numpy.array([test_X])
            pred = loaded_model.predict(test_X)
            
            if pred*100 > 90:
                #predict = "FALL  " + str(int(pred*100)) + "%"
                predict = "FALL"
                danger = 1
            else :
                #predict = "STAND  " + str(100-int(pred*100)) + "%"
                predict = "STAND"
                danger = 0
            person_id.text = str(predict)

            if self.isValid(neck):
                #person_id.pose.position = Point(nose.point.x, nose.point.y + self.id_text_offset, nose.point.z)
                #print(neck.score)
                #print(type(neck.score))
                person_id.pose.position = Point(neck.point.z+global_x-0.1 - 0.9, global_y-neck.point.x, 2)
                PP = self.create_position(neck.point.x, neck.point.y, neck.point.z, now)
                print(neck.point.z)
                if neck.score > 0.6 and neck.point.z < 2:
                    HH = self.create_human(danger, neck.point.x, neck.point.y, neck.point.z, now, person_counter)
                    Group_Array.markers.append(HH)
                    all_array.markers.append(HH)
                    print("x : "+ str(neck.point.x))
                    print("y : "+ str(neck.point.y))
                    print("z : "+ str(neck.point.z))
                    #DD = self.create_drone(now)
                    Group_Array.markers.append(person_id)
                    #Drone_Array.markers.append(DD)
                    all_array.markers.append(person_id)

                pos_array.people.append(PP)

            new_line = str("'"+'pose: {position: {x:'+ str(neck.point.x) + ' y:'+ str(neck.point.y) +' z: '+ str(0.9) + '}} ''name: "new_name" ''allow_renaming: true'+"'")
            global pastx
            global pasty
            global pastz
            global spawn_cnt
            global fire_cnt
            global fire_sw
            global fire_scale
            pastx = neck.point.x
            pasty = neck.point.y
            pastz = neck.point.z
            #print(new_line)
            person_counter += 1
        '''
        for i in range(1,4) : 
            FF = self.create_fire(1,2,i,rospy.Time.now())
            fire_id = self.create_marker(i, ColorRGBA(0, 0, 0, 1.00), Marker.TEXT_VIEW_FACING, 0.8, now)
            fire_id.pose.position = Point(-2, i, 3)
            fire_id.text = str("fire")
            fire_array.markers.append(fire_id)
            fire_array.markers.append(FF)
            all_array.markers.append(fire_id)
            all_array.markers.append(FF)
        # publish the markers
        '''
        
        

        self.skeleton_pub.publish(marker_array)
        self.position_pub.publish(pos_array)
        self.group_pub.publish(Group_Array)
        #self.fire_pub.publish(fire_array)
        self.path_pub.publish(path_array)
        self.all_pub.publish(all_array)
        #self.drone_pub.publish(Drone_Array)

        Group_Array.markers = []
        fire_array.markers = []
        path_array.markers = []
        all_array.markers = []
        #Drone_Array.markers = []

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
