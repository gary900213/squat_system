from PyQt5.QtGui import QImage, QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt
from numpy import loadtxt
import time
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from random import randint
import math 
import os
import openpyxl
from datetime import date,datetime
import re
import csv

# sys.path.insert(1, '../')
import pykinect_azure as pykinect
from pykinect_azure.k4a import Device
from pykinect_azure.k4abt import body2d
from pykinect_azure.k4abt._k4abtTypes import K4ABT_SEGMENT_PAIRS_Right,K4ABT_SEGMENT_PAIRS_Left

from scipy.interpolate import interp1d 
from pykalman import KalmanFilter
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

import warnings
# from sklearn import preprocessing

from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtCore import QTimer 


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray,np.ndarray)
    # change_pixmap_signal = pyqtSignal(np.ndarray)
    showfps = False
    showbar = False
    showskeleton = True
    showforce = True
    body_leg_angle = 0
    dorsi_angle = 0
    neck_anlge = 0
    angle_Ratio = 0
    real_time = 0

    
    
    center_x = 0
    # finish = False
    # start_squat = False
    # aof_hip = 0
    # aof_knee = 0
    bar_shift = 0
    hip = [0,0]
    knee = [0,0]
    jointspoints = []
    # h = 136
    # s = 87
    # v = 111

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def draw_bar_point (self,bar_x_points,bar_y_points,color_skeleton,center_x,bar_height,recent_bar_x_points,result):
    
        hsvFrame = cv2.cvtColor(color_skeleton, cv2.COLOR_BGR2HSV)
        # hsvFrame = color_skeleton
        #### Set range for yellow color and define mask
        # yellow_lower = np.array([22, 93, 0], np.uint8)
        # yellow_upper = np.array([45, 255, 255], np.uint8)
        # yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
        
        #### Set range for red color and define mask
        # red_lower = np.array([255, 0, 0], np.uint8) 
        # red_upper = np.array([255, 125, 125], np.uint8)
        red_lower = np.array([150, 120, 150], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        # red_lower = np.array([56,42,140], np.uint8) 
        # red_upper = np.array([55,39,146], np.uint8)
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

        # lower_red = np.array([0,50,50])
        # upper_red = np.array([10,255,255])
        # mask0 = cv2.inRange(hsvFrame, lower_red, upper_red)
        # lower_red = np.array([170,50,50])
        # upper_red = np.array([180,255,255])
        # mask1 = cv2.inRange(hsvFrame, lower_red, upper_red)

        # red_mask = mask0 + mask1
        
        #### Morphological Transform, Dilation for each color and bitwise_and operator
        #### between imageFrame and mask determines to detect only that particular color
        kernal = np.ones((5, 5), "uint8")
        
        #### For yellow color
        # yellow_mask = cv2.dilate(yellow_mask, kernal)
        # res_yellow = cv2.bitwise_and(color_skeleton, color_skeleton,
        #                         mask = yellow_mask)
        #### For red color
        red_mask = cv2.dilate(red_mask, kernal)
        res_red = cv2.bitwise_and(color_skeleton, color_skeleton, 
                                mask = red_mask)

        #### Creating contour to track yellow color
        # contours, hierarchy = cv2.findContours(yellow_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #### Creating contour to track red color ####
        contours, hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        neck_point_y = 0
        try:
            if self.jointspoints != []:
                neck_point = self.jointspoints[3].get_coordinates()
                neck_point_y = neck_point[1]
        except:
            pass

        for pic, contour in enumerate(contours):
            #### get red spot contour ####
            area = cv2.contourArea(contour)
            # if(area > 500):
            if(area > 195 and area < 800):
                x, y, w, h = cv2.boundingRect(contour)
                color_skeleton = cv2.rectangle(color_skeleton, (x, y),(x + w, y + h),(0, 0, 255), 2)
                # if abs(int(x+(w/2)) - center_x) < 150 and abs(int(y+(h/2)) - neck_point_y) < 150:                
                bar_x_points.append(int(x+(w/2)))
                bar_y_points.append(int(y+(h/2)))
                bar_height = int(y+(h/2))
                if len(recent_bar_x_points) <60:
                    recent_bar_x_points.append(int(x + (w / 2)))
                    self.recent_bar_x_points = recent_bar_x_points  
                flag_processed = False
                if result is None and len(recent_bar_x_points) == 60:
                    flag_processed = True  
                    iqr1 = 0         
                    iqr1 = iqr(recent_bar_x_points)
                    outliers_index = np.where((recent_bar_x_points < np.percentile(recent_bar_x_points, 25) - 1.5 * iqr1) | (recent_bar_x_points > np.percentile(recent_bar_x_points, 75) + 1.5 * iqr1))
                    matrix_cleaned = np.delete(recent_bar_x_points, outliers_index)
                    result = np.mean(matrix_cleaned)
                    self.result = result
                if len(bar_x_points)>20:
                    bar_x_points.pop(0)
                    bar_y_points.pop(0)
                ##### 用圓圈(紅色)畫槓的位置 #####
                for i in range(len(bar_x_points)):
                    color_skeleton = cv2.circle(color_skeleton,(bar_x_points[i],bar_y_points[i]),5,(0,0,255),2)    
        return color_skeleton,bar_x_points,bar_height

    def get_angle(self,point1,point2,point3,point4): 
        vector1 = [point1[0]-point2[0],point1[1]-point2[1]]
        vector2 = [point3[0]-point4[0],point3[1]-point4[1]]

        distance1 = math.hypot(point1[0] - point2[0], point1[1] - point2[1])
        distance2 = math.hypot(point3[0] - point4[0], point3[1] - point4[1])
        product = vector1[0]*vector2[0] + vector1[1]*vector2[1]
        angleInDegree=0
        if distance1*distance2 != 0:
            angle = product/(distance1*distance2)
            if angle >=-1 and angle <=1:
                angleInDegree = 180-math.degrees(math.acos(angle))
        return angleInDegree

    #### get center line of people ####
    def get_center_line(self,image,ankle,foot):
        # distance = math.sqrt((ankle[0]-foot[0])**2+(ankle[1]-foot[1])**2)
        center_x = ankle[0]*0.5 + foot[0]*0.5
        center_y = ankle[1]*0.5 + foot[1]*0.5
        # image = cv2.circle(image, (int(center_x),int(center_y)), 3, (0,0,0), 3)
        # image = cv2.line(image, (int(center_x),100), (int(center_x),int(center_y)+30),(0, 0, 0), 2)
        # image = cv2.line()
        return image

    def judge_side(self,image):
        r_side = False
        center_x = 0
        if len(self.jointspoints)>0:
            point1 = self.jointspoints[1].get_coordinates() # SPINE_NAVAL
            r_ankle = self.jointspoints[24].get_coordinates()# right angle
            r_foot = self.jointspoints[25].get_coordinates()# right foot
            r_heel = (r_ankle[0],r_foot[1])
            l_ankle = self.jointspoints[20].get_coordinates()# left angle
            l_foot = self.jointspoints[21].get_coordinates()# left foot
            l_heel = (l_ankle[0],l_foot[1])
            if r_heel[1] > l_heel[1]:
                r_side = True
                hip = self.jointspoints[22].get_coordinates()
                knee = self.jointspoints[23].get_coordinates()
                ankle = self.jointspoints[24].get_coordinates()
                foot = self.jointspoints[25].get_coordinates()                
            else:
                hip = self.jointspoints[18].get_coordinates()
                knee = self.jointspoints[19].get_coordinates()
                ankle = self.jointspoints[20].get_coordinates()
                foot = self.jointspoints[21].get_coordinates()
            center_x = int(ankle[0]*0.5 + foot[0]*0.5)
            center_y = int(ankle[1]*0.5 + foot[1]*0.5)
            center_point = (center_x,center_y)
            center_point_new = (center_x,100)
            self.center_x = center_x
            self.dorsi_angle = 180 - self.get_angle(knee,ankle,center_point_new,center_point)
            self.neck_anlge = 180 - self.get_angle(point1,hip,center_point_new,center_point)
            if self.neck_anlge != 0:
                self.angle_Ratio = round(self.dorsi_angle/self.neck_anlge,2)
            # angle_leg = self.get_angle(hip,knee,knee,ankle)
            # image = cv2.putText(image, str(angle_leg), (knee[0]+50,knee[1]), cv2.FONT_HERSHEY_COMPLEX,1.0, (0, 0, 255))
            #
            # image = cv2.putText(image, str(self.dorsi_angle), (ankle[0]+50,ankle[1]), cv2.FONT_HERSHEY_COMPLEX,1.0, (0,255,0))
            #
            image = self.get_center_line(image, ankle, foot)
            # self.get_arm_of_force(hip,knee,center_x)
            # cv2.putText(image, str(ankle), ankle,cv2.FONT_HERSHEY_COMPLEX,1.0, (0, 255, 255))
            # cv2.putText(image, str(foot), foot,cv2.FONT_HERSHEY_COMPLEX,1.0, (0, 255, 255))		
        return image,r_side,center_x

    def get_body_leg_angle(self,image,r_side):
        if len(self.jointspoints)>0:
            point1 = self.jointspoints[1].get_coordinates()
            if r_side:
                hip = self.jointspoints[22].get_coordinates()
                knee = self.jointspoints[23].get_coordinates()
            else:
                hip = self.jointspoints[18].get_coordinates()
                knee = self.jointspoints[19].get_coordinates()
            self.body_leg_angle = self.get_angle(point1,hip,hip,knee)
            #
            # image = cv2.putText(image, str(self.body_leg_angle), self.jointspoints[22].get_coordinates(),cv2.FONT_HERSHEY_COMPLEX,1.0, (0,255,0))
            self.real_time = datetime.today()
            # image = cv2.putText(image, str(self.real_time), (600,30),cv2.FONT_HERSHEY_COMPLEX,1.0, (0,0,255))
            #

            self.hip = hip
            self.knee = knee

            # """Judge Squat Start and End"""
            # start_threshold = 140
            # if self.body_leg_angle < start_threshold:
            #     self.start_squat = True

            # if hip[1] + 40 > knee[1] and self.start_squat == True and self.finish == False:
            #     self.squrt_finish_count += 1
            #     self.finish = True
            
            # if self.body_leg_angle > start_threshold and self.start_squat == True:
            #     self.start_squat = False
            #     self.squrt_count += 1
            #     self.aof_hip = 0
            #     self.aof_knee = 0
            #     self.finish = False

        return image


    def draw_skeleton(self,image,r_side):
        points=[]	
        if len(self.jointspoints)>0:
            if r_side:
                for segmentId in range(len(K4ABT_SEGMENT_PAIRS_Right)): # draw line
                    segment_pair = K4ABT_SEGMENT_PAIRS_Right[segmentId]
                    points,image = self.draw_point_line(image, segment_pair, points, (202, 183, 42))
            else :
                for segmentId in range(len(K4ABT_SEGMENT_PAIRS_Left)): # draw line
                    segment_pair = K4ABT_SEGMENT_PAIRS_Left[segmentId]
                    points,image = self.draw_point_line(image, segment_pair, points, (202, 183, 42))
        return image

    def draw_point_line(self, image, segment_pair, points, color):
        point1 = self.jointspoints[segment_pair[0]].get_coordinates()
        point2 = self.jointspoints[segment_pair[1]].get_coordinates()
        image = cv2.line(image, point1, point2,color, 2)
        if segment_pair[0] not in points:
            points.append(segment_pair[0])
            image = cv2.circle(image, point1, 3, color, 3)
        if segment_pair[1] not in points:
            points.append(segment_pair[1])
            image = cv2.circle(image, point2, 3, color, 3)
        return points,image

    def run(self):
        ########### 啟動相機時呼叫函數 ###########
        pykinect.initialize_libraries(track_body=True)
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
        device = pykinect.start_device(config=device_config)
        bodyTracker = pykinect.start_body_tracker()
        ############################################

        bar_x_points=[]
        bar_y_points=[]
        recent_bar_x_points = []
        recent_bar_x_points1 = []
        result = None
        bar_height = 0
        fps_start_time=0
        fps=0
        max_fps=0
        min_fps=0
        count=0
        r_side=False

        while self._run_flag:
                        
            fps_end_time=time.time()
            time_diff=fps_end_time-fps_start_time
            fps=1/(time_diff)
            ######### 剛啟動時fps較低，跑一下讓fps提高再開始記錄 #########
            if count==10:
                min_fps = fps
            if fps>max_fps:
                max_fps = fps
            if fps<min_fps:
                min_fps = fps
            fps_start_time=fps_end_time
            count=count+1
            fps_text = "FPS:{:.2f}".format(fps)
            max_fps_text = "FPS:{:.2f}".format(max_fps)
            min_fps_text = "FPS:{:.2f}".format(min_fps)
            #############################################

            capture = device.update()
            body_frame = bodyTracker.update()
            ret, color_image = capture.get_color_image()
            # color_image = color_image[:,425:775]
            color_skeleton, left_joints = body_frame.draw_bodies(color_image, pykinect.K4A_CALIBRATION_TYPE_COLOR) #%#%
            self.jointspoints = left_joints
            self.testimage = color_skeleton
            color_skeleton,r_side,center_x = self.judge_side(color_skeleton)
            color_skeleton = self.get_body_leg_angle(color_skeleton,r_side)            
            if self.showbar:
                color_skeleton , bar_x_points , bar_height = self.draw_bar_point(bar_x_points , bar_y_points , color_skeleton , center_x , bar_height,recent_bar_x_points,result)
                if len(bar_x_points) > 0:
                    index = len(bar_x_points) -1
                    self.bar_shift = bar_x_points[index] - center_x
                    self.bar_height = bar_height
                    self.bar_x_points = bar_x_points
                # self.bar_shift = bar_x_points[0] - center_x
            if self.showforce and self.knee != 0 and self.hip != 0:
                cv2.line(color_skeleton,(self.knee[0],self.knee[1]),(self.center_x,self.knee[1]),(0, 0, 255), 5)
                cv2.line(color_skeleton,(self.hip[0],self.hip[1]),(self.center_x,self.hip[1]),(0, 0, 255), 5)
                # cv2.line(color_skeleton,(self.knee[0],self.knee[1]),(0,0),(0, 0, 255), 5)

            if self.showskeleton:
                color_skeleton = self.draw_skeleton(color_skeleton,r_side)
            # color_skeleton = color_skeleton[:,200:1080]  #880
            # color_skeleton = cv2.resize(color_skeleton, (1188, 972), interpolation=cv2.INTER_CUBIC)
            color_skeleton = color_skeleton[:,425:775]  #350
            color_skeleton = cv2.resize(color_skeleton, (472, 972), interpolation=cv2.INTER_CUBIC)
            self.frame = color_skeleton
            color_skeleton = cv2.flip(color_skeleton, 1)

            self.new_frame = color_skeleton
            
            if self.showfps:
                cv2.putText(self.new_frame,fps_text,(5,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
                cv2.putText(self.new_frame,max_fps_text,(5,60),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
                cv2.putText(self.new_frame,min_fps_text,(5,90),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0))           
            
            if ret:
                self.change_pixmap_signal.emit(self.new_frame,self.frame)            

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class Ui_MainWindow(QtWidgets.QWidget):
    x_line_bar_shift = []
    y_line_bar_shift = []
    x_line_body_leg = []
    y_line_body_leg = []
    x_line_dorsi = []
    y_line_dorsi = []
    neck_anlge_digram = [0 for _ in range(10)]
    def __init__(self):
        super().__init__()        
        self.showpoint=False
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()
        
        self.datapath = ""
        self.video_pre_store = []
        self.video_post_store = []
        self.add_pre_video = False
        self.add_post_video = True
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # self.testout = cv2.VideoWriter('test_video.mp4', fourcc, 13.0, (880, 720))
        # self.testout = cv2.VideoWriter('test_video.mp4', fourcc, 13.0, (350, 720))
        self.testout = cv2.VideoWriter('test_video.mp4', fourcc, 28.0, (472, 972))
        self.start_squat = False
        self.finsih = False
        self.squrt_count = 0
        self.aof_hip = 0
        self.aof_knee = 0
        self.ratio = 0 
        self.squrt_finish_count = 0
        self.finish = False
        self.times = 1
        self.Saving = False
        self.row = 0
        self.each_count = False # judge each squat count(each squat just predict once)
        
        #
        self.hip_lowest = 0
        self.dorsi_post = []
        self.dorsi_pre = []
        self.body_pre = []
        self.body_post = []
        self.ratio_pre = []
        self.ratio_post = []
        self.angle_ratio_pre = []
        self.angle_ratio_post = []
        self.bar_pre = []
        self.bar_post = []
        self.bar_height_pre = []
        self.bar_height_post = []
        self.clock_pre = []
        self.acc = []
        #
        #### kwei's model ####
        # self.model = joblib.load("Model\\Random_Forest_ratio&z_score.h5")
        # self.model_2_10 = joblib.load("Model\\RF_2_10.h5")
        # self.model_4 = joblib.load("Model\\RF_4.h5")
        # self.model_5 = joblib.load("Model\\RF_5.h5")
        # self.model_6 = joblib.load("Model\\RF_6.h5")
        # self.model_3_7 = joblib.load("Model\\RF_3_7.h5")
        # self.model_11 = joblib.load("Model\\RF_11.h5")
        #### sexy Miles's model ####
        self.model_2 = load_model('C:/Users/N2610/squat_classify/model/stupid_miles_model/depth/LSTM_after_SHAP.h5')
        self.model_3_4 = load_model('C:/Users/N2610/squat_classify/model/stupid_miles_model/lean_forward_backeard/1D-CNN_after_SHAP.h5')
        self.model_6_7 = load_model('C:/Users/N2610/squat_classify/model/stupid_miles_model/descending_not_synchronize/1D-CNN_after_SHAP.h5')
        self.model_5 = load_model('C:/Users/N2610/squat_classify/model/stupid_miles_model/rise_not_synchronize/LSTM_filter_features.h5')

        ########## tab3(video controller) member ##########
        self.video_path = ""
        self.qpixmap_fix_width = 1120 # 16x9 = 1920x1080 = 1280x720 = 1120x630 = 800x450
        self.qpixmap_fix_height = 630
        self.current_frame_no = 0
        self.videoplayer_state = "stop"
        self.speed = 1000

    def on_tab_changed(self, index):
        if index == 1:
            self.error_button()

    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1708, 972)
        self.tabs = QtWidgets.QTabWidget(MainWindow)
        self.tab1 = QtWidgets.QWidget()
        self.tab3 = QtWidgets.QWidget()
        self.tabs.addTab(self.tab1,"pro")
        self.tabs.addTab(self.tab3,"playback")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.tabs.resize(1708, 972)

        #放攝影畫面的那個長條
        # self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.tab1) #指定在第幾頁tab
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 500, 972)) #設定物件位置以及大小  x y 寬度  高度
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget") #設定物件名稱
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget) #把該物件設定成layout
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.image = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.image.setText("")
        self.image.setObjectName("image")
        self.verticalLayout_4.addWidget(self.image)

        #放波形圖的那個長條
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.tab1)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(370, 0, 580, 972)) #移動這個可以改tab1波型圖位置
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_5.setContentsMargins(110, 0, 0, 0) # 改這個又會移到波形圖的位置
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setContentsMargins(10, -1, -1, -1) #改這個可以改start按鈕那邊

        #最偏右邊那個長條，有放一張ex照片
        self.verticalLayoutWidget_6 = QtWidgets.QWidget(self.tab1)
        self.verticalLayoutWidget_6.setGeometry(QtCore.QRect(1200, 190, 650, 822))
        self.verticalLayoutWidget_6.setObjectName("verticalLayoutWidget_6")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_6)
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setSpacing(6)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout_3.setContentsMargins(10, -1, -1, -1)
        label15 = QtWidgets.QLabel(self)
        self.img_example = QPixmap('schematic_EN_resize.png')
        label15.setPixmap(self.img_example)
        self.verticalLayout_9.addWidget(label15)

        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)

        word = QtGui.QFont()
        word.setFamily("標楷體")
        word.setPointSize(25)


        # new layout for buttons at coordinate 1030
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.tab1)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(980, 10, 600, 300))  # Set the position and size
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")

        self.gridLayout_1 = QtWidgets.QGridLayout(self.verticalLayoutWidget_5)
        self.gridLayout_1.setContentsMargins(10, 10, 10, 10)
        self.gridLayout_1.setObjectName("gridLayout_1")
        self.gridLayout_1.setSpacing(3)  # 設置行距為3像素
        # Set column minimum width for the first column in gridLayout_1
        # self.gridLayout_1.setColumnMinimumWidth(0, 50)  # Set the first column's minimum width to 200 pixels
        self.gridLayout_1.setColumnMinimumWidth(1, 400)  # Set the second column's minimum width to 200 pixels
        self.gridLayout_1.setColumnStretch(0, 1)  # First column's stretch factor
        self.gridLayout_1.setColumnStretch(1, 1)  # Second column's stretch factor




        # Bar checkbox
        self.Bar_checkbox = QtWidgets.QCheckBox(self.verticalLayoutWidget_5)
        self.Bar_checkbox.setObjectName("Bar_checkbox")
        self.Bar_checkbox.setFont(font)
        self.Bar_checkbox.setText("Bar Checkbox")
        self.Bar_checkbox.clicked.connect(self.Bar_checkBoxClicked)
        self.gridLayout_1.addWidget(self.Bar_checkbox, 0, 0, 1, 1)

        # Squat Count text
        self.label1 = QtWidgets.QLabel(self.verticalLayoutWidget_5)
        self.label1.setObjectName("label1")
        self.label1.setFont(font)
        self.label1.setText("Squat Count")
        self.gridLayout_1.addWidget(self.label1, 1, 0, 1, 1)

        # Saving text
        self.label7 = QtWidgets.QLabel(self.verticalLayoutWidget_5)
        self.label7.setObjectName("label7")
        self.label7.setFont(font)
        self.label7.setText("Saving")
        self.gridLayout_1.addWidget(self.label7, 2, 0, 1, 1)

        # Start button
        self.bt_start = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        self.bt_start.setObjectName("bt_start")
        self.bt_start.setFont(font)
        self.bt_start.setText("Start")
        self.bt_start.clicked.connect(self.Create_data_folder)
        self.gridLayout_1.addWidget(self.bt_start, 3, 0, 1, 1)

        # Model predict text
        self.label8 = QtWidgets.QLabel(self.verticalLayoutWidget_5)
        self.label8.setObjectName("label8")
        self.label8.setFont(font)
        self.label8.setText("Model Predict")
        self.gridLayout_1.addWidget(self.label8, 0, 1, 1, 1)

        # Predict result text
        self.predict_result = QtWidgets.QLabel(self.verticalLayoutWidget_5)
        self.predict_result.setObjectName("predict_result")
        self.predict_result.setFont(font)
        self.predict_result.setText("Predict Result")
        self.gridLayout_1.addWidget(self.predict_result, 1, 1, 1, 1)

        # Predict result score text
        self.predict_result_score = QtWidgets.QLabel(self.verticalLayoutWidget_5)
        self.predict_result_score.setObjectName("predict_result_score")
        self.predict_result_score.setFont(font)
        self.predict_result_score.setText("Predict Result Score")
        self.gridLayout_1.addWidget(self.predict_result_score, 4, 0, 1, 1)


        self.verticalLayout_5.addLayout(self.gridLayout)             

        styles = {"color": "#f00", "font-size": "15px"}
        # Bar_Shift diagram (tab1)
        self.graphWidget = pg.PlotWidget()        
        self.x = list(range(100))  # 100 time points
        self.y_bar_shift = [0 for _ in range(100)]  # 100 data points
        # self.graphWidget.setTitle("Bar_Shift", color="b", size="15pt") # 英文版
        self.graphWidget.setTitle("Bar Shift", color="b", size="15pt") # 中文版
        self.graphWidget.setLabel("left", "Shift", **styles)
        self.graphWidget.setLabel("bottom", "Time", **styles)
        self.graphWidget.setBackground('w')
        self.graphWidget.showGrid(x=True, y=True)
        self.graphWidget.setYRange(-90, 90, padding=0)
        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line = self.graphWidget.plot(self.x, self.y_bar_shift, pen=pen)

        # Body-Thigh diagram (tab1)
        self.graphWidget_1 = pg.PlotWidget()        
        self.y_body_leg = [0 for _ in range(100)]  # 100 data points
        # self.graphWidget_1.setTitle("Body-Thigh", color="b", size="15pt") # 英文版
        self.graphWidget_1.setTitle("Body-Thigh Angle", color="b", size="15pt") # 中文版
        self.graphWidget_1.setLabel("left", "Angle (°)", **styles)
        self.graphWidget_1.setLabel("bottom", "Time", **styles)
        self.graphWidget_1.setBackground('w')
        self.graphWidget_1.showGrid(x=True, y=True)
        self.graphWidget_1.setYRange(70, 180, padding=0)
        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line_1 = self.graphWidget_1.plot(self.x, self.y_body_leg, pen=pen)

        # Dorsiflexion diagram (tab1)
        self.graphWidget_2 = pg.PlotWidget()        
        self.y_dorsi = [0 for _ in range(100)] # 100 data points
        # self.graphWidget_2.setTitle("Dorsiflexion", color="b", size="15pt") # 英文版
        self.graphWidget_2.setTitle("Dorsiflexion", color="b", size="15pt") # 中文版
        self.graphWidget_2.setLabel("left", "Angle (°)", **styles)
        self.graphWidget_2.setLabel("bottom", "Time", **styles)
        self.graphWidget_2.setBackground('w')
        self.graphWidget_2.showGrid(x=True, y=True)
        self.graphWidget_2.setYRange(0, 50, padding=0)
        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line_2 = self.graphWidget_2.plot(self.x, self.y_dorsi, pen=pen)

        # angle_variation_ratio diagram (tab1)
        self.graphWidget_7 = pg.PlotWidget()   
        self.y_angle_variation_ratio = [0 for _ in range(100)] # 100 data points
        # self.graphWidget_7.setTitle("Knee-hip Ratio", color="b", size="15pt") # 英文版
        self.graphWidget_7.setTitle("Knee-Hip Ratio", color="b", size="15pt") # 中文版
        self.graphWidget_7.setLabel("left", "Angle (°)", **styles)
        self.graphWidget_7.setLabel("bottom", "Time", **styles)
        self.graphWidget_7.setBackground('w')
        self.graphWidget_7.showGrid(x=True, y=True)
        self.graphWidget_7.setYRange(-20, 20, padding=0)  #應該要改 先隨便用一個range
        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line_7 = self.graphWidget_7.plot(self.x, self.y_angle_variation_ratio, pen=pen)


        # set timer to update data and graph
        self.timer = QtCore.QTimer(self.tab1)
        self.timer.setInterval(50) # each 50ms update once
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

        # # # #在TAB1的四個波形圖的位置
        self.topbox = QtWidgets.QHBoxLayout()
        self.topbox.setObjectName("topbox")
        self.topbox.setContentsMargins(10, -1, -1, -1)  
        self.topbox.addWidget(self.graphWidget)  
        self.verticalLayout_5.addLayout(self.topbox)
        self.midbox = QtWidgets.QHBoxLayout()
        self.midbox.setObjectName("midbox")
        self.midbox.setContentsMargins(10, -1, -1, -1)
        self.midbox.addWidget(self.graphWidget_1)
        self.verticalLayout_5.addLayout(self.midbox)
        self.bottombox = QtWidgets.QHBoxLayout()
        self.bottombox.setObjectName("bottombox")
        self.bottombox.setContentsMargins(10, -1, -1, -1)
        self.bottombox.addWidget(self.graphWidget_2)
        self.verticalLayout_5.addLayout(self.bottombox)
        #在這邊增加knee hip的物件
        self.lastbox = QtWidgets.QHBoxLayout()
        self.lastbox.setObjectName("lastbox")
        self.lastbox.setContentsMargins(10, -1, -1, -1)  
        self.lastbox.addWidget(self.graphWidget_7)  
        self.verticalLayout_5.addLayout(self.lastbox)


    

        ############################ tab3 ############################
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.tab3)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(1189, 0, 519, 972))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout_2.setContentsMargins(10, -1, -1, -1)

        # self.verticalLayoutWidget_7 = QtWidgets.QWidget(self.tab3)
        # self.verticalLayoutWidget_7.setGeometry(QtCore.QRect(1200, 50, 650, 822))
        # # self.verticalLayoutWidget.setGeometry(QtCore.QRect(200, 0, 472, 972))
        # self.verticalLayoutWidget_7.setObjectName("verticalLayoutWidget_7")
        # self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_7)
        # self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
        # self.verticalLayout_10.setObjectName("verticalLayout_10")
        # self.gridLayout_4 = QtWidgets.QGridLayout()
        # self.gridLayout_4.setSpacing(6)
        # self.gridLayout_4.setObjectName("gridLayout_4")
        # self.gridLayout_4.setContentsMargins(10, -1, -1, -1)
        # label16 = QtWidgets.QLabel(self)
        # self.img_example_sol = QPixmap('solution_with_graph.png')
        # label16.setPixmap(self.img_example_sol)
        # self.verticalLayout_10.addWidget(label16)

        self.label_photo = QtWidgets.QLabel(self.tab3)
        self.label_photo.setGeometry(682, 100, 517, 711)  # 設置位置和大小
        self.label_photo.setPixmap(QPixmap("en_resize.png"))
        pixmap = QPixmap("en_resize.png")
        scaled_pixmap = pixmap.scaled(517, 711)
        self.label_photo.setPixmap(scaled_pixmap)

        # Bar_Shift diagram (tab3)
        self.graphWidget_3 = pg.PlotWidget()        
        self.x = list(range(100))  # 100 time points
        self.y_bar_shift_tab3 = [0 for _ in range(100)] # 100 data points
        self.graphWidget_3.setTitle("Bar Shift", color="b", size="15pt")
        self.graphWidget_3.setLabel("left", "Angle (°)", **styles)
        self.graphWidget_3.setLabel("bottom", "Time", **styles)
        self.graphWidget_3.setBackground('w')
        self.graphWidget_3.showGrid(x=True, y=True)
        self.graphWidget_3.setYRange(-70, 120, padding=0)
        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line_3 = self.graphWidget_3.plot(self.x, self.y_bar_shift_tab3, pen=pen)

        # Body-Thigh diagram (tab3)
        self.graphWidget_4 = pg.PlotWidget()        
        self.y_body_leg = [0 for _ in range(100)]  # 100 data points
        self.graphWidget_4.setTitle("Body-Thigh Angle", color="b", size="15pt")
        self.graphWidget_4.setLabel("left", "Angle (°)", **styles)
        self.graphWidget_4.setLabel("bottom", "Time", **styles)
        self.graphWidget_4.setBackground('w')
        self.graphWidget_4.showGrid(x=True, y=True)
        self.graphWidget_4.setYRange(70, 180, padding=0)
        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line_4 = self.graphWidget_4.plot(self.x, self.y_body_leg, pen=pen)

        # Dorsiflexion diagram (tab3)
        self.graphWidget_5 = pg.PlotWidget()        
        self.y_dorsi_tab3 = [0 for _ in range(100)]  # 100 data points
        self.graphWidget_5.setTitle("Dorsiflexion", color="b", size="15pt")
        self.graphWidget_5.setLabel("left", "Shift", **styles)
        self.graphWidget_5.setLabel("bottom", "Time", **styles)
        self.graphWidget_5.setBackground('w')
        self.graphWidget_5.showGrid(x=True, y=True)
        self.graphWidget_5.setYRange(0, 50, padding=0)
        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line_5 = self.graphWidget_5.plot(self.x, self.y_dorsi_tab3, pen=pen)

        # Angle_variation_ratio diagram (tab3)
        self.graphWidget_6 = pg.PlotWidget()      
        self.y_angle_variation_ratio_tab3 = [0 for _ in range(100)] # 100 data points
        self.graphWidget_6.setTitle("Knee-Hip Ratio", color="b", size="15pt")
        self.graphWidget_6.setLabel("left", "Angle (°)", **styles)
        self.graphWidget_6.setLabel("bottom", "Time", **styles)
        self.graphWidget_6.setBackground('w')
        self.graphWidget_6.showGrid(x=True, y=True)
        self.graphWidget_6.setYRange(-5, 5, padding=0)
        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line_6 = self.graphWidget_6.plot(self.x, self.y_angle_variation_ratio_tab3, pen=pen)

        self.topbox_1 = QtWidgets.QHBoxLayout()
        self.topbox_1.setObjectName("topbox_1")
        self.topbox_1.setContentsMargins(10, -1, -1, -1)  
        self.topbox_1.addWidget(self.graphWidget_3)  
        self.verticalLayout_8.addLayout(self.topbox_1)
        self.midbox_1 = QtWidgets.QHBoxLayout()
        self.midbox_1.setObjectName("midbox_1")
        self.midbox_1.setContentsMargins(10, -1, -1, -1)
        self.midbox_1.addWidget(self.graphWidget_4)
        self.verticalLayout_8.addLayout(self.midbox_1)
        self.bottombox_1 = QtWidgets.QHBoxLayout()
        self.bottombox_1.setObjectName("bottombox_1")
        self.bottombox_1.setContentsMargins(10, -1, -1, -1)
        self.bottombox_1.addWidget(self.graphWidget_5)
        self.verticalLayout_8.addLayout(self.bottombox_1)
        self.lastbox = QtWidgets.QHBoxLayout()
        self.lastbox.setObjectName("lastbox")
        self.lastbox.setContentsMargins(10, -1, -1, -1)  
        self.lastbox.addWidget(self.graphWidget_6)  
        self.verticalLayout_8.addLayout(self.lastbox)

        # video player
        # font format
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)

        # video player frame
        self.label_videoframe = QtWidgets.QLabel(self.tab3)
        self.label_videoframe.setGeometry(QtCore.QRect(40, 50, 306, 630))
        self.label_videoframe.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.label_videoframe.setObjectName("label_videoframe")
        self.label_videoframe.setFont(font)
        self.label_videoframe.setStyleSheet("background-color: #D3D3D3; border: none;")  # 使用較淺的灰色值 #D3D3D3
        # Open File Button
        self.button_openfile = QtWidgets.QPushButton(self.tab3)
        self.button_openfile.setGeometry(QtCore.QRect(40, 700, 113, 32))
        self.button_openfile.setObjectName("button_openfile")
        self.button_openfile.setFont(font)
        self.button_openfile.clicked.connect(self.open_file)

        # Show error message
        self.error_message_label = QtWidgets.QLabel(self.tab3)
        self.error_message_label.setGeometry(QtCore.QRect(406, 10, 650, 32))
        self.error_message_label.setObjectName("error_message_label")
        self.error_message_label.setFont(font)
        self.error_message_label.setText("Error Message (Press the button on the left to play the squat video.)")


        # Open File button 1---for play back
        self.button_openfile_1 = QtWidgets.QPushButton(self.tab3)
        self.button_openfile_1.setGeometry(QtCore.QRect(406, 50, 113, 32))
        self.button_openfile_1.setObjectName("button_openfile_playback_1")
        self.button_openfile_1.setFont(font)
        self.button_openfile_1.clicked.connect(self.open_file_button_1)

        # # Open File button 2---for play back
        self.button_openfile_2 = QtWidgets.QPushButton(self.tab3)
        self.button_openfile_2.setGeometry(QtCore.QRect(406, 50+56*1, 113, 32))
        self.button_openfile_2.setObjectName("button_openfile_playback_2")
        self.button_openfile_2.setFont(font)
        self.button_openfile_2.clicked.connect(self.open_file_button_2)

        # Open File button 3---for play back
        self.button_openfile_3 = QtWidgets.QPushButton(self.tab3)
        self.button_openfile_3.setGeometry(QtCore.QRect(406, 50+56*2, 113, 32))
        self.button_openfile_3.setObjectName("button_openfile_playback_3")
        self.button_openfile_3.setFont(font)
        self.button_openfile_3.clicked.connect(self.open_file_button_3)

        # Open File button 4---for play back
        self.button_openfile_4 = QtWidgets.QPushButton(self.tab3)
        self.button_openfile_4.setGeometry(QtCore.QRect(406, 50+56*3, 113, 32))
        self.button_openfile_4.setObjectName("button_openfile_playback_4")
        self.button_openfile_4.setFont(font)
        self.button_openfile_4.clicked.connect(self.open_file_button_4)

        # Open File button 5---for play back
        self.button_openfile_5 = QtWidgets.QPushButton(self.tab3)
        self.button_openfile_5.setGeometry(QtCore.QRect(406, 50+56*4, 113, 32))
        self.button_openfile_5.setObjectName("button_openfile_playback_5")
        self.button_openfile_5.setFont(font)
        self.button_openfile_5.clicked.connect(self.open_file_button_5)

        # Open File button 6---for play back
        self.button_openfile_6 = QtWidgets.QPushButton(self.tab3)
        self.button_openfile_6.setGeometry(QtCore.QRect(406, 50+56*5, 113, 32))
        self.button_openfile_6.setObjectName("button_openfile_playback_6")
        self.button_openfile_6.setFont(font)
        self.button_openfile_6.clicked.connect(self.open_file_button_6)

        # Open File button 7---for play back
        self.button_openfile_7 = QtWidgets.QPushButton(self.tab3)
        self.button_openfile_7.setGeometry(QtCore.QRect(406, 50+56*6, 113, 32))
        self.button_openfile_7.setObjectName("button_openfile_playback_7")
        self.button_openfile_7.setFont(font)
        self.button_openfile_7.clicked.connect(self.open_file_button_7)

        # Open File button 8---for play back
        self.button_openfile_8 = QtWidgets.QPushButton(self.tab3)
        self.button_openfile_8.setGeometry(QtCore.QRect(406, 50+56*7, 113, 32))
        self.button_openfile_8.setObjectName("button_openfile_playback_8")
        self.button_openfile_8.setFont(font)
        self.button_openfile_8.clicked.connect(self.open_file_button_8)

        # Open File button 9---for play back
        self.button_openfile_9 = QtWidgets.QPushButton(self.tab3)
        self.button_openfile_9.setGeometry(QtCore.QRect(406, 50+56*8, 113, 32))
        self.button_openfile_9.setObjectName("button_openfile_playback_9")
        self.button_openfile_9.setFont(font)
        self.button_openfile_9.clicked.connect(self.open_file_button_9)

        # Open File button 10---for play back
        self.button_openfile_10 = QtWidgets.QPushButton(self.tab3)
        self.button_openfile_10.setGeometry(QtCore.QRect(406, 50+56*9, 113, 32))
        self.button_openfile_10.setObjectName("button_openfile_playback_10")
        self.button_openfile_10.setFont(font)
        self.button_openfile_10.clicked.connect(self.open_file_button_10)

        # Open File button 11---for play back
        self.button_openfile_11 = QtWidgets.QPushButton(self.tab3)
        self.button_openfile_11.setGeometry(QtCore.QRect(406, 50+56*10, 113, 32))
        self.button_openfile_11.setObjectName("button_openfile_playback_11")
        self.button_openfile_11.setFont(font)
        self.button_openfile_11.clicked.connect(self.open_file_button_11)

        # Open File button 12---for play back
        self.button_openfile_12 = QtWidgets.QPushButton(self.tab3)
        self.button_openfile_12.setGeometry(QtCore.QRect(406, 50+56*11, 113, 32))
        self.button_openfile_12.setObjectName("button_openfile_playback_12")
        self.button_openfile_12.setFont(font)
        self.button_openfile_12.clicked.connect(self.open_file_button_12)

        # Open File button 13---for play back
        self.button_openfile_13 = QtWidgets.QPushButton(self.tab3)
        self.button_openfile_13.setGeometry(QtCore.QRect(406, 50+56*12, 113, 32))
        self.button_openfile_13.setObjectName("button_openfile_playback_13")
        self.button_openfile_13.setFont(font)
        self.button_openfile_13.clicked.connect(self.open_file_button_13)

        # Open File button 14---for play back
        self.button_openfile_14 = QtWidgets.QPushButton(self.tab3)
        self.button_openfile_14.setGeometry(QtCore.QRect(406, 50+56*13, 113, 32))
        self.button_openfile_14.setObjectName("button_openfile_playback_14")
        self.button_openfile_14.setFont(font)
        self.button_openfile_14.clicked.connect(self.open_file_button_14)

        # Open File button 15---for play back
        self.button_openfile_15 = QtWidgets.QPushButton(self.tab3)
        self.button_openfile_15.setGeometry(QtCore.QRect(406, 50+56*14, 113, 32))
        self.button_openfile_15.setObjectName("button_openfile_playback_15")
        self.button_openfile_15.setFont(font)
        self.button_openfile_15.clicked.connect(self.open_file_button_15)

        # Open File text button 1---for play back
        self.label_text_1 = QtWidgets.QLabel(self.tab3)
        self.label_text_1.setGeometry(QtCore.QRect(536, 50, 700, 32))
        self.label_text_1.setObjectName("label_text_1")
        self.label_text_1.setFont(font)

        # Open File text button 2---for play back
        self.label_text_2 = QtWidgets.QLabel(self.tab3)
        self.label_text_2.setGeometry(QtCore.QRect(536, 50+56*1, 700, 32))
        self.label_text_2.setObjectName("label_text_2")
        self.label_text_2.setFont(font)

        # Open File text button 3---for play back
        self.label_text_3 = QtWidgets.QLabel(self.tab3)
        self.label_text_3.setGeometry(QtCore.QRect(536, 50+56*2, 700, 32))
        self.label_text_3.setObjectName("label_text_3")
        self.label_text_3.setFont(font)

        # Open File text button 4---for play back
        self.label_text_4 = QtWidgets.QLabel(self.tab3)
        self.label_text_4.setGeometry(QtCore.QRect(536, 50+56*3, 700, 32))
        self.label_text_4.setObjectName("label_text_4")
        self.label_text_4.setFont(font)

        # Open File text button 5---for play back
        self.label_text_5 = QtWidgets.QLabel(self.tab3)
        self.label_text_5.setGeometry(QtCore.QRect(536, 50+56*4, 700, 32))
        self.label_text_5.setObjectName("label_text_5")
        self.label_text_5.setFont(font)

        # Open File text button 6---for play back
        self.label_text_6 = QtWidgets.QLabel(self.tab3)
        self.label_text_6.setGeometry(QtCore.QRect(536, 50+56*5, 700, 32))
        self.label_text_6.setObjectName("label_text_1")
        self.label_text_6.setFont(font)

        # Open File text button 7---for play back
        self.label_text_7 = QtWidgets.QLabel(self.tab3)
        self.label_text_7.setGeometry(QtCore.QRect(536, 50+56*6, 700, 32))
        self.label_text_7.setObjectName("label_text_7")
        self.label_text_7.setFont(font)

        # Open File text button 8---for play back
        self.label_text_8 = QtWidgets.QLabel(self.tab3)
        self.label_text_8.setGeometry(QtCore.QRect(536, 50+56*7, 700, 32))
        self.label_text_8.setObjectName("label_text_8")
        self.label_text_8.setFont(font)

        # Open File text button 9---for play back
        self.label_text_9 = QtWidgets.QLabel(self.tab3)
        self.label_text_9.setGeometry(QtCore.QRect(536, 50+56*8, 700, 32))
        self.label_text_9.setObjectName("label_text_9")
        self.label_text_9.setFont(font)

        # Open File text button 10---for play back
        self.label_text_10 = QtWidgets.QLabel(self.tab3)
        self.label_text_10.setGeometry(QtCore.QRect(536, 50+56*9, 700, 32))
        self.label_text_10.setObjectName("label_text_10")
        self.label_text_10.setFont(font)

        # Open File text button 11---for play back
        self.label_text_11 = QtWidgets.QLabel(self.tab3)
        self.label_text_11.setGeometry(QtCore.QRect(536, 50+56*10, 700, 32))
        self.label_text_11.setObjectName("label_text_11")
        self.label_text_11.setFont(font)

        # Open File text button 12---for play back
        self.label_text_12 = QtWidgets.QLabel(self.tab3)
        self.label_text_12.setGeometry(QtCore.QRect(536, 50+56*11, 700, 32))
        self.label_text_12.setObjectName("label_text_12")
        self.label_text_12.setFont(font)

        # Open File text button 13---for play back
        self.label_text_13 = QtWidgets.QLabel(self.tab3)
        self.label_text_13.setGeometry(QtCore.QRect(536, 50+56*12, 700, 32))
        self.label_text_13.setObjectName("label_text_13")
        self.label_text_13.setFont(font)

        # Open File text button 14---for play back
        self.label_text_14 = QtWidgets.QLabel(self.tab3)
        self.label_text_14.setGeometry(QtCore.QRect(536, 50+56*13, 700, 32))
        self.label_text_14.setObjectName("label_text_14")
        self.label_text_14.setFont(font)

        # Open File text button 15---for play back
        self.label_text_15 = QtWidgets.QLabel(self.tab3)
        self.label_text_15.setGeometry(QtCore.QRect(536, 50+56*14, 700, 32))
        self.label_text_15.setObjectName("label_text_15")
        self.label_text_15.setFont(font)


        # (Text) current_frame/total_frame
        self.label_framecnt = QtWidgets.QLabel(self.tab3)
        self.label_framecnt.setGeometry(QtCore.QRect(160, 705, 243, 21))
        self.label_framecnt.setObjectName("label_framecnt")
        self.label_framecnt.setFont(font)

        # file path
        self.label_filepath = QtWidgets.QLabel(self.tab3)
        self.label_filepath.setGeometry(QtCore.QRect(40, 870, 841, 41))
        self.label_filepath.setObjectName("label_filepath")
        self.label_filepath.setFont(font)
        # stop button
        self.button_stop = QtWidgets.QPushButton(self.tab3)
        self.button_stop.setGeometry(QtCore.QRect(40, 750, 90, 32))
        self.button_stop.setObjectName("button_stop")
        self.button_stop.setFont(font)
        self.button_stop.clicked.connect(self.stop)
        # pause button
        self.button_pause = QtWidgets.QPushButton(self.tab3)
        self.button_pause.setGeometry(QtCore.QRect(142, 750, 90, 32))
        self.button_pause.setObjectName("button_pause")
        self.button_pause.setFont(font)
        self.button_pause.clicked.connect(self.pause)
        # play button
        self.button_play = QtWidgets.QPushButton(self.tab3)
        self.button_play.setGeometry(QtCore.QRect(244, 750, 90, 32))
        self.button_play.setObjectName("button_play")
        self.button_play.setFont(font)
        self.button_play.clicked.connect(self.play)
        # slow down x0.5
        self.button_double = QtWidgets.QPushButton(self.tab3)
        self.button_double.setGeometry(QtCore.QRect(40, 800, 90, 32))
        self.button_double.setObjectName("button_double")
        self.button_double.setFont(font)
        self.button_double.clicked.connect(self.double)
        # slow down x0.25
        self.button_quarter = QtWidgets.QPushButton(self.tab3)
        self.button_quarter.setGeometry(QtCore.QRect(142, 800, 90, 32))
        self.button_quarter.setObjectName("button_quarter")
        self.button_quarter.setFont(font)
        self.button_quarter.clicked.connect(self.quarter)
        # slow down x1
        self.button_defult = QtWidgets.QPushButton(self.tab3)
        self.button_defult.setGeometry(QtCore.QRect(244, 800, 90, 32))
        self.button_defult.setObjectName("button_defult")
        self.button_defult.setFont(font)
        self.button_defult.clicked.connect(self.defult)
        # speed text(x0.5 , x0.25 , x1)
        self.label_speed = QtWidgets.QLabel(self.tab3)
        self.label_speed.setGeometry(QtCore.QRect(40, 835, 841, 32))
        self.label_speed.setObjectName("label_speed")
        self.label_speed.setFont(font)

        
        MainWindow.setCentralWidget(self.tabs)
        # self.menubar = QtWidgets.QMenuBar(MainWindow)
        # self.menubar.setGeometry(QtCore.QRect(0, 0, 999, 21))
        # self.menubar.setObjectName("menubar")
        # MainWindow.setMenuBar(self.menubar)
        # self.statusbar = QtWidgets.QStatusBar(MainWindow)
        # self.statusbar.setObjectName("statusbar")
        # MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        ######################## tab1 ########################
        MainWindow.setWindowTitle(_translate("MainWindow", "Body Tracker"))
        # self.FPS_checkbox.setText(_translate("MainWindow", "FPS"))
        self.Bar_checkbox.setText(_translate("MainWindow", "Barbell"))
        self.bt_start.setText(_translate("MainWindow", "Start"))
        self.label1.setText(_translate("MainWindow", "Squat Count : 0"))
        # self.label2.setText(_translate("MainWindow", "Finish : 0"))
        self.label7.setText(_translate("MainWindow", "Saving"))
        self.label8.setText(_translate("MainWindow", "Model_predict : "))

        ######################## tab3 ########################
        self.label_videoframe.setText(_translate("MainWindow", "Video Player"))
        self.button_openfile.setText(_translate("MainWindow", "Openfile"))
        self.label_framecnt.setText(_translate("MainWindow", "current frame/total frame"))
        self.button_play.setText(_translate("MainWindow", "Play"))
        self.button_stop.setText(_translate("MainWindow", "Stop"))
        self.label_filepath.setText(_translate("MainWindow", "file path:"))
        self.button_pause.setText(_translate("MainWindow", "Pause"))
        self.button_double.setText(_translate("MainWindow", "X 0.5"))
        self.button_quarter.setText(_translate("MainWindow", "X 0.25"))
        self.button_defult.setText(_translate("MainWindow", "X 1"))
        self.label_speed.setText(_translate("MainWindow", "Now Speed : X 1"))


        self.button_openfile_1.setText(_translate("MainWindow", "1st  sqaut"))
        self.button_openfile_2.setText(_translate("MainWindow", "2nd  sqaut"))
        self.button_openfile_3.setText(_translate("MainWindow", "3rd  sqaut"))
        self.button_openfile_4.setText(_translate("MainWindow", "4th  sqaut"))
        self.button_openfile_5.setText(_translate("MainWindow", "5th  sqaut"))
        self.button_openfile_6.setText(_translate("MainWindow", "6th  sqaut"))
        self.button_openfile_7.setText(_translate("MainWindow", "7th  sqaut"))
        self.button_openfile_8.setText(_translate("MainWindow", "8th  sqaut"))
        self.button_openfile_9.setText(_translate("MainWindow", "9th  sqaut"))
        self.button_openfile_10.setText(_translate("MainWindow", "10th sqaut"))
        self.button_openfile_11.setText(_translate("MainWindow", "11th sqaut"))
        self.button_openfile_12.setText(_translate("MainWindow", "12th sqaut"))
        self.button_openfile_13.setText(_translate("MainWindow", "13th sqaut"))
        self.button_openfile_14.setText(_translate("MainWindow", "14th sqaut"))
        self.button_openfile_15.setText(_translate("MainWindow", "15th sqaut"))

        self.label_text_1.setText(_translate("MainWindow", " "))
        self.label_text_2.setText(_translate("MainWindow", " "))
        self.label_text_3.setText(_translate("MainWindow", " "))
        self.label_text_4.setText(_translate("MainWindow", " "))
        self.label_text_5.setText(_translate("MainWindow", " "))
        self.label_text_6.setText(_translate("MainWindow", " "))
        self.label_text_7.setText(_translate("MainWindow", " "))
        self.label_text_8.setText(_translate("MainWindow", " "))
        self.label_text_9.setText(_translate("MainWindow", " "))
        self.label_text_10.setText(_translate("MainWindow", " "))
        self.label_text_11.setText(_translate("MainWindow", " "))
        self.label_text_12.setText(_translate("MainWindow", " "))
        self.label_text_13.setText(_translate("MainWindow", " "))
        self.label_text_14.setText(_translate("MainWindow", " "))
        self.label_text_15.setText(_translate("MainWindow", " "))


    def update_color(self):
        print(self.slider_h.value())

    def update_plot_data(self):
        
        self.x = self.x[1:]  # Remove the first y element.
        self.x.append(self.x[-1] + 1)  # Add a new value 1 higher than the last.

        # Update dorsi data
        self.y_dorsi = self.y_dorsi[1:]  # Remove the first
        self.y_dorsi.append(self.thread.dorsi_angle)
        self.data_line_2.setData(self.x, self.y_dorsi)
        # Update body_leg data
        self.y_body_leg = self.y_body_leg[1:]  # Remove the first
        self.y_body_leg.append(self.thread.body_leg_angle)      
        self.data_line_1.setData(self.x, self.y_body_leg)
        # Update bar_shift data
        if self.Bar_checkbox.isChecked():
            self.y_bar_shift = self.y_bar_shift[1:]  # Remove the first
            self.y_bar_shift.append(self.thread.bar_shift)
            self.data_line.setData(self.x, self.y_bar_shift)
        else:
            self.y_bar_shift = [0 for _ in range(100)]
            self.data_line.setData(self.x, self.y_bar_shift)

        #### self.thread.angle_Ratio
        # Update angle_Ratio data
        # print("before pop: ", self.y_angle_variation_ratio)
        self.y_angle_variation_ratio = self.y_angle_variation_ratio[1:]  # Remove the first
        # print("after pop: ", self.y_angle_variation_ratio)
        self.neck_anlge_digram.append(self.thread.neck_anlge)
        numerator = self.y_dorsi[-1] - self.y_dorsi[-2]
        denominator = self.neck_anlge_digram[-1] - self.neck_anlge_digram[-2]
        if denominator != 0:
            self.y_angle_variation_ratio.append(round(numerator / denominator,2))
            # if (numerator / denominator) > 10:
            #     self.y_angle_variation_ratio.append(10)
            # elif (numerator / denominator) < -10:
            #     self.y_angle_variation_ratio.append(-10)
            # else:
            #     self.y_angle_variation_ratio.append(round(numerator / denominator,2))
        else:
            self.y_angle_variation_ratio.append(10)
        # print("shape of y_angle_variation_ratio :", len(self.y_angle_variation_ratio))
        self.data_line_7.setData(self.x, self.y_angle_variation_ratio)
        # print("angle_variation_ratio: ", self.y_angle_variation_ratio)

        # # Add angle_variation_ratio to the plot
        # self.y_angle_variation_ratio = self.y_angle_variation_ratio[1:]  
        # self.y_angle_variation_ratio.append(angle_Variation_ratio[-1])  # Append the latest value
        # self.data_line_7.setData(self.x, self.y_angle_variation_ratio)  # Update the plot with angle_variation_ratio


    def mouseClicked(self,event,x,y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.showpoint=True
            self.x_points=x
            self.y_points=y
            print(x,y)

    def Reset_Squat_Count(self):
        self.label1.setText('Squat Count : 0')
        # self.label2.setText('Finish : 0')
        self.thread.squrt_count = 0
        self.thread.squrt_finish_count = 0
        # self.lineEdit.setText("")

    def Create_data_folder(self):
        if self.Saving == False:
            self.Saving = True
            self.flaggg = False
            self.bt_start.setText("Stop")
            path = "data/"
            now = date.today()
            current_time = now.strftime("%Y-%m-%d")
            Folderpath = path + str(current_time)        
            while True:
                if os.path.exists(Folderpath + "_" + str(self.times)):
                    self.times +=1
                    # self.testout = cv2.VideoWriter(self.datapath + 'test_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 28.0, (472, 972))
                    # print("self.datapath : ",self.datapath)
                else:
                    Folderpath = Folderpath + "_" + str(self.times)
                    os.mkdir(Folderpath)
                    self.datapath = Folderpath + "/"
                    break

        else:
            self.Saving = False
            self.bt_start.setText("Start")
            self.label1.setText('Squat Count : 0')
            # self.label2.setText('Finish : 0')
            self.squrt_count = 0
            self.squrt_finish_count = 0
            self.video_pre_store = []
            self.video_post_store = []
            # self.=[]
            self.out.release()
        self.predict_result.setText("")
        # self.predict_result_score.setText("得分:")
    
    # def FPS_checkBoxClicked(self):        
    #     if self.FPS_checkbox.isChecked():
    #         self.thread.showfps=True
    #     else:
    #         self.thread.showfps=False

    def Bar_checkBoxClicked(self):
        if self.Bar_checkbox.isChecked():
            self.thread.showbar = True
        else:
            self.thread.showbar = False

    def Skeleton_checkBoxClicked(self):
        if self.Skeleton_checkbox.isChecked():
            self.thread.showskeleton = True
        else:
            self.thread.showskeleton = False

    def Force_checkBoxClicked(self):
        if self.Force_checkbox.isChecked():
            self.thread.showforce = True
        else:
            self.thread.showforce = False

    def closeEvent(self, event):
        self.thread.stop()
        device = Device
        device.close
        event.accept()
        # path = "data/" + str(date.today().strftime("%Y-%m-%d"))
        # self.workbook.save(path + "_" + 'data.xlsx')
        # self.out.release() 

    # def load_acc(self):
    #     csvfile = open('C:/Program Files/Xsens/DOT PC SDK 2022.2/SDK Files/Examples/python/logfile_D4-22-CD-00-48-E0.csv')     # 開啟 CSV 檔案
    #     raw_data = csv.reader(csvfile)     # 讀取 CSV 檔案
    #     data = list(raw_data)
    #     time_1 = data[0]
    #     time_1 = time_1[7]
    #     substring = time_1[11:34]
    #     time_1 = datetime.strptime(substring, "%Y-%m-%d %H:%M:%S.%f")
    #     with open('0_clock.txt', 'r') as file:
    #         # 读取文件内容并将每一行存储在列表中
    #         lines = file.readlines()
    #     data_str2 = lines[len(lines)-1]
    #     data_str2 = data_str2.strip()
    #     # print(data_str2)
    #     data1 = datetime.strptime(data_str2, "%Y-%m-%d %H:%M:%S.%f")
    #     a = data1 - time_1
    #     b = a.total_seconds()
    #     c = int(b/0.0333)
    #     data_str = lines[0]
    #     data_str = data_str.strip()
    #     data2 = datetime.strptime(data_str, "%Y-%m-%d %H:%M:%S.%f")
    #     d = data2 - time_1
    #     e = d.total_seconds()
    #     f = int(e/0.0333)
    #     for i in range(c,f):
    #         p = data[i]
    #         p = p[7]
    #         self.acc.append(p)


    def Save_squrt_data(self):
        if self.datapath != "" and self.Saving == True:
            
            if self.start_squat == True and len(self.video_pre_store) != 0:
                count=0
                if self.add_pre_video == False:
                    for i in range(len(self.video_pre_store)):
                        self.out.write(self.video_pre_store[i])
                        self.f_dorsi.write(str(self.dorsi_pre[i]) + "\n")
                        self.f_body.write(str(self.body_pre[i]) + "\n")
                        self.f_ratio.write(str(self.ratio_pre[i]) + "\n")          #save pre data
                        self.f_angle_ratio.write(str(self.angle_ratio_pre[i]) + "\n")
                        self.f_clock.write(str(self.clock_pre[i]) + "\n")
                        

                    if self.thread.showbar == True:
                        for i in range(len(self.video_pre_store)):
                            self.f_bar.write(str(round(self.bar_pre[i], 2)) + "\n")
                            self.f_bar_height.write(str(round(self.bar_height_pre[i], 2)) + "\n")
                    self.add_pre_video = True

                self.f_dorsi.write(str(round(self.thread.dorsi_angle, 2)) + "\n")
                self.f_body.write(str(round(self.thread.body_leg_angle, 2)) + "\n")
                self.f_ratio.write(str(self.ratio) + "\n")
                self.f_angle_ratio.write(str(self.thread.angle_Ratio) + "\n")    #save maim data
                self.f_clock.write(str(self.thread.real_time) + "\n")
                self.out.write(self.thread.new_frame)


                # video_cap = cv2.VideoCapture(self.datapath + str(self.squrt_count-1) + '_video.mp4')
                # frame_count = 0
                # all_frames = []
                # while(True):
                #     ret, frame = video_cap.read()
                #     if ret is False:
                #         print('1111111111111111111111111111111')
                #         break
                #     all_frames.append(frame)
                #     frame_count = frame_count + 1
                #     print (frame_count)                                                               #刪影片加在這
                # if frame_count<20:   ############
                #     path = "data/"
                #     now = date.today()
                #     current_time = now.strftime("%Y-%m-%d")
                #     Folderpath = path + str(current_time) +"/"
                #     g="C:/Users/hp/squat_classify/"
                #     file = (g+self.datapath + str(self.squrt_count-1) + '_video.mp4') #str(self.squrt_count)
                #     os.remove(file)
                
                if self.thread.showbar == True:
                    self.f_bar.write(str(round(self.thread.bar_shift, 2)) + "\n")
                    self.f_bar_height.write(str(round(self.thread.bar_height, 2)) + "\n")
                
                if self.thread.hip[1] > self.hip_lowest :
                    self.hip_lowest = self.thread.hip[1]
                    cv2.imwrite(self.datapath + 'lowest' + str(self.squrt_count) + '.png',self.thread.new_frame)
                
                self.add_post_video = True
                # self.out.release()
                # self.f_dorsi.close()
                # self.f_body.close()
                # self.f_ratio.close()
                # self.f_angle_ratio.close()
                # self.f_clock.close()

            else:
                # width = 880
                # width = 350
                width = 472
                # height = 720
                height = 972
                self.add_pre_video = False
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out = cv2.VideoWriter(self.datapath + str(self.squrt_count) + '_video.mp4', fourcc, 28.0, (width, height))
                savepath_dorsi = self.datapath + str(self.squrt_count) + "_dorsi.txt"
                self.f_dorsi = open(savepath_dorsi, 'w')
                savepath_body = self.datapath + str(self.squrt_count) + "_body_leg.txt"   #存txt video
                self.f_body = open(savepath_body, 'w')
                savepath_ratio = self.datapath + str(self.squrt_count) + "_ratio.txt"
                self.f_ratio = open(savepath_ratio, 'w')
                savepath_angle_ratio = self.datapath + str(self.squrt_count) + "_angle_ratio.txt"
                self.f_angle_ratio = open(savepath_angle_ratio, 'w')
                savepath_clock = self.datapath + str(self.squrt_count) + "_clock.txt"
                self.f_clock = open(savepath_clock, 'w')
                # if  len(self.datapath + str(self.squrt_count) + '_video.mp4')<60:
                #     path = "data/"
                #     now = date.today()
                #     current_time = now.strftime("%Y-%m-%d")
                #     Folderpath = path + str(current_time) +"/"
                #     gg="C:/Users/hp/squat_classify/"
                #     file = (gg+self.datapath + str(self.squrt_count) + '_video.mp4') #str(self.squrt_count)
                #     print(gg+self.datapath + str(self.squrt_count) + '_video.mp4')
                #     os.remove(file)
                # self.out = cv2.VideoWriter(self.datapath + str(self.squrt_count) + '_video.mp4', fourcc, 13.0, (width, height))

                if self.thread.showbar == True:
                    savepath_bar = self.datapath + str(self.squrt_count) + "_bar_shift.txt"
                    self.f_bar = open(savepath_bar, 'w')
                    savepath_bar_height = self.datapath + str(self.squrt_count) + "_bar_height.txt"
                    self.f_bar_height = open(savepath_bar_height, 'w')
                if len(self.video_pre_store) == 7:
                    self.video_pre_store = self.video_pre_store[1:]
                self.video_pre_store.append(self.thread.new_frame)
                # if len(self.video_post_store) == 20:
                #     self.video_post_store = self.video_post_store[1:]
                # self.video_post_store.append(self.thread.new_frame)
                if len(self.dorsi_pre) == 7:
                    self.dorsi_pre = self.dorsi_pre[1:]
                self.dorsi_pre.append(round(self.thread.dorsi_angle, 2))
                if len(self.dorsi_post) == 7:
                    self.dorsi_post = self.dorsi_post[1:]
                self.dorsi_post.append(round(self.thread.dorsi_angle, 2))
                if len(self.body_pre) == 7:
                    self.body_pre = self.body_pre[1:]
                self.body_pre.append(round(self.thread.body_leg_angle, 2))
                if len(self.body_post) == 7:
                    self.body_post = self.body_post[1:]
                self.body_post.append(round(self.thread.body_leg_angle, 2))
                if len(self.ratio_pre) == 7:
                    self.ratio_pre = self.ratio_pre[1:]
                self.ratio_pre.append(round(self.ratio, 2))
                if len(self.ratio_post) == 7:
                    self.ratio_post = self.ratio_post[1:]
                self.ratio_post.append(round(self.ratio, 2))
                if len(self.angle_ratio_pre) == 7:
                    self.angle_ratio_pre = self.angle_ratio_pre[1:]
                self.angle_ratio_pre.append(round(self.thread.angle_Ratio, 2))
                if len(self.angle_ratio_post) == 7:
                    self.angle_ratio_post = self.angle_ratio_post[1:]
                self.angle_ratio_post.append(round(self.thread.angle_Ratio, 2))
                if len(self.clock_pre) == 7:
                    self.clock_pre = self.clock_pre[1:]
                self.clock_pre.append(self.thread.real_time)




                if self.thread.showbar == True:
                    if len(self.bar_pre) == 7:
                        self.bar_pre = self.bar_pre[1:]
                    self.bar_pre.append(round(self.thread.bar_shift, 2))
                    if len(self.bar_post) == 7:
                        self.bar_post = self.bar_post[1:]
                    self.bar_post.append(round(self.thread.bar_shift, 2))
                    if len(self.bar_height_pre) == 7:
                        self.bar_height_pre = self.bar_height_pre[1:]
                    self.bar_height_pre.append(round(self.thread.bar_height, 2))
                    if len(self.bar_height_post) == 7:
                        self.bar_height_post = self.bar_height_post[1:]
                    # if self.thread.result - 10 < self.thread.bar_x_points[-1] and self.squrt_count!= 0 :
                    #     self.Saving = False
                    #     self.bt_start.setText("Start")
                    #     self.label1.setText('Squat Count : 0')
                        # self.label2.setText('Finish : 0')
                    #     self.squrt_count = 0
                    #     self.squrt_finish_count = 0
                    #     self.video_pre_store = []
                    #     self.video_post_store = []
                    #     # self.=[]
                    #     self.out.release()
                        
                    self.bar_height_post.append(round(self.thread.bar_height, 2))
        self.testout.write(self.thread.new_frame)

    def Update_data(self):
        """Judge Squat Start and End"""
        start_threshold = 140
        if self.Saving:
            # if self.thread.result - 100 > self.thread.bar_x_points[-1]:
            if self.thread.body_leg_angle < start_threshold:
                self.start_squat = True
                self.predict_result.setText("Predicting.....")
                

            if self.thread.hip[1] + 40 > self.thread.knee[1] and self.start_squat == True and self.finish == False:
                self.squrt_finish_count += 1
                self.finish = True
            
            if self.thread.body_leg_angle > start_threshold and self.start_squat == True:
                self.start_squat = False
                self.squrt_count += 1
                self.each_count = True
                self.out.release()
                self.f_dorsi.close()
                self.f_body.close()
                self.f_ratio.close()
                self.f_angle_ratio.close()
                self.f_clock.close()
                file_path = self.datapath + str(self.squrt_count-1) + '_dorsi.txt'  # 替换为实际的文件路径
                txtfile = np.loadtxt(file_path)    
                                                                                #刪影片加在這
                if len(txtfile)<30:   ############
                    video_file = (self.datapath + str(self.squrt_count-1) + '_video.mp4') #str(self.squrt_count)'\7
                    # with open(video_file, 'r') as video:
                    #     pass
                    os.remove(video_file)
                    self.squrt_count-=1
                
                self.aof_hip = 0
                self.aof_knee = 0
                self.finish = False
                #
                self.hip_lowest = 0
                self.add_post_video = False
                self.video_post_store = []
                self.dorsi_post = []
                    # self.f_ratio.close()

                
    # def get_images_from_video(video_name, time_F):
    # video_images = []
    # vc = cv2.VideoCapture(video_name)
    # c = 1
    
    # if vc.isOpened(): #判斷是否開啟影片
    #     rval, video_frame = vc.read()
    # else:
    #     rval = False

    # while rval:   #擷取視頻至結束
    #     rval, video_frame = vc.read()
        
    #     if(c % time_F == 0): #每隔幾幀進行擷取
    #         video_images.append(video_frame)     
    #     c = c + 1
    # vc.release()
    
    # return video_images
    def get_arm_of_force(self):
        # if self.start_squat == True and self.thread.hip != []:
        if self.thread.hip != []:
            self.aof_hip = abs(self.thread.hip[0]-self.thread.center_x)
            self.aof_knee = abs(self.thread.knee[0]-self.thread.center_x)
            if self.aof_hip != 0:
                self.ratio = round(self.aof_knee/self.aof_hip,2)
    
    def model_test(self):
        self.preprocessing()

    def preprocessing(self):
        if self.each_count == True:
            if self.Saving:
                if self.thread.body_leg_angle > 140:
                    self.each_count = False
                    large = 50
                    barshift = np.loadtxt(self.datapath + str(self.squrt_count-1) + "_bar_shift.txt", dtype='i', delimiter=',')
                    body_leg = np.loadtxt(self.datapath + str(self.squrt_count-1) + "_body_leg.txt", dtype='i', delimiter=',')
                    dorsi = np.loadtxt(self.datapath + str(self.squrt_count-1) + "_dorsi.txt", dtype='i', delimiter=',')
                    angle_ratio = np.loadtxt(self.datapath + str(self.squrt_count-1) + "_angle_ratio.txt", dtype = float, delimiter=',')
                    if len(barshift) !=0:
                        size = len(body_leg)
                        barshift = barshift-barshift.min()+1
                        ##### Calculate angle-variation-ratio #####
                        # Calculate the Torso angle first
                        torso_angle = []
                        for angle_ratio_, dorsi_ in zip(angle_ratio, dorsi):
                            if angle_ratio_ != 0:
                                torso_angle.append(dorsi_ / angle_ratio_)
                            else:
                                torso_angle.append(30.0)
                    #     print('torso_angle',torso_angle)
                        # Calculate the Angle Variation Ratio
                        angle_Variation_ratio = []
                        for i in range(1, len(dorsi)):
                            numerator = dorsi[i] - dorsi[i-1]
                            denominator = torso_angle[i] - torso_angle[i-1]
                            if denominator != 0:
                                if (numerator / denominator) > 10:
                                    angle_Variation_ratio.append(10)
                                elif (numerator / denominator) < -10:
                                    angle_Variation_ratio.append(-10)
                                else:
                                    angle_Variation_ratio.append(round(numerator / denominator,2)) # 四捨五入取到小數點後兩位
                            else:
                                angle_Variation_ratio.append(10)
                    #     print('angle_Variation_ratio',angle_Variation_ratio)
                        ##### Interpolation #####
                        if size != large:
                            f_new = interp1d(np.linspace(1, 10, size), barshift)
                            barshift = f_new(np.linspace(1, 10, large))
                            f_new = interp1d(np.linspace(1, 10, size), body_leg)
                            body_leg = f_new(np.linspace(1, 10, large))
                            f_new = interp1d(np.linspace(1, 10, size), dorsi)
                            dorsi = f_new(np.linspace(1, 10, large))
                        if len(angle_Variation_ratio) != large:
                            f_new = interp1d(np.linspace(1, 10, len(angle_Variation_ratio)), angle_Variation_ratio)
                            angle_Variation_ratio = f_new(np.linspace(1, 10, large))
                        self.normalize(barshift, body_leg, dorsi, angle_Variation_ratio)
        
    # def kalman(self,data):
    #     kf = KalmanFilter(transition_matrices = [1],
    #                   observation_matrices = [1],
    #                   initial_state_mean = data[0],
    #                   initial_state_covariance = 1,
    #                   observation_covariance=1,
    #                   transition_covariance=.01)
    #     state_means,_ = kf.filter(data)
    #     state_means = np.reshape(state_means,(1,50))
    #     return state_means
    
    def normalize(self, barshift, body_leg, dorsi, angle_Variation_ratio):
        barshift_z_score = self.z_score(barshift)
        body_leg_z_score = self.z_score(body_leg)
        dorsi_z_score = self.z_score(dorsi)
        angle_variation_ratio_z_score = self.z_score(angle_Variation_ratio)

        barshift_Variation = self.Variation(barshift)
        body_leg_Variation = self.Variation(body_leg)
        dorsi_Variation = self.Variation(dorsi)
        angle_variation_ratio_Variation = self.Variation(angle_Variation_ratio)

        barshift_Variation_ratio = self.Variation_ratio(barshift)
        body_leg_Variation_ratio = self.Variation_ratio(body_leg)
        dorsi_Variation_ratio = self.Variation_ratio(dorsi)
        angle_variation_ratio_Variation_ratio = self.Variation_ratio(angle_Variation_ratio)
        
        #### kwei's model ####
        # x = []
        # aa = [barshift_Variation,body_leg_Variation,dorsi_Variation,barshift_z_score,body_leg_z_score,dorsi_z_score]
        # x.append(pd.DataFrame(list(map(list, zip(*aa)))))
        # X_test_Variation = np.array(x)
        
        # x = []
        # aa = [barshift_Variation_ratio,body_leg_Variation_ratio,dorsi_Variation_ratio,barshift_z_score,body_leg_z_score,dorsi_z_score]
        # x.append(pd.DataFrame(list(map(list, zip(*aa)))))
        # X_test_Variation_ratio = np.array(x)

        # X_test_Variation = np.reshape(X_test_Variation,(1,300))
        # X_test_Variation_ratio = np.reshape(X_test_Variation_ratio,(1,300))
        # self.model_predict(X_test_Variation)

        #### stupid Miles's model ####
        ############ depth ############
        x_2 = []
        aa_2 = [barshift_Variation, body_leg_Variation, dorsi_Variation,
                barshift_Variation_ratio, body_leg_Variation_ratio, dorsi_Variation_ratio, angle_variation_ratio_Variation_ratio,
                barshift_z_score, angle_variation_ratio_z_score]   
        x_2.append(pd.DataFrame(list(map(list, zip(*aa_2)))))
        X_test_depth = np.array(x_2)
        X_test_depth = np.reshape(X_test_depth,(1,50,9))
        ############ lean_forward_backeard ############
        x_3_4 = []
        aa_3_4 = [body_leg_Variation, dorsi_Variation, angle_variation_ratio_Variation,
                  barshift_Variation_ratio, body_leg_Variation_ratio, dorsi_Variation_ratio,
                  barshift_z_score, dorsi_z_score]    
        x_3_4.append(pd.DataFrame(list(map(list, zip(*aa_3_4)))))
        X_test_lean_forward_backeard = np.array(x_3_4)
        X_test_lean_forward_backeard = np.reshape(X_test_lean_forward_backeard,(1,400))
        ##### descending_not_synchronize #####
        x_6_7 = []
        aa_6_7 = [barshift_Variation, body_leg_Variation, dorsi_Variation, angle_variation_ratio_Variation,
                  body_leg_Variation_ratio,
                  dorsi_z_score, angle_variation_ratio_z_score]        
        x_6_7.append(pd.DataFrame(list(map(list, zip(*aa_6_7)))))
        X_test_descending_not_synchronize = np.array(x_6_7)
        X_test_descending_not_synchronize = np.reshape(X_test_descending_not_synchronize,(1,350))
        ##### rise_not_synchronize #####
        x_5 = []
        aa_5 = [barshift_Variation, dorsi_Variation, 
                barshift_z_score, dorsi_z_score]    
        x_5.append(pd.DataFrame(list(map(list, zip(*aa_5)))))
        X_test_rise_not_synchronize = np.array(x_5)
        X_test_rise_not_synchronize = np.reshape(X_test_rise_not_synchronize,(1,50,4))

        self.model_predict(X_test_depth, X_test_lean_forward_backeard, X_test_descending_not_synchronize, X_test_rise_not_synchronize)
        

    def model_predict(self,X_test_depth, X_test_lean_forward_backeard, X_test_descending_not_synchronize, X_test_rise_not_synchronize):
        #### kwei's model ####
        # y_pred = self.model.predict(x_test)
        # # print(y_pred)
        
        # # f_predict = open(savepath_predict, 'w')

        # if y_pred == 1:
        #     self.predict_result.setText("Good")
        #     savepath_predict = self.datapath + str(self.squrt_count-1) + "_Good.txt"
        # else:
        #     self.predict_result.setText("Poor")            
        #     savepath_predict = self.datapath + str(self.squrt_count-1) + "_Poor.txt"
        #     self.model_predict_category(x_test)

        # score = self.model.predict_proba(x_test)[0][1]

        #### sexy Miles's model ####
        ##### depth #####
        y_pred_2 = self.model_2.predict(X_test_depth)
        if y_pred_2 > 0.5:
            output_y_pred_2 = 2
        else:
            output_y_pred_2 = 1
        ##### lean_forward_backeard #####
        y_pred_3_4 = self.model_3_4.predict(X_test_lean_forward_backeard)
        output_y_pred_3_4 = np.argmax(y_pred_3_4)
        if output_y_pred_3_4 == 0:
            output_y_pred_3_4 = 0
        elif output_y_pred_3_4 == 1:
            output_y_pred_3_4 = 3
        else:
            output_y_pred_3_4 = 4
        ##### descending_not_synchronize #####
        y_pred_6_7 = self.model_6_7.predict(X_test_descending_not_synchronize)
        output_y_pred_6_7 = np.argmax(y_pred_6_7)
        if output_y_pred_6_7 == 0:
            output_y_pred_6_7 = 1
        elif output_y_pred_6_7 == 1:
            output_y_pred_6_7 = 6
        else:
            output_y_pred_6_7 = 7
        #### rise_not_synchronize #####
        y_pred_5 = self.model_5.predict(X_test_rise_not_synchronize)
        if y_pred_5 > 0.5:
            output_y_pred_5 = 5
        else:
            output_y_pred_5 = 1
  
        ##### compute score #####
        error_message_complete_text = []
        score = 100
        if output_y_pred_2 == 2:
            score = score - 15
            # self.predict_result.setText("深度不夠低")
            error_message_complete_text.append("Squat too shallow")
            savepath_category = self.datapath + str(self.squrt_count-1) + "_2.txt"
            f_category = open(savepath_category, 'w') 
        if output_y_pred_3_4 == 3:
            score = score - 20
            # self.predict_result.setText("骨盆後傾")
            error_message_complete_text.append("Posterior pelvic tilt")
            savepath_category = self.datapath + str(self.squrt_count-1) + "_3.txt"
            f_category = open(savepath_category, 'w') 
        if output_y_pred_3_4 == 4:
            score = score - 10
            # self.predict_result.setText("骨盆前傾")
            error_message_complete_text.append("Anterior pelvic tilt")
            savepath_category = self.datapath + str(self.squrt_count-1) + "_4.txt"
            f_category = open(savepath_category, 'w') 
        if output_y_pred_6_7 == 6:
            score = score - 27.5
            # self.predict_result.setText("身體下降過程中髖角太大,背曲角度太大")
            error_message_complete_text.append("Excessive knee dominant(descending phase)")
            savepath_category = self.datapath + str(self.squrt_count-1) + "_6.txt"
            f_category = open(savepath_category, 'w') 
        if output_y_pred_6_7 == 7:
            score = score - 27.5
            # self.predict_result.setText("身體下降過程中髖角太小,背曲角度太小")
            error_message_complete_text.append("Excessive hip dominant(descending phase)")
            savepath_category = self.datapath + str(self.squrt_count-1) + "_7.txt"
            f_category = open(savepath_category, 'w') 
        if output_y_pred_5 == 5:
            score = score - 10
            # self.predict_result.setText("身體上升過程中髖角與背屈變化不同步")
            error_message_complete_text.append("Hip rising too fast (ascending phase)")
            savepath_category = self.datapath + str(self.squrt_count-1) + "_5.txt"
            f_category = open(savepath_category, 'w')


        

        ##### determine GOOD or POOR squat #####
        if score == 100:
            # self.predict_result.setText("Perfect Squat!")
            error_message_complete_text.append("Perfect Squat!")
            savepath_category = self.datapath + str(self.squrt_count-1) + "_Good.txt"
            f_category = open(savepath_category, 'w')
        else:
            savepath_category = self.datapath + str(self.squrt_count-1) + "_Poor.txt"
            f_category = open(savepath_category, 'w')

            
        #顯示所有錯誤訊息
        self.predict_result.setText("<br>".join(error_message_complete_text))

        
        ##### save the calculated score #####
        print('predict_score : ', score)
        self.predict_result_score.setText(str(score))
        savepath_predict_score = self.datapath + str(self.squrt_count-1) + "_" + str(score) + ".txt"
        f_predict_score = open(savepath_predict_score, 'w')


        self.predict_result_score.setText(str(score))
        savepath_predict_score = self.datapath + str(self.squrt_count-1) + "_" + str(score) + ".txt"
        f_predict_score = open(savepath_predict_score, 'w')


    def z_score(self,data):
        # data = data.astype("float64")
        data = np.array(data)
        mu = data.mean()
        std = data.std()
        z_score = (data - mu) / std
        # print(z_score.shape)
        z_score = np.reshape(z_score,(50))
        return z_score
    
    def Variation(self,data):
        Variation = [0]
        for value in range(0,len(data)-1):
            if data[value] == 0:
                Variation.append(0)
            else:
                Variation.append((data[value] - data[value+1]))
        Variation = np.array(Variation)
        # print(Variation.shape)
        # Variation = np.reshape(Variation,(1,50))
        return Variation
    
    def Variation_ratio(self,data):
        Variation_ratio = [0]
        for value in range(0,len(data)-1):
            if data[value] == 0:
                Variation_ratio.append(0)
            else:
                VR_value = ((data[value] - data[value+1])/data[value])
                if VR_value < -10: # set the lower limitation
                    Variation_ratio.append(-10)
                elif VR_value > 10: # set the higher limitation
                    Variation_ratio.append(10)
                else:
                    Variation_ratio.append(VR_value)
        Variation_ratio = np.array(Variation_ratio)
        # print(Variation_ratio.shape)
        # Variation_ratio = np.reshape(Variation_ratio,(1,50))
        return Variation_ratio

    @pyqtSlot(np.ndarray,np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.Update_data()
        self.get_arm_of_force()
        if self.showpoint:
            qt_img = cv2.circle(qt_img, (self.x_points,self.y_points), 3, (0,0,255), 3)
        self.image.setPixmap(qt_img)
        self.label1.setText('Squat Count : '+ str(self.squrt_count))
        # self.label2.setText('Finish : '+ str(self.squrt_finish_count))
        # self.label9.setText('angle_Ratio:'+ str(self.thread.angle_Ratio))
        self.Save_squrt_data()
        if self.squrt_count != 0:
        #     # print("**")
            self.preprocessing()

    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)
    
    ############ tab3 ############
    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open File", self.datapath, "Video Files(*.mp4 *.avi *.mkv)") # start path        
        self.video_path = filename
        print("filename :",filename)
        self.label_filepath.setText(f"video path: {self.video_path}")
        self.init_video_info()
        self.set_video_player()


    ###這邊把錯誤訊息改成代碼
    def error_button(self):
        self.video_path = self.datapath
        print("self.datapath: ",self.datapath)
        errors = []  # 創建一個空的錯誤列表
        for i in range(15):
            file_path_good = os.path.join(self.video_path, f"{i}_Good.txt")
            file_path_poor = os.path.join(self.video_path, f"{i}_Poor.txt")
            label_attr_name = f"label_text_{i+1}"
        
            if os.path.exists(file_path_good):
                getattr(self, label_attr_name).setText("GOOD")
            elif os.path.exists(file_path_poor):
                error_messages = []  # 創建一個空的錯誤消息列表
            
                file_path_poor_2 = os.path.join(self.video_path, f"{i}_2.txt")
                file_path_poor_3 = os.path.join(self.video_path, f"{i}_3.txt")
                file_path_poor_4 = os.path.join(self.video_path, f"{i}_4.txt")
                file_path_poor_5 = os.path.join(self.video_path, f"{i}_5.txt")
                file_path_poor_6 = os.path.join(self.video_path, f"{i}_6.txt")
                file_path_poor_7 = os.path.join(self.video_path, f"{i}_7.txt")
                newscore = 100
                if os.path.exists(file_path_poor_2):
                    error_messages.append("1")
                    newscore-=15
                if os.path.exists(file_path_poor_3):
                    error_messages.append("6")
                    newscore-=20
                if os.path.exists(file_path_poor_4):
                    error_messages.append("5")
                    newscore-=10
                if os.path.exists(file_path_poor_5):
                    error_messages.append("4")
                    newscore-=10
                if os.path.exists(file_path_poor_6):
                    error_messages.append("2")
                    newscore-=27.5
                if os.path.exists(file_path_poor_7):
                    error_messages.append("3")
                    newscore-=27.5
            
                if error_messages:
                # 將錯誤消息連結成一個字串，並設置到對應的 label 上
                    getattr(self, label_attr_name).setText(", ".join(error_messages) + ", " + str(newscore))

                errors.extend(error_messages)  # 將錯誤消息添加到 errors 列表中
            else:
                getattr(self, label_attr_name).setText("NONE")
    # 15 button to open each video 
    def open_file_button_1(self):
        self.video_path = self.datapath+'/0_video.mp4'
        self.init_video_info()
        self.set_video_player()
        self.read_file(self.video_path)

    def open_file_button_2(self):
        self.video_path = self.datapath+'/1_video.mp4'
        self.init_video_info()
        self.set_video_player()
        self.read_file(self.video_path)

    def open_file_button_3(self):
        self.video_path = self.datapath+'/2_video.mp4'
        self.init_video_info()
        self.set_video_player()
        self.read_file(self.video_path)

    def open_file_button_4(self):
        self.video_path = self.datapath+'/3_video.mp4'
        self.init_video_info()
        self.set_video_player()
        self.read_file(self.video_path)

    def open_file_button_5(self):
        self.video_path = self.datapath+'/4_video.mp4'
        self.init_video_info()
        self.set_video_player()
        self.read_file(self.video_path)

    def open_file_button_6(self):
        self.video_path = self.datapath+'/5_video.mp4'
        self.init_video_info()
        self.set_video_player()
        self.read_file(self.video_path)

    def open_file_button_7(self):
        self.video_path = self.datapath+'/6_video.mp4'
        self.init_video_info()
        self.set_video_player()
        self.read_file(self.video_path)

    def open_file_button_8(self):
        self.video_path = self.datapath+'/7_video.mp4'
        self.init_video_info()
        self.set_video_player()
        self.read_file(self.video_path)

    def open_file_button_9(self):
        self.video_path = self.datapath+'/8_video.mp4'
        self.init_video_info()
        self.set_video_player()
        self.read_file(self.video_path)

    def open_file_button_10(self):
        self.video_path = self.datapath+'/9_video.mp4'
        self.init_video_info()
        self.set_video_player()
        self.read_file(self.video_path)

    def open_file_button_11(self):
        self.video_path = self.datapath+'/10_video.mp4'
        self.init_video_info()
        self.set_video_player()
        self.read_file(self.video_path)

    def open_file_button_12(self):
        self.video_path = self.datapath+'/11_video.mp4'
        self.init_video_info()
        self.set_video_player()
        self.read_file(self.video_path)

    def open_file_button_13(self):
        self.video_path = self.datapath+'/12_video.mp4'
        self.init_video_info()
        self.set_video_player()
        self.read_file(self.video_path)

    def open_file_button_14(self):
        self.video_path = self.datapath+'/13_video.mp4'
        self.init_video_info()
        self.set_video_player()
        self.read_file(self.video_path)

    def open_file_button_15(self):
        self.video_path = self.datapath+'/14_video.mp4'
        self.init_video_info()
        self.set_video_player()
        self.read_file(self.video_path)


    def init_video_info(self):
        videoinfo = opencv_engine.getvideoinfo(self.video_path)
        self.vc = videoinfo["vc"] 
        self.video_fps = videoinfo["fps"] 
        self.video_total_frame_count = videoinfo["frame_count"] 
        self.video_width = videoinfo["width"]
        self.video_height = videoinfo["height"] 


    def set_video_player(self):
        self.timer = QTimer() # init QTimer
        self.timer.timeout.connect(self.timer_timeout_job) # when timeout, do run one
        self.timer.start(self.speed//self.video_fps) # start Timer, here we set '1000ms//Nfps' while timeout one time
        # self.timer.start(1) # but if CPU can not decode as fast as fps, we set 1 (need decode time)
        print(self.speed)

    def __get_frame_from_frame_no(self, frame_no):
        self.vc.set(1, frame_no)
        ret, frame = self.vc.read()
        # Avoid counting frame(frame_no) more than the total frame(self.video_total_frame_count)
        if frame_no < self.video_total_frame_count:
            self.label_framecnt.setText(f"frame number: {frame_no}/{self.video_total_frame_count}")
        else:
            self.label_framecnt.setText(f"frame number: {self.video_total_frame_count}/{self.video_total_frame_count}")
            self.stop()
        return frame

    def __update_label_frame(self, frame):       
        bytesPerline = 3 * self.video_width
        qimg = QImage(frame, self.video_width, self.video_height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(qimg)

        if self.qpixmap.width()/16 >= self.qpixmap.height()/9: # like 1600/16 > 90/9, height is shorter, align width
            self.qpixmap = self.qpixmap.scaledToWidth(self.qpixmap_fix_width)
        else: # like 1600/16 < 9000/9, width is shorter, align height
            self.qpixmap = self.qpixmap.scaledToHeight(self.qpixmap_fix_height)
        self.label_videoframe.setPixmap(self.qpixmap)
        # self.label_videoframe.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop) # up and left
        self.label_videoframe.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter) # Center

    def __update_label_frame(self, frame):       
        bytesPerline = 3 * self.video_width
        qimg = QImage(frame, self.video_width, self.video_height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(qimg)

        # 新的高度和寬度
        new_height = 630
        new_width = 306

        # 將影像調整為新的大小
        self.qpixmap = self.qpixmap.scaled(new_width, new_height, QtCore.Qt.KeepAspectRatio)
        self.label_videoframe.setPixmap(self.qpixmap)
        self.label_videoframe.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter) # Center


    def play(self):
        self.videoplayer_state = "play"

    def stop(self):
        self.videoplayer_state = "stop"

    def pause(self):
        self.videoplayer_state = "pause"

    def timer_timeout_job(self):
        frame = self.__get_frame_from_frame_no(self.current_frame_no)
        self.__update_label_frame(frame)

        if (self.current_frame_no == self.video_total_frame_count):
            frame = self.video_total_frame_count
        else:
            frame = self.__get_frame_from_frame_no(self.current_frame_no)
            
        if (self.videoplayer_state == "play"):
            self.current_frame_no += 1

        if (self.videoplayer_state == "stop"):
            self.current_frame_no = 0

        if (self.videoplayer_state == "pause"):
            self.current_frame_no = self.current_frame_no
        ### update vertical line
        self.graphWidget_3.clear()
        self.graphWidget_4.clear()
        self.graphWidget_5.clear()
        self.graphWidget_6.clear()
        ### vertical line bar_shift
        self.x_line_bar_shift = [self.current_frame_no for _ in range(-70,180)]
        self.y_line_bar_shift = []
        for i in range(-70,180):
            self.y_line_bar_shift.append(i)
        ### vertical line body_leg
        self.x_line_body_leg = [self.current_frame_no for _ in range(70,180)]
        self.y_line_body_leg = []
        for i in range(70,180):
            self.y_line_body_leg.append(i)
        ### vertical line dorsi
        self.x_line_dorsi = [self.current_frame_no for _ in range(50)]
        self.y_line_dorsi = []
        for i in range(50):
            self.y_line_dorsi.append(i)
        ### vertical line angle_ratio
        self.x_line_angle_ratio = [self.current_frame_no for _ in range(-15, 20)]
        self.y_line_angle_ratio = []
        for i in range(-15,20):
            self.y_line_angle_ratio.append(i)

        self.read_file(self.video_path)


    def double(self):
        self.speed = 2000
        self.set_video_player()
        self.label_speed.setText("Now Speed : X 0.5")

    def quarter(self):
        self.speed = 4000
        self.set_video_player()
        self.label_speed.setText("Now Speed : X 0.25")

    def defult(self):
        self.speed = 1000
        self.set_video_player()
        self.label_speed.setText("Now Speed : X 1")

    def read_file(self, video_path):
        file_num = video_path.split('/')[-1].split('_')[0] # find which selected file number
        new_video_path = re.sub(video_path.split('/',-1)[-1],"",video_path)

        #### read bar_shift data and draw it ####
        # read_bar_shift = loadtxt(new_video_path + str(file_num) + "_bar_shift.txt", delimiter="\n")
        read_bar_shift = loadtxt(new_video_path + str(file_num) + "_bar_shift.txt")
        x_bar_shift = list(range(len(read_bar_shift)))
        self.graphWidget_3.plot(x_bar_shift, read_bar_shift, pen = pg.mkPen(color='r'))
        self.graphWidget_3.plot(self.x_line_bar_shift, self.y_line_bar_shift, pen = pg.mkPen(color='g'), symbolBrush=('g'))

        #### read body_leg data and draw it ####
        # read_body_leg = loadtxt(new_video_path + str(file_num) + "_body_leg.txt", delimiter="\n")
        read_body_leg = loadtxt(new_video_path + str(file_num) + "_body_leg.txt")
        x_body_leg = list(range(len(read_body_leg)))
        self.graphWidget_4.plot(x_body_leg, read_body_leg, pen = pg.mkPen(color='r'))
        self.graphWidget_4.plot(self.x_line_body_leg, self.y_line_body_leg, pen = pg.mkPen(color='g'), symbolBrush=('g'))

        #### read dorsiflexion data and draw it ####
        # read_dorsi = loadtxt(new_video_path + str(file_num) + "_dorsi.txt", delimiter="\n")
        read_dorsi = loadtxt(new_video_path + str(file_num) + "_dorsi.txt")
        x_dorsi = list(range(len(read_dorsi)))
        self.graphWidget_5.plot(x_dorsi, read_dorsi, pen = pg.mkPen(color='r'))
        self.graphWidget_5.plot(self.x_line_dorsi, self.y_line_dorsi, pen = pg.mkPen(color='g'), symbolBrush=('g'))

        #### read angle-variation-ratio and draw it ####
        read_angle_ratio = loadtxt(new_video_path + str(file_num) + "_angle_ratio.txt")
        x_angle_ratio = list(range(len(read_angle_ratio)))
        self.graphWidget_6.plot(x_angle_ratio, read_angle_ratio, pen = pg.mkPen(color='r'))
        self.graphWidget_6.plot(self.x_line_angle_ratio, self.y_line_angle_ratio, pen = pg.mkPen(color='g'), symbolBrush=('g'))


class opencv_engine(object):

    @staticmethod
    def point_float_to_int(point):
        return (int(point[0]), int(point[1]))

    @staticmethod
    def read_image(file_path):
        return cv2.imread(file_path)

    @staticmethod
    def draw_point(img, point=(0, 0), color = (0, 0, 255)): # red
        point = opencv_engine.point_float_to_int(point)
        print(f"get {point=}")
        point_size = 1
        thickness = 4
        return cv2.circle(img, point, point_size, color, thickness)

    @staticmethod
    def draw_line(img, start_point = (0, 0), end_point = (0, 0), color = (0, 255, 0)): # green
        start_point = opencv_engine.point_float_to_int(start_point)
        end_point = opencv_engine.point_float_to_int(end_point)
        thickness = 3 # width
        return cv2.line(img, start_point, end_point, color, thickness)

    @staticmethod
    def draw_rectangle_by_points(img, left_up=(0, 0), right_down=(0, 0), color = (0, 0, 255)): # red
        left_up = opencv_engine.point_float_to_int(left_up)
        right_down = opencv_engine.point_float_to_int(right_down)
        thickness = 2 # 寬度 (-1 表示填滿)
        return cv2.rectangle(img, left_up, right_down, color, thickness)

    @staticmethod
    def draw_rectangle_by_xywh(img, xywh=(0, 0, 0, 0), color = (0, 0, 255)): # red
        left_up = opencv_engine.point_float_to_int((xywh[0], xywh[1]))
        right_down = opencv_engine.point_float_to_int((xywh[0]+xywh[2], xywh[1]+xywh[3]))
        thickness = 2 # 寬度 (-1 表示填滿)
        return cv2.rectangle(img, left_up, right_down, color, thickness)

    @staticmethod
    def getvideoinfo(video_path): 
        videoinfo = {}
        vc = cv2.VideoCapture(video_path)
        videoinfo["vc"] = vc
        videoinfo["fps"] = vc.get(cv2.CAP_PROP_FPS)
        videoinfo["frame_count"] = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        videoinfo["width"] = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        videoinfo["height"] = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return videoinfo


if __name__=="__main__":
    # app = QApplication(sys.argv)
    # a = App()
    # a.show()
    import sys
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.closeEvent = lambda event:ui.closeEvent(event)
    MainWindow.show()    
    sys.exit(app.exec_())