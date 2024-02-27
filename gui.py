from os import getcwd

import os
from imageio import imread
from matplotlib import image
import numpy as np 
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.pyplot import get
from PyQt5.QtCore import pyqtSlot, QTimer, QDate
import cv2, imutils
from numpy import rec
from depth_gui import DepthEstimator
from vo_gui import PoseEstimator

class Ui_OFCPose(QtWidgets.QWidget):
    def setupUi(self, OFCPose):
        OFCPose.setObjectName("OFCPose")
        OFCPose.resize(995, 795)
        
        self.VO = PoseEstimator()
        self.DISP = DepthEstimator()
        self.traj = np.zeros((340,340,3), dtype=np.uint8)
        self.groundTruthTraj = np.zeros((340,340,3), dtype=np.uint8)
        self.imagePath = None
        self.groundTruthPath = None
        self.pixmap =  QPixmap('./black.jpg')
        
        self.centralwidget = QtWidgets.QWidget(OFCPose)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(90, 650, 801, 81))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.PoseSelect = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.PoseSelect.setObjectName("PoseSelect")
        self.verticalLayout.addWidget(self.PoseSelect)
        self.DepthSelect = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.DepthSelect.setObjectName("DepthSelect")
        self.verticalLayout.addWidget(self.DepthSelect)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.InferenceRun = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.InferenceRun.setCheckable(True)
        self.InferenceRun.setObjectName("InferenceRun")
        self.horizontalLayout.addWidget(self.InferenceRun)
        self.Cancel = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.Cancel.setCheckable(False)
        self.Cancel.setObjectName("Cancel")
        self.horizontalLayout.addWidget(self.Cancel)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(90, 480, 231, 71))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(30)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.OpenImageFolder = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.OpenImageFolder.setDefault(False)
        self.OpenImageFolder.setFlat(False)
        self.OpenImageFolder.setObjectName("OpenImageFolder")
        self.horizontalLayout_2.addWidget(self.OpenImageFolder)
        self.directoryPath = QtWidgets.QLabel(self.centralwidget)
        self.directoryPath.setGeometry(QtCore.QRect(440, 500, 451, 51))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.directoryPath.sizePolicy().hasHeightForWidth())
        self.directoryPath.setSizePolicy(sizePolicy)
        self.directoryPath.setText("")
        self.directoryPath.setObjectName("directoryPath")
        self.directoryPath.setWordWrap(True)
        self.vo_frame = QtWidgets.QLabel(self.centralwidget)
        self.vo_frame.setGeometry(QtCore.QRect(510, 90, 391, 341))
        self.vo_frame.setFrameShape(QtWidgets.QFrame.Box)
        self.vo_frame.setText("")
        self.vo_frame.setTextFormat(QtCore.Qt.PlainText)
        self.vo_frame.setObjectName("vo_frame")
        self.directortyPathText = QtWidgets.QLabel(self.centralwidget)
        self.directortyPathText.setGeometry(QtCore.QRect(330, 480, 121, 69))
        self.directortyPathText.setFrameShadow(QtWidgets.QFrame.Raised)
        self.directortyPathText.setObjectName("directortyPathText")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(90, 90, 391, 381))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.image_frame = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.image_frame.setFrameShape(QtWidgets.QFrame.Box)
        self.image_frame.setText("")
        self.image_frame.setObjectName("image_frame")
        self.image_frame.setFixedSize(389,159)
        self.verticalLayout_2.addWidget(self.image_frame)
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label_2.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.depth_frame = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.depth_frame.setEnabled(True)
        self.depth_frame.setFrameShape(QtWidgets.QFrame.Box)
        self.depth_frame.setText("")
        self.depth_frame.setObjectName("depth_frame")
        self.depth_frame.setFixedSize(389,159)
        self.verticalLayout_2.addWidget(self.depth_frame)
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setBaseSize(QtCore.QSize(0, 0))
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(310, 10, 341, 61))
        self.label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.label.setObjectName("label")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(510, 440, 389, 17))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setMinimumSize(QtCore.QSize(0, 10))
        self.label_4.setBaseSize(QtCore.QSize(0, 0))
        self.label_4.setObjectName("label_4")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(90, 560, 231, 71))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(30)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.GroundTruthSelect = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.GroundTruthSelect.setDefault(False)
        self.GroundTruthSelect.setFlat(False)
        self.GroundTruthSelect.setObjectName("GroundTruthSelect")
        self.horizontalLayout_3.addWidget(self.GroundTruthSelect)
        self.groundtruthpathtext = QtWidgets.QLabel(self.centralwidget)
        self.groundtruthpathtext.setGeometry(QtCore.QRect(330, 560, 141, 69))
        self.groundtruthpathtext.setFrameShadow(QtWidgets.QFrame.Raised)
        self.groundtruthpathtext.setObjectName("groundtruthpathtext")
        OFCPose.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(OFCPose)
        self.statusbar.setObjectName("statusbar")
        self.directoryPath_2 = QtWidgets.QLabel(self.centralwidget)
        self.directoryPath_2.setGeometry(QtCore.QRect(460, 580, 431, 51))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.directoryPath_2.sizePolicy().hasHeightForWidth())
        self.directoryPath_2.setSizePolicy(sizePolicy)
        self.directoryPath_2.setText("")
        self.directoryPath_2.setWordWrap(True)
        self.directoryPath_2.setObjectName("directoryPath_2")
        OFCPose.setStatusBar(self.statusbar)
        self.timer = QTimer(self)
        self.retranslateUi(OFCPose)
        QtCore.QMetaObject.connectSlotsByName(OFCPose)
        self.OpenImageFolder.clicked.connect(self.selectTarget)
        self.InferenceRun.clicked.connect(self.startSequence)
        self.Cancel.clicked.connect(self.stopSequence)
        self.GroundTruthSelect.clicked.connect(self.selectTargetGroundTruth)
        self.imagePath = './dataset/sequences/09/image_2/'
        _, _, files = next(os.walk('./dataset/sequences/09/image_2/'))
        self.image_count = len(files)

    def retranslateUi(self, OFCPose):
        _translate = QtCore.QCoreApplication.translate
        OFCPose.setWindowTitle(_translate("OFCPose", "AFPose"))
        self.PoseSelect.setText(_translate("OFCPose", "Pose Prediction"))
        self.DepthSelect.setText(_translate("OFCPose", "Depth Prediction"))
        self.InferenceRun.setText(_translate("OFCPose", "Run "))
        self.Cancel.setText(_translate("OFCPose", "Cancel"))
        self.OpenImageFolder.setText(_translate("OFCPose", "Select KITTI Image Sequence"))
        self.directortyPathText.setText(_translate("OFCPose", "Sequence Path: "))
        self.label_2.setText(_translate("OFCPose", "Raw Image Input"))
        self.label_3.setText(_translate("OFCPose", "Depth Prediction"))
        self.label.setText(_translate("OFCPose", "Visual Odometry Estimation with Deep Learning"))
        self.label_4.setText(_translate("OFCPose", "Pose (Prediction vs Ground Truth)"))
        self.GroundTruthSelect.setText(_translate("OFCPose", "Select KITTI Ground Truth File"))
        self.groundtruthpathtext.setText(_translate("OFCPose", "Ground Truth File: "))
     

    def stopSequence(self):
        self.VO.setInitialState()
        self.timer.timeout.disconnect()
        self.InferenceRun.clicked.disconnect()
        self.current_frame = 0
        self.pixmap = QPixmap('./black.jpg')
        self.traj = np.zeros((340,340,3), dtype=np.uint8)
      
        self.vo_frame.setPixmap(self.pixmap)
        self.vo_frame.setScaledContents(True)
        self.image_frame.setPixmap(self.pixmap)
        self.PoseSelect.setEnabled (True)
        if(self.InferenceRun.isChecked):
            self.InferenceRun.setChecked (False)
        self.InferenceRun.setEnabled (True)
        self.InferenceRun.clicked.connect(self.startSequence)

        
    def selectTarget(self):
        dialog = QtWidgets.QFileDialog
        imagePath = dialog.getExistingDirectory(parent=self,
                                caption ='Select Image Sequence',
                                directory =os.getcwd(),
        )        
        self.imagePath = imagePath
        self.directoryPath.setText(imagePath)
        _, _, files = next(os.walk(imagePath))
        self.image_count = len(files)
        self.VO.setImagePath(imagePath)
    
    def selectTargetGroundTruth(self):
        dialog = QtWidgets.QFileDialog
        groundTruthPath = dialog.getOpenFileName(parent=self,
                                caption ='Select Ground Truth Text File'                  
        )[0]        
        self.groundTruthPath = groundTruthPath
        self.directoryPath_2.setText(groundTruthPath)
 
        self.VO.setGroundTruthPath(groundTruthPath)

    def startSequence(self):      
        self.current_frame = 0
        self.timer.timeout.connect(self.update_frame)  # Connect timeout to the output function
        self.timer.start(1)
        self.InferenceRun.setEnabled(False)
        

    def update_frame(self):
        """
        each frame is updated
        """  
        if(self.image_count == self.current_frame + 1):
            self.timer.timeout.disconnect()
            self.InferenceRun.clicked.disconnect()
            self.current_frame = 0
            self.InferenceRun.clicked.connect(self.startSequence)
          
            
        else:
            self.image = cv2.imread(self.imagePath +'/' + str(self.current_frame).zfill(6)+'.png')
            self.NextImage = cv2.imread(self.imagePath +'/' + str(self.current_frame + 1).zfill(6)+'.png')
            self.displayImage(self.image)  

            if(self.DepthSelect.isChecked()):
                img = imread(self.imagePath +'/' + str(self.current_frame).zfill(6)+'.png').astype(np.float32)
                disparityImage = self.DISP.calculateDepth(img)           
                self.displayDepthImage(disparityImage)            
            
            if(self.image_count > self.current_frame + 1 and self.PoseSelect.isChecked()):
                self.PoseSelect.setEnabled (False)
                print(self.current_frame, "current frame")
                self.VO.calculatePose(self.VO.test_files[self.current_frame], self.VO.test_files[self.current_frame+1], self.current_frame)
                self.displayPose()
        
    def displayImage(self, image):
        """
        Display the raw image
        """          
  
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        self.image_frame.setPixmap(QPixmap.fromImage(outImage))
        self.image_frame.setScaledContents(True)
        self.current_frame += 1
        
    def displayDepthImage(self, image):
        """
        Display the depth image
        """          
      
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        self.depth_frame.setPixmap(QPixmap.fromImage(outImage))
        self.depth_frame.setScaledContents(True)
 
    def displayPose(self):
    
        start_coordinateX = 150
        start_coordinateY = 50
         
        scale = 3
        cv2.circle(self.traj, (int(self.VO.predX/scale)+start_coordinateX,int(self.VO.predZ/scale)+start_coordinateY), 1, (0,0,255), 2)
        cv2.circle(self.traj, (int(self.VO.trueX/scale)+start_coordinateX,int(self.VO.trueZ/scale)+start_coordinateY), 1, (0,255,0), 2)
             
        image = self.traj
        
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        self.pixmap = QPixmap.fromImage(outImage)        
     
        self.vo_frame.setPixmap(self.pixmap)
        self.vo_frame.setScaledContents(True)
    


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    OFCPose = QtWidgets.QMainWindow()
    ui = Ui_OFCPose()
    ui.setupUi(OFCPose)
    OFCPose.show()
    sys.exit(app.exec_())
