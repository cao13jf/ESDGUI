import os
import cv2
import sys
import time
import warnings
import argparse
import numpy as np
import pandas as pd
from threading import Thread
import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.Qt import *
import torch
import random

from utils.guis import PhaseCom, draw_segmentation, add_text
from utils.parser import ParserUse
from utils.gui_parts import *
from canvas import Canvas
from canvas_video import MainWindow
from utils.sam import *
#from utils.guis import PhaseCom, draw_segmentation, add_text
#from utils.report_tools import generate_report, get_meta

warnings.filterwarnings("ignore")
DEFAULT_STYLE = """
                QProgressBar{
                    border: 2px solid grey;
                    background-color: white;
                    text-align: center;
                    height: 20px;
                }
                QProgressBar::chunk {
                    background-color: #807e7c;
                }
                """
COMBOBOX = """
            QComboBox {
                border: 1px solid grey;
                border-radius: 3px;
                padding: 1px 2px 1px 2px;  
                min-width: 10em;
                min-height: 30px;  
            }
            QComboBox QAbstractItemView::item 
            {
                min-height: 20px;
            }
        """
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    process_img_signal = pyqtSignal(np.ndarray, int)

    def run(self):
        frame_idx = 0
        # capture from web cam
        cap = cv2.VideoCapture(0)  # TODO: Camera input
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        time.sleep(0.5)
        if cap.isOpened():
            while True:
                frame_idx += 1
                ret, cv_img = cap.read()       
                #print(ret)
                time.sleep(0.15)  # TODO: removing sleep for camera
                if ret:
                    #print(cv_img)
                    
                    self.change_pixmap_signal.emit(cv_img)
                    self.process_img_signal.emit(cv_img, frame_idx)
                    
                else:
                    cap.release()
                    assert "Cannot get frame"
        else:
            cap.release()
            assert "Cannot get frame"


class Ui_iPhaser(QMainWindow):
    def __init__(self):
        super(Ui_iPhaser, self).__init__()

    def setupUi(self, cfg):
        self.setObjectName("iPhaser")
        self.resize(1825, 1175)
        self.setMinimumHeight(965)
        self.setMinimumWidth(965)
        self.setStyleSheet("QWidget#iPhaser{background-color: #f8f8f8}")
        old_pos = self.frameGeometry().getRect()
        curr_x = old_pos[2]
        curr_y = old_pos[3]
        self.redo = False
        self.point_size = 1
        self.mQImage = QPixmap('./images/test.jpg')
        self.cbFlag = 0
        self.size = QSize(curr_x-25-500, curr_y-65-250)
        self.old_pos = self.frameGeometry().getRect()
        self.save_folder = "./Records"
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
        self.down_ratio = 1#cfg.down_ratio
        # Statue parameters
        self.init_status()
        Vblocks = ['case information','phase recognition', 'object detection', 'summary report']
        Hblocks = ['training session']
        self.FRAME_WIDTH, self.FRAME_HEIGHT, self.stream_fps = self.get_frame_size()
        self.MANUAL_FRAMES = self.stream_fps * cfg.manual_set_fps_ratio
        self.manual_frame = 0
        self.enable_seg = False
        self.force_seg = False
        self.seg_alpha = 0.2
        # self.FRAME_WIDTH = 1280
        # self.FRAME_HEIGHT = 720
        self.CODEC = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.pbFlag = True
        #self.date_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S.%f")
        self.fps = 0
        self.selected_tool = 0
        self.f_image = None
        self.surgeons = QComboBox()
        self.surgeons.setObjectName("SurgeonsName")
        self.surgeons.setStyleSheet(COMBOBOX)
        self.surgeons.setEditable(True)
        self.surgeons.lineEdit().setAlignment(Qt.AlignCenter)
        self.surgeons.lineEdit().setFont(QFont("Arial", 16, QFont.Bold))
        self.surgeons.lineEdit().setReadOnly(True)
        self.surgeons.setCurrentIndex(-1)
        self.pred = "--"
        self.pop_image = {}
        self.pop_idx = []
        self.total_diff = []
        self.pop_image_count = []
        self.seg_pred = torch.zeros([4, 224, 224])#.numpy()
        self.log_data = []
        self.actionList = {}
        self.last_idx = []
        self.count_image = {}
        self.image_count = []
        self.ru = 0
        self.aaa = 0
        self.bbb = 0
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        # # 在主窗口中添加usbVideo控件
        # self.usbVideo = usbVideo(self.size, parent=self.centralwidget)
        # self.usbVideo.setGeometry(QtCore.QRect(500, 250, curr_x-25-500, curr_y-65-250))
        # newly added
        self.DisplayVideo = QtWidgets.QLabel(self.centralwidget)
        self.DisplayVideo.setGeometry(QtCore.QRect(500, 250, curr_x-25-500, curr_y-65-250))
        self.DisplayVideo.setScaledContents(True)
        #self.DisplayVideo.setStyleSheet("background-color: rgb(197, 197, 197);")
        self.DisplayVideo.setText("")
        self.DisplayVideo.setObjectName("DisplayVideo")
        self.phaseseg = PhaseCom(arg=cfg)
        self.video = False
        self.disply_width = 1080  # TODO: Change resFRAME_WIDTHolutions
        self.display_height = 720
        self.start_x = 0
        self.end_x = 450
        self.start_y = 0
        # self.start_y = 450
        self.end_y = 450
        # cv_img[0:1150, 450:1800]
        self.save_folder = "../Records"
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
        self.down_ratio = 1#cfg.down_ratio
        self.start_time = "--:--:--"
        self.trainee_name = "--"
        self.manual_set = "--"
        self.thread = None
        # model prediction
        # self.centralwidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
        # self.setCentralWidget(self.usbVideo)
        #self.setCentralWidget(self.centralwidget)
        self.label_mask = QImage()
        self.image = QImage()
        self.canvas = Canvas(self.label_mask, self.image, parent=self.centralwidget)
        #self.canvas.setGeometry(QtCore.QRect(500, 250, curr_x-25-500, curr_y-65-250))
        #self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #self.canvas.setScaledContents(True)
        #self.setCentralWidget(self.centralwidget)
        #self.centralWidget.addWidget(self.canvas)
        self.bu = 0
        self.fileButton = QPushButton('File', self)
        self.fileButton.setFont(QFont("Arial", 12, QFont.Bold))
        self.fileButton.setGeometry(QtCore.QRect(0, 0, 60, 25))
        self.fileButton.setStyleSheet("background-color:#dee0e3;cocanvas import Canvaslor:black;")
        self.fileButton.setObjectName('FileButton')
        self.settingButton = QPushButton('Setting', self)
        self.settingButton.setFont(QFont("Arial", 12, QFont.Bold))
        self.settingButton.setGeometry(QtCore.QRect(60, 0, 60, 25))
        self.settingButton.setStyleSheet("background-color:#dee0e3;color:black;")
        self.settingButton.setObjectName('SettingButton')
        self.startButton = QPushButton(self)
        #self.startButton.setFont(QFont("Arial",12, QFont.Bold))
        self.startButton.setGeometry(QtCore.QRect(1550, 70, 80, 80))
        self.startButton.setStyleSheet("background-color:#dee0e3;")
        self.startButton.setObjectName('StartButton')
        self.start_pixmap = QtGui.QIcon()
        self.start_pixmap.addFile("./images/start.png", QtCore.QSize(80,80), QtGui.QIcon.Active, QtGui.QIcon.On)
        self.startButton.setIcon(self.start_pixmap)
        self.startButton.setIconSize(QtCore.QSize(150,150))
        self.startButton.pressed.connect(self.onButtonClickStart)
        self.stopButton = QPushButton(self)
        #self.stopButton.setFont(QFont("Arial",12, QFont.Bold))
        self.stopButton.setGeometry(QtCore.QRect(1650, 70, 80, 80))
        #self.stopButton.setStyleSheet("background-color:#dee0e3;border:10px;padding:10px")
        self.stopButton.setObjectName('StopButton')
        self.stop_pixmap = QtGui.QIcon()
        self.stop_pixmap.addFile("./images/stop.png", QtCore.QSize(80,80), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.stopButton.setIcon(self.stop_pixmap)
        self.stopButton.setIconSize(QtCore.QSize(150,150))
        self.stopButton.clicked.connect(self.onButtonClickStop)
        self.promptEnsure = QPushButton("OK", self)
        self.promptEnsure.setObjectName("PromptEnsureButton")
        self.promptEnsure.setFont(QFont("Arial", 14))
        self.promptEnsure.setGeometry(QtCore.QRect(1550, 155, 80, 80))
        self.promptEnsure.clicked.connect(self.onButtonPromptEnsure)
        self.promptCancel = QPushButton("CANCEL", self)
        self.promptCancel.setObjectName("PromptCancelButton")
        self.promptCancel.setFont(QFont("Arial", 14))
        self.promptCancel.clicked.connect(self.onButtonPromptCancel)
        self.promptCancel.setGeometry(QtCore.QRect(1650, 155, 80, 80))
        
        # video window
        self.promptEnsure = QPushButton("Video ", self)
        self.promptEnsure.setObjectName("Video")
        self.promptEnsure.setFont(QFont("Arial", 14))
        self.promptEnsure.setGeometry(QtCore.QRect(1427, 155, 80, 80))
        self.promptEnsure.clicked.connect(self.open_main_window)
        
        
        
        self.layoutWidget = QtWidgets.QWidget(self)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 20, 440, 870))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.layoutWidget1 = QtWidgets.QWidget(self)
        self.layoutWidget1.setGeometry(QtCore.QRect(490, 20, 920, 225))
        self.layoutWidget.setObjectName("layoutWidget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.VLayout1 = QtWidgets.QVBoxLayout()
        self.VLayout1.setObjectName("VLayout1")
        self.trainLabelTitle = QtWidgets.QLabel('Training Session')
        self.trainLabelTitle.setObjectName("TrainLabelTitle")
        self.trainLabelTitle.setStyleSheet("color:blue;")
        self.trainLabelTitle.setFont(QFont("Arial", 18, QFont.Bold))
        self.VLayout1.addWidget(self.trainLabelTitle, 16)
        self.trainLabel = QtWidgets.QWidget()
        self.trainLabel.setObjectName("TrainLabel")
        self.trainLabel.setAttribute(Qt.WA_StyledBackground, True)
        self.trainLabel.setStyleSheet("background-color: #dee0e3; border-radius:5px")
        self.VLayout1.addWidget(self.trainLabel, 84)
        self.VLayout2 = QtWidgets.QVBoxLayout()
        self.VLayout2.setObjectName("VLayout2")
        self.painterTitle = QtWidgets.QLabel('Painter')
        self.painterTitle.setObjectName("PainterTitle")
        self.painterTitle.setStyleSheet("color:blue;")
        self.painterTitle.setFont(QFont("Arial", 18, QFont.Bold))
        self.VLayout2.addWidget(self.painterTitle, 16)
        self.labelSelector = QComboBox()
        self.labelSelector.setObjectName("LabelSelector")
        self.labelSelector.setStyleSheet(COMBOBOX)
        self.labelSelector.setEditable(True)
        self.labelSelector.lineEdit().setAlignment(Qt.AlignCenter)
        self.labelSelector.lineEdit().setFont(QFont("Arial", 16, QFont.Bold))
        self.labelSelector.lineEdit().setReadOnly(True)
        self.labelSelector.setCurrentIndex(-1)
        self.labelSelector.currentTextChanged.connect(self.on_combobox_changed)
        self.painter = QtWidgets.QWidget()
        self.painter.setObjectName("Painter")
        self.painter.setAttribute(Qt.WA_StyledBackground, True)
        self.painter.setStyleSheet("QWidget#Painter{background-color: #dee0e3; border-radius:5px}")
        self.VLayout2.addWidget(self.painter, 84)
        self.horizontalLayout.addLayout(self.VLayout1, 50)
        self.horizontalLayout.addLayout(self.VLayout2, 50)
        Vpercent = 100 / len(Vblocks)
        Hpercent = 50#100 / len(Hblocks)
        for i in Vblocks:
            self.setVLayout(i, Vpercent)
        
        self.setupCaseInformation()
        self.setupTrainer()
        self.setupPhaseRecog()
        self.setupObjectDetection(4)
        self.setupSummary()
        self.setupPainter()
        self.setCentralWidget(self.centralwidget)
        self.retranslateUi()
    
    
    def init_video(self):
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot   
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.process_img_signal.connect(self.process_img)
        # start the thread
        self.thread.start()

    def update_image(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        # Collect settings of functional keys
        # cv_img = cv_img[30:1050, 695:1850]

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.manual_frame = self.manual_frame - 1
        if self.manual_frame <= 0:
            self.manual_frame = 0
            self.manual_set = "--"
        if self.INIT:
            self.date_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
            if self.manual_frame > 0:
                self.pred = self.manual_set

            #print(self.seg_pred)
            rgb_image = self.phaseseg.draw_segmentation(self.seg_pred, rgb_image, start_x=self.start_x, end_x=self.end_x, start_y=self.start_y, end_y=self.end_y, alpha=self.seg_alpha)
            rgb_image = self.phaseseg.add_text(self.date_time, self.pred, self.trainee_name, rgb_image)
        if self.WORKING:
            #print('write', rgb_image.shape)
            rbg_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            self.output_video.write(rbg_image)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.size.width(), self.size.height(), Qt.KeepAspectRatio)  # TODO: Change size of display
        p = QPixmap.fromImage(p)
        #print('done')
        #print(self.INIT)
        self.DisplayVideo.setPixmap(p)
        
    def process_img(self, cv_img, frame_idx):
        # print("test")
        # cv_img = cv_img[30:1050, 695:1850]
        cv_img = cv_img[self.start_x:self.end_x, self.start_y:self.end_y]  # Crop images
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.states = {0:self.phase1_state, 1:self.phase2_state, 2:self.phase3_state, 3:self.phase4_state}
        if frame_idx % self.down_ratio == 0 and self.WORKING:
            self.date_time = datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S.%f")
            start_time = time.time()
            import torch
            self.pred, self.prob, index = self.phaseseg.phase_frame(rgb_image)
            for key, val in self.states.items():
                val.setChecked(False)
            self.phase1_prob.setValue(int(self.prob[0]*100))
            self.phase2_prob.setValue(int(self.prob[1]*100))
            self.phase3_prob.setValue(int(self.prob[2]*100))
            self.phase4_prob.setValue(int(self.prob[3]*100))
            self.states[index].setChecked(True)
            print(self.pred)
            self.seg_pred = self.phaseseg.seg_frame(rgb_image)
            
            if self.force_seg:
                self.seg_alpha = 0.2
            elif self.enable_seg and (self.pred == "dissection" or self.manual_set == "dissection"):
                self.seg_alpha = 0.2
            else:
                self.seg_alpha = 0.0

            # self.DisplayPhase.setText(self.pred)
            end_time = time.time()
            self.fps = 1/np.round(end_time - start_time, 3)
            print("Inference FPS {}".format(self.fps, np.round(end_time - start_time, 3)))

    
    def onButtonClickStart(self):
        self.canvas.clear()
        self.flag = True
        self.WORKING = True
        self.video = True
        self.init_video()
        video_file_name = os.path.join(self.save_folder, self.e1.text()+ "_" + self.e1.text() + "_" + self.start_time.replace(":", "-") + ".avi")
        self.output_video = cv2.VideoWriter(video_file_name, self.CODEC, self.stream_fps, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        self.startTime = datetime.datetime.now()
        lineEdits = self.trainLabel.findChildren(QLineEdit)
        for lineEdit in lineEdits:
            if lineEdit.objectName() == 'MentorInput':
                if self.surgeons.findText(lineEdit.text()) == -1 and lineEdit.text()!= '':
                    self.surgeons.addItem(lineEdit.text())
                    self.surgeons.setCurrentIndex(-1)
        
    def pick_color(self):
        color = QColorDialog.getColor()
        idx = self.labelSelector.currentIndex()
        if color.isValid() and idx != -1:
            self.canvas.brush_color = color
            self.color.pop(idx)
            self.color.insert(idx, color)
            label = self.findChild(QLabel, f"Object{idx+1}")
            label.setStyleSheet(f"background-color: {color.name()}")
            
    
    def onButtonClickStop(self):
        self.flag = False
        self.WORKING = False
        self.video = False
        self.output_video.release()
        self.thread.terminate()
        self.thread.wait(1)
    
    def open_main_window(self):
        self.main_window = MainWindow()
        self.main_window.show()
        
    
    def resizeEvent(self, event):
        old_pos = self.frameGeometry().getRect()
        curr_x = old_pos[2]
        curr_y = old_pos[3]
        self.size = QSize(curr_x-25-500, curr_y-65-250)
        self.pos = QSize(500, 250)
        #self.usbVideo.setGeometry(QtCore.QRect(500, 250, curr_x-25-500, curr_y-65-250))
        #self.canvas.label_mask.scaled(curr_x-25-500, curr_y-65-250, QtCore.Qt.KeepAspectRatio)
        
    def setupTrainer(self):
        e1 = QLabel('Mentor:')
        e1.setObjectName("Mentor")
        e1.setFont(QFont("Arial", 14))
        e1.setStyleSheet("color:black;")
        e2 = QLabel('Lesion Location:')
        e2.setObjectName("LesionLocation")
        e2.setFont(QFont("Arial", 14))
        e2.setStyleSheet("color:black;")
        e3 = QLineEdit()
        e3.setFixedHeight(35)
        e3.setObjectName("MentorInput")
        e3.setStyleSheet("background-color: white;color:black")
        e3.setFont(QFont("Arial", 14))
        e3.setAlignment(Qt.AlignCenter)
        e4 = QLineEdit()
        e4.setFixedHeight(35)
        e4.setObjectName("LesionLocationInput")
        e4.setStyleSheet("background-color: white;color:black")
        e4.setFont(QFont("Arial", 14))
        e4.setAlignment(Qt.AlignCenter)
        e5 = QLabel("Trainee:")
        e5.setObjectName("Trainee")
        e5.setFont(QFont("Arial", 14))
        e5.setStyleSheet("color:black;")
        e6 = QLabel('Bed:')
        e6.setObjectName("Bed")
        e6.setFont(QFont("Arial", 14))
        e6.setStyleSheet("color:black;")
        e7 = QLineEdit()
        e7.setFixedHeight(35)
        e7.setObjectName("TraineeInput")
        e7.setStyleSheet("background-color: white;color:black")
        e7.setFont(QFont("Arial", 14))
        e7.setAlignment(Qt.AlignCenter)
        e8 = QLineEdit()
        e8.setFixedHeight(35)
        e8.setObjectName("BedInput")
        e8.setStyleSheet("background-color: white;color:black")
        e8.setFont(QFont("Arial", 14))
        e8.setAlignment(Qt.AlignCenter)
        e = QGridLayout()
        e.addWidget(e1, 0, 0)
        e.addWidget(e2, 0, 1)
        e.addWidget(e3, 1, 0)
        e.addWidget(e4, 1, 1)
        e.addWidget(e5, 2, 0)
        e.addWidget(e6, 2, 1)
        e.addWidget(e7, 3, 0)
        e.addWidget(e8, 3, 1)
        self.trainLabel.setLayout(e)
    
    def setupPainter(self):
        egrid = QGridLayout()
        ehlay = QHBoxLayout()
        ehlay2 = QHBoxLayout()
        ehlay3 = QHBoxLayout()
        ehlay4 = QHBoxLayout()
        e00 = QPushButton('Add New Label')
        e00.setObjectName("AddLabelButton")
        e00.setFont(QFont("Arial", 14))
        e00.clicked.connect(self.onButtonAddLabel)
        e0 = QLabel('Label Selector:')
        e0.setObjectName("LabelSelector")
        e0.setFont(QFont("Arial", 14))
        e1 = QPushButton('Load')
        e1.setObjectName("LoadImageButton")
        e1.setFont(QFont("Arial", 14))
        e1.clicked.connect(self.load_image)
        e2 = QPushButton('Erase')
        e2.setObjectName("EraseButton")
        e2.setFont(QFont("Arial", 14))
        e2.clicked.connect(self.onButtonErase)
        e3 = QPushButton('Paint')
        e3.setObjectName("PaintButton")
        e3.setFont(QFont("Arial", 14))
        e3.clicked.connect(self.onButtonPaint)
        e7 = QPushButton('SAM')
        e7.setObjectName("SAMButton")
        e7.setFont(QFont("Arial", 14))
        e7.clicked.connect(self.onButtonPredict)
        e4 = QPushButton('Save')
        e4.setObjectName("SaveButton")
        e4.setFont(QFont("Arial", 14))
        e4.clicked.connect(self.onButtonSave)
        e5 = QPushButton('Color')
        e5.setObjectName("PickColorButton")
        e5.setFont(QFont("Arial", 14))
        e5.clicked.connect(self.pick_color)
        #e7.clicked.connect(self.onButtonRedo)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setObjectName("ThicknessSlider")
        self.slider.setMinimum(1)
        self.slider.setMaximum(25)
        self.slider.setSingleStep(1)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.onSliderValueChanged)
        self.thickness = QLineEdit('1')
        self.thickness.setObjectName("ThicknessInput")
        self.thickness.setStyleSheet("background-color: white;color:black")
        self.thickness.setAlignment(Qt.AlignCenter)
        self.thickness.setFont(QFont("Arial", 14))
        self.thickness.setFixedWidth(50)
        self.thickness.textEdited.connect(self.onLineEditsChanged)
        self.beginLabel = QLabel('Begin: (0, 0)')
        self.beginLabel.setObjectName("BeginLabel")
        self.beginLabel.setFont(QFont("Arial", 14))
        self.beginLabel.setAlignment(Qt.AlignLeft)
        self.endLabel = QLabel('End: (0, 0)')
        self.endLabel.setObjectName("EndLabel")
        self.endLabel.setFont(QFont("Arial", 14))
        self.endLabel.setAlignment(Qt.AlignLeft)
        ehlay.addWidget(e00)
        ehlay.addWidget(e0)
        ehlay.addWidget(self.labelSelector)
        ehlay2.addWidget(e1)
        ehlay2.addWidget(e2)
        ehlay2.addWidget(e3)
        ehlay2.addWidget(e7)
        ehlay2.addWidget(e4)
        ehlay2.addWidget(e5)
        ehlay3.addWidget(self.slider)
        ehlay3.addWidget(self.thickness)
        ehlay4.addWidget(self.beginLabel)
        ehlay4.addWidget(self.endLabel)
        egrid.addLayout(ehlay, 0, 0)
        egrid.addLayout(ehlay2, 1, 0)
        egrid.addLayout(ehlay3, 2, 0)
        egrid.addLayout(ehlay4, 3, 0)
        self.painter.setLayout(egrid)
    
    
    def onButtonPromptEnsure(self):
        box_0 = self.beginLabel.text().replace(" ", "")
        box_0 = box_0.split('(')[1].split(')')[0].strip(' ')
        box_1 = self.endLabel.text().replace(" ", "")
        box_1 = box_1.split('(')[1].split(')')[0].strip(' ')
        assert box_0.split(',')[0].isnumeric() and box_0.split(',')[1].isnumeric()
        assert box_1.split(',')[0].isnumeric() and box_1.split(',')[1].isnumeric()
        box_0x = int(box_0.split(',')[0])
        box_0y = int(box_0.split(',')[1])
        box_1x = int(box_1.split(',')[0])
        box_1y = int(box_1.split(',')[1])
        input_box = np.array([box_0x, box_0y, box_1x, box_1y])
        device = torch.device('cuda:0')
        image_dim = (self.canvas.image.size().width(), self.canvas.image.size().height())
        masks, overlay = prompt_sam_predict(self.filepath, input_box, image_dim, device=device)
        height, width, channel = overlay.shape
        bytes_per_line = 3 * width
        image = QImage(overlay, width, height, bytes_per_line, QImage.Format_RGB888).scaled(self.size.width(), self.size.height(), Qt.KeepAspectRatio)
        self.canvas.image = image
        self.label_mask = QImage(image.size(), QImage.Format_ARGB32)
        self.label_mask.fill(Qt.transparent)
        self.canvas.label_mask = self.label_mask
        self.canvas.setPixmap(QPixmap.fromImage(image))
        self.canvas.setGeometry(QtCore.QRect(500, 250, image.size().width(), image.size().height()))
        self.labelSelector.setCurrentIndex(-1)


    
    def onButtonPromptCancel(self):
        pass
    
    def load_image(self):
        if self.video:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Please stop the video before loading a new image")
            msg.setWindowTitle("Warning")
            msg.exec_()
        else:
            self.DisplayVideo.clear()
            file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.xpm *.jpg *.bmp)')
            self.filepath = file_path
            if file_path:
                image = QImage(file_path).scaled(self.size.width(), self.size.height(), Qt.KeepAspectRatio)
                self.canvas.image = image.convertToFormat(QImage.Format_RGB32)
                self.canvas.beginLabel = self.beginLabel
                self.canvas.endLabel = self.endLabel
                self.label_mask = QImage(image.size(), QImage.Format_ARGB32)
                self.label_mask.fill(Qt.transparent)
                self.canvas.label_mask = self.label_mask
                self.canvas.setPixmap(QPixmap.fromImage(image))
                self.canvas.setGeometry(QtCore.QRect(500, 250, self.size.width(), self.size.height()))
                self.labelSelector.setCurrentIndex(-1)
    
    def on_combobox_changed(self):
        if self.labelSelector.currentIndex() != -1:
            idx = self.labelSelector.currentIndex()
            if self.ru != 0:
                self.canvas.brush_color = self.color[idx]
                self.last_idx.append(self.labelSelector.currentIndex())
                if str(self.labelSelector.currentIndex()) not in self.count_image.keys():
                    self.count_image[str(self.labelSelector.currentIndex())] = 0
                self.image_count.append(self.count_image[str(self.labelSelector.currentIndex())])
                self.bbb = len(self.image_count)
            self.ru += 1
            
    def onSliderValueChanged(self, value):
        self.canvas.brush_size = value
        self.thickness.setText(str(value))
    
    def onLineEditsChanged(self, text):
        if text.isnumeric() and int(text) <= self.slider.maximum() and int(text) >= self.slider.minimum():
            self.canvas.brush_size = int(text)
            self.slider.setValue(int(text))
            self.pbFlag = True
        elif text.isnumeric() and int(text) > self.slider.maximum():
            self.thickness.setText(str(self.slider.maximum()))
            self.canvas.brush_size = self.slider.maximum()
            self.slider.setValue(self.slider.maximum())
            self.pbFlag = True
        elif text.isnumeric() and int(text) < self.slider.minimum():
            self.thickness.setText(str(self.slider.minimum()))
            self.canvas.brush_size = self.slider.minimum()
            self.slider.setValue(self.slider.minimum())
            self.pbFlag = True
        else:
            self.canvas.brush_size = 0
            self.pbFlag = True
    
    def onButtonPaint(self):
        idx = self.labelSelector.currentIndex()
        if idx != -1:
            self.canvas.brush_color = self.color[idx]
            self.canvas.erase = False
    
    def onButtonErase(self):
        self.canvas.erase = True
        #self.canvas.brush_color.setAlphaF(0.01)
    
    def onButtonSave(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save Label Mask', '', 'Images (*.png)')
        if file_path:
            self.canvas.label_mask.save(file_path)
    
    def onButtonAddLabel(self):
        new_color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        new_r, new_g, new_b = hex_to_rgb(new_color)
        new_color1 = QColor(new_b, new_g, new_r)
        if new_color1 not in self.color:
            dlg = CustomMB(self.labelSelector)
            dlg.exec_()
            if dlg.finished:
                self.labelSelector.addItem(dlg.getText())
                self.color.append(new_color1)
                count = self.labelSelector.count()
                self.labelSelector.setCurrentIndex(count-1)
                e1_group = QGroupBox()
                e1_group.setObjectName(f"Object{count}Group")
                e1_group.setStyleSheet(f"QGroupBox#Object{count}Group"+"{border:0;}")
                e1_button = QRadioButton()
                e1_button.setChecked(True)
                e1_button.setStyleSheet(
                                        "QRadioButton"
                                        "{"
                                        "color : green;"
                                        "}"
                                        "QRadioButton::indicator"
                                        "{"
                                        "width : 20px;"
                                        "height : 20px;"
                                        "}")
                e1_button.setObjectName(f"Object{count}Button")
                e1 = QLabel()
                e1.setObjectName(f"Object{count}")
                e1.setStyleSheet(f"background-color : {new_color};")
                e1.setFixedWidth(20)
                e1.setFixedHeight(20)
                e2 = QLabel(dlg.getText().title())
                e2.setObjectName(f"Object{count}Name")
                e2.setFont(QFont("Arial", 14, QFont.Bold))
                e2.setStyleSheet("color:black;")
                hbox_1 = QHBoxLayout()
                hbox_1.setObjectName(f"Object{count}Layout")
                hbox_1.addWidget(e1_button)
                hbox_1.addWidget(e1)
                hbox_1.addWidget(e2)
                hbox_1.setAlignment(Qt.AlignLeft)
                e1_group.setLayout(hbox_1)
                self.elayout.addWidget(e1_group, count-1, 0)
                self.elayout.setAlignment(Qt.AlignLeft)
    
    
    def onButtonPredict(self):
        self.SAMBOX = SAMMB(self.samFull, self.samPrompt)
        self.SAMBOX.exec_()
        
    def samPrompt(self):
        self.canvas.prompt = True
        
    
    def samFull(self):
        device = torch.device('cuda:0')
        masks, overlay = full_sam_predict(self.filepath, device=device)
        height, width, channel = overlay.shape
        bytes_per_line = 3 * width
        image = QImage(overlay, width, height, bytes_per_line, QImage.Format_RGB888).scaled(self.size.width(), self.size.height(), Qt.KeepAspectRatio)
        self.canvas.image = image
        self.label_mask = QImage(image.size(), QImage.Format_ARGB32)
        self.label_mask.fill(Qt.transparent)
        self.canvas.label_mask = self.label_mask
        self.canvas.setPixmap(QPixmap.fromImage(image))
        self.canvas.setGeometry(QtCore.QRect(500, 250, image.size().width(), image.size().height()))
        self.labelSelector.setCurrentIndex(-1)
    
    
    def setupCaseInformation(self):
        num_widget = self.verticalLayout.count()
        while num_widget > 0:
            widget = self.verticalLayout.itemAt(num_widget-1).widget()
            if widget.objectName() == 'CaseInformation':
                break
            num_widget -= 1
        vlayout = QVBoxLayout(widget)
        vlayout.setObjectName("CaseInformationVlayout")
        self.e1 = QLabel('Patient ID:')
        self.e1.setObjectName("PID")
        self.e1.setFont(QFont("Arial", 14))
        self.e1.setStyleSheet("color:black;")
        vlayout.addWidget(self.e1)
        hlayout = QHBoxLayout()
        e2 = QLineEdit()
        e2.setFixedHeight(35)
        e2.setFixedWidth(180)
        e2.setObjectName("PID1")
        e2.setStyleSheet("background: white;border-radius:5px;color: black")
        e2.setAlignment(Qt.AlignCenter)
        e2.setFont(QFont("Arial", 14))
        hlayout.addWidget(e2)
        e3 = QFrame()
        e3.setFrameShape(QFrame.HLine)
        e3.setFrameShadow(QFrame.Plain)
        e3.setLineWidth(2)
        e3.setObjectName("PIDSpace")
        hlayout.addWidget(e3)
        e4 = QLineEdit()
        e4.setFixedHeight(35)
        e4.setFixedWidth(180)
        e4.setObjectName("PID2")
        e4.setStyleSheet("background: white;border-radius:5px;color:black")
        e4.setAlignment(Qt.AlignCenter)
        e4.setFont(QFont("Arial", 14))
        hlayout.addWidget(e4)
        hlayout.setSpacing(15)
        vlayout.addLayout(hlayout)
        e5 = QLabel('Date:')
        e5.setObjectName("PIDDate")
        e5.setFont(QFont("Arial", 14))
        e5.setStyleSheet("color:black;")
        vlayout.addWidget(e5)
        hlayout1 = QHBoxLayout()
        e6 = QLineEdit()
        e6.setFixedHeight(35)
        e6.setFixedWidth(105)
        e6.setObjectName("PIDDateYear")
        e6.setStyleSheet("background-color: white;border-radius:5px;color:black")
        e6.setAlignment(Qt.AlignCenter)
        e6.setFont(QFont("Arial", 14))
        hlayout1.addWidget(e6)
        e7 = QFrame()
        e7.setFrameShape(QFrame.HLine)
        e7.setFrameShadow(QFrame.Plain)
        e7.setLineWidth(2)
        e7.setObjectName("PIDDateSpace")
        hlayout1.addWidget(e7)
        e8 = QLineEdit()
        hlayout1.addWidget(e8)
        e9 = QFrame()
        e9.setFrameShape(QFrame.HLine)
        e9.setFrameShadow(QFrame.Plain)
        e9.setLineWidth(2)
        e9.setObjectName("PIDDateSpace1")
        hlayout1.addWidget(e9)
        e10 = QLineEdit()
        e10.setFixedHeight(35)
        e10.setFixedWidth(105)
        e10.setObjectName("PIDDateDay")
        e10.setStyleSheet("background-color: white;border-radius:5px;color: black")
        e10.setAlignment(Qt.AlignCenter)
        e10.setFont(QFont("Arial", 14))
        hlayout1.addWidget(e10)
        hlayout1.setSpacing(15)
        vlayout.addLayout(hlayout1)
        
    def setupPhaseRecog(self):
        num_widget = self.verticalLayout.count()
        while num_widget > 0:
            widget = self.verticalLayout.itemAt(num_widget-1).widget()
            if widget.objectName() == 'PhaseRecognition':
                break
            num_widget -= 1
        e1 = QLabel('Idle')
        e1.setObjectName("Idle")
        e1.setFont(QFont("Arial", 16, QFont.Bold))
        e1.setStyleSheet("color:black;")
        self.phase1_state = QRadioButton()
        self.phase1_state.setChecked(False)
        #e2.setTristate(True)
        self.phase1_state.setObjectName("IdleCheck")
        self.phase1_state.setStyleSheet("QRadioButton"
                         "{"
                         "color : green;"
                         "}"
                         "QRadioButton::indicator"
                         "{"
                         "width : 20px;"
                         "height : 20px;"
                         "}")
        self.phase1_prob = QProgressBar()
        self.phase1_prob.setObjectName("IdleProgress")
        self.phase1_prob.setStyleSheet(DEFAULT_STYLE)
        self.phase1_prob.setValue(10)
        self.phase1_prob.setTextVisible(False)
        e4 = QLabel('Marking')
        e4.setObjectName("Marking")
        e4.setFont(QFont("Arial", 16, QFont.Bold))
        e4.setStyleSheet("color:black;")
        self.phase2_state = QRadioButton()
        self.phase2_state.setChecked(False)
        #e2.setTristate(True)
        self.phase2_state.setObjectName("MarkingCheck")
        self.phase2_state.setStyleSheet("QRadioButton"
                         "{"
                         "color : green;"
                         "}"
                         "QRadioButton::indicator"
                         "{"
                         "width : 20px;"
                         "height : 20px;"
                         "}")
        self.phase2_prob = QProgressBar()
        self.phase2_prob.setObjectName("MarkingProgress")
        self.phase2_prob.setStyleSheet(DEFAULT_STYLE)
        self.phase2_prob.setValue(15)
        self.phase2_prob.setTextVisible(False)
        e7 = QLabel('Injection')
        e7.setObjectName("Injection")
        e7.setFont(QFont("Arial", 16, QFont.Bold))
        e7.setStyleSheet("color:black;")
        self.phase3_state = QRadioButton()
        self.phase3_state.setChecked(True)
        #e2.setTristate(True)
        self.phase3_state.setObjectName("InjectionCheck")
        self.phase3_state.setStyleSheet("QRadioButton"
                         "{"
                         "color : green;"
                         "}"
                         "QRadioButton::indicator"
                         "{"
                         "width : 20px;"
                         "height : 20px;"
                         "}")
        self.phase3_prob = QProgressBar()
        self.phase3_prob.setObjectName("InjectionProgress")
        self.phase3_prob.setStyleSheet(DEFAULT_STYLE)
        self.phase3_prob.setValue(70)
        self.phase3_prob.setTextVisible(False)
        e10 = QLabel('Dissection')
        e10.setObjectName("Dissection")
        e10.setFont(QFont("Arial", 16, QFont.Bold))
        e10.setStyleSheet("color:black;")
        self.phase4_state = QRadioButton()
        self.phase4_state.setChecked(False)
        #e2.setTristate(True)
        self.phase4_state.setObjectName("DissectionCheck")
        self.phase4_state.setStyleSheet("QRadioButton"
                         "{"
                         "color : green;"
                         "}"
                         "QRadioButton::indicator"
                         "{"
                         "width : 20px;"
                         "height : 20px;"
                         "}")
        self.phase4_prob = QProgressBar()
        self.phase4_prob.setObjectName("DissectionProgress")
        self.phase4_prob.setStyleSheet(DEFAULT_STYLE)
        self.phase4_prob.setValue(10)
        self.phase4_prob.setTextVisible(False)
        egrid = QGridLayout()
        egrid.addWidget(e1, 0, 0)
        egrid.addWidget(self.phase1_state, 0, 1)
        egrid.addWidget(self.phase1_prob, 0, 2)
        egrid.addWidget(e4, 1, 0)
        egrid.addWidget(self.phase2_state, 1, 1)
        egrid.addWidget(self.phase2_prob, 1, 2)
        egrid.addWidget(e7, 2, 0)
        egrid.addWidget(self.phase3_state, 2, 1)
        egrid.addWidget(self.phase3_prob, 2, 2)
        egrid.addWidget(e10, 3, 0)
        egrid.addWidget(self.phase4_state, 3, 1)
        egrid.addWidget(self.phase4_prob, 3, 2)
        egrid.setAlignment(Qt.AlignCenter)
        widget.setLayout(egrid)
     
    def setupObjectDetection(self, number_of_colors):
        self.raw_color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
        self.color = []
        for i in self.raw_color:
            new_r, new_g, new_b = hex_to_rgb(i)
            self.color.append(QColor(new_b, new_g, new_r))
            
        num_widget = self.verticalLayout.count()
        while num_widget > 0:
            widget = self.verticalLayout.itemAt(num_widget-1).widget()
            if widget.objectName() == 'ObjectDetection':
                break
            num_widget -= 1
        
        self.vvlayout = QVBoxLayout(widget)
        self.vvlayout.setObjectName("ObjectDetectionVlayout")
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("ObjectDetectionScrollArea")
        self.scrollArea.setStyleSheet("QScrollArea#ObjectDetectionScrollArea{border:0;}")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setObjectName("ObjectDetectionScrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setAttribute(Qt.WA_StyledBackground, True)
        self.scrollAreaWidgetContents.setStyleSheet("QWidget#ObjectDetectionScrollAreaWidgetContents{background-color: #dee0e3;border:0;}")
        self.elayout = QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.vvlayout.addWidget(self.scrollArea)
        e1_group = QGroupBox()
        e1_group.setObjectName("Object1Group")
        e1_group.setStyleSheet("QGroupBox#Object1Group{border:0;}")
        self.e1_button = QRadioButton()
        self.e1_button.setChecked(True)
        self.e1_button.setStyleSheet(
                                "QRadioButton"
                                "{"
                                "color : green;"
                                "}"
                                "QRadioButton::indicator"
                                "{"
                                "width : 20px;"
                                "height : 20px;"
                                "}")
        self.e1_button.setObjectName("Object1Button")
        self.e1_button.toggled.connect(self.e1_button_toggled)
        self.e1_object = QLabel()
        self.e1_object.setObjectName("Object1")
        self.e1_object.setStyleSheet(f"background-color : {self.raw_color[0]};")
        self.e1_object.setFixedWidth(20)
        self.e1_object.setFixedHeight(20)
        e2 = QLabel("Object 1 Name")
        e2.setObjectName("Object1Name")
        e2.setFont(QFont("Arial", 14, QFont.Bold))
        e2.setStyleSheet("color:black;")
        hbox_1 = QHBoxLayout()
        hbox_1.setObjectName("Object1Layout")
        hbox_1.addWidget(self.e1_button)
        hbox_1.addWidget(self.e1_object)
        hbox_1.addWidget(e2)
        hbox_1.setAlignment(Qt.AlignLeft)
        e1_group.setLayout(hbox_1)
        e2_group = QGroupBox()
        e2_group.setObjectName("Object2Group")
        e2_group.setStyleSheet("QGroupBox#Object2Group{border:0;}")
        e3_button = QRadioButton()
        e3_button.setObjectName("Object2Button")
        e3_button.setChecked(False)
        e3_button.setStyleSheet(
                                "QRadioButton"
                                "{"
                                "color : green;"
                                "}"
                                "QRadioButton::indicator"
                                "{"
                                "width : 20px;"
                                "height : 20px;"
                                "}")
        e3 = QLabel()
        e3.setObjectName("Object2")
        e3.setStyleSheet(f"background-color : {self.raw_color[1]};")
        e3.setFixedWidth(20)
        e3.setFixedHeight(20)
        e4 = QLabel("Object 2 Name")
        e4.setObjectName("Object2Name")
        e4.setFont(QFont("Arial", 14, QFont.Bold))
        e4.setStyleSheet("color:black;")
        hbox_2 = QHBoxLayout()
        hbox_2.setObjectName("Object2Layout")
        hbox_2.addWidget(e3_button)
        hbox_2.addWidget(e3)
        hbox_2.addWidget(e4)
        hbox_2.setAlignment(Qt.AlignLeft)
        e2_group.setLayout(hbox_2)
        e3_group = QGroupBox()
        e3_group.setObjectName("Object3Group")
        e3_group.setStyleSheet("QGroupBox#Object3Group{border:0;}")
        e5_button = QRadioButton()
        e5_button.setObjectName("Object3Button")
        e5_button.setChecked(False)
        e5_button.setStyleSheet(
                                "QRadioButton"
                                "{"
                                "color : green;"
                                "}"
                                "QRadioButton::indicator"
                                "{"
                                "width : 20px;"
                                "height : 20px;"
                                "}")
        e5 = QLabel()
        e5.setObjectName("Object3")
        e5.setStyleSheet(f"background-color : {self.raw_color[2]};")
        e5.setFixedWidth(20)
        e5.setFixedHeight(20)
        e6 = QLabel("Object 3 Name")
        e6.setObjectName("Object3Name")
        e6.setFont(QFont("Arial", 14, QFont.Bold))
        e6.setStyleSheet("color:black;")
        hbox_3 = QHBoxLayout()
        hbox_3.setObjectName("Object3Layout")
        hbox_3.addWidget(e5_button)
        hbox_3.addWidget(e5)
        hbox_3.addWidget(e6)
        hbox_3.setAlignment(Qt.AlignLeft)
        e3_group.setLayout(hbox_3)
        e4_group = QGroupBox()
        e4_group.setObjectName("Object4Group")
        e4_group.setStyleSheet("QGroupBox#Object4Group{border:0;}")
        e7_button = QRadioButton()
        e7_button.setObjectName("Object4Button")
        e7_button.setChecked(False)
        e7_button.setStyleSheet(
                                "QRadioButton"
                                "{"
                                "color : green;"
                                "}"
                                "QRadioButton::indicator"
                                "{"
                                "width : 20px;"
                                "height : 20px;"
                                "}")
        e7 = QLabel()
        e7.setObjectName("Object4")
        e7.setStyleSheet(f"background-color : {self.raw_color[3]};")
        e7.setFixedWidth(20)
        e7.setFixedHeight(20)
        e8 = QLabel("Object 4 Name")
        e8.setObjectName("Object4Name")
        e8.setFont(QFont("Arial", 14, QFont.Bold))
        e8.setStyleSheet("color:black;")
        hbox_4 = QHBoxLayout()
        hbox_4.setObjectName("Object4Layout")
        hbox_4.addWidget(e7_button)
        hbox_4.addWidget(e7)
        hbox_4.addWidget(e8)
        hbox_4.setAlignment(Qt.AlignLeft)
        e4_group.setLayout(hbox_4)
        self.elayout.addWidget(e1_group, 0, 0)
        self.elayout.addWidget(e2_group, 1, 0)
        self.elayout.addWidget(e3_group, 2, 0)
        self.elayout.addWidget(e4_group, 3, 0)
        self.elayout.setAlignment(Qt.AlignLeft)
        self.labelSelector.addItems(['Object 1', 'Object 2', 'Object 3', 'Object 4'])
        self.labelSelector.setCurrentIndex(-1)
    
    def e1_button_toggled(self):
        pass
    
    def countTime(self):
        if self.flag:
            current_time = datetime.datetime.now()
            diff_time = (current_time - self.startTime).total_seconds()
            hour = int(diff_time // 3600)
            minute = int(diff_time % 3600 // 60)
            second = int(diff_time % 60)
            self.duraHour.setText('{:02d}'.format(hour))
            self.duraMinute.setText('{:02d}'.format(minute))
            self.duraSecond.setText('{:02d}'.format(second))
    
    def generateReport(self):
        print("66666")
    
    def setupSummary(self):
        num_widget = self.verticalLayout.count()
        while num_widget > 0:
            widget = self.verticalLayout.itemAt(num_widget-1).widget()
            if widget.objectName() == 'SummaryReport':
                break
            num_widget -= 1
        self.summary = widget 
        egrid = QGridLayout()
        group1 = QGroupBox()
        group1.setObjectName("DurationGroup")
        group1.setStyleSheet("QGroupBox#DurationGroup{border:0;}")
        hbox = QHBoxLayout()
        hbox.setObjectName("DurationLayout")
        e1 = QLabel("Duration:")
        e1.setObjectName("Duration")
        e1.setFont(QFont("Arial", 16, QFont.Bold))
        self.flag = False
        self.Timer = QtCore.QTimer()
        self.Timer.timeout.connect(self.countTime)
        self.Timer.start(1)
        self.hour = 0
        self.minute = 0
        self.second = 0
        e2 = QLabel('{:02d}'.format(self.hour))
        e2.setAlignment(Qt.AlignCenter)
        e2.setObjectName("DurationHour")
        e2.setFont(QFont("Arial", 16, QFont.Bold))
        e2.setStyleSheet("background-color: white;")
        self.duraHour = e2
        e3 = QLabel("hrs:")
        e3.setObjectName("DurationHourUnit")
        e3.setFont(QFont("Arial", 16))
        e4 = QLabel('{:02d}'.format(self.minute))
        e4.setAlignment(Qt.AlignCenter)
        e4.setObjectName("DurationMinute")
        e4.setFont(QFont("Arial", 16, QFont.Bold))
        e4.setStyleSheet("background-color: white;")
        self.duraMinute = e4
        e5 = QLabel("min:")
        e5.setObjectName("DurationMinuteUnit")
        e5.setFont(QFont("Arial", 16))
        e6 = QLabel('{:02d}'.format(self.second))
        e6.setAlignment(Qt.AlignCenter)
        e6.setObjectName("DurationSecond")
        e6.setFont(QFont("Arial", 16, QFont.Bold))
        e6.setStyleSheet("background-color: white;")
        self.duraSecond = e6
        e7 = QLabel("sec")
        e7.setObjectName("DurationSecondUnit")
        e7.setFont(QFont("Arial", 16))
        hbox.addWidget(e1)
        hbox.addWidget(e2)
        hbox.addWidget(e3)
        hbox.addWidget(e4)
        hbox.addWidget(e5)
        hbox.addWidget(e6)
        hbox.addWidget(e7)
        hbox.setSpacing(5)
        group1.setLayout(hbox)
        group2 = QGroupBox()
        group2.setObjectName("SurgeonsGroup")
        group2.setStyleSheet("QGroupBox#SurgeonsGroup{border:0;}")
        hbox_1 = QHBoxLayout()
        hbox_1.setObjectName("SurgeonsLayout")
        e8 = QLabel("Surgeons:")
        e8.setObjectName("Surgeons")
        e8.setFont(QFont("Arial", 16, QFont.Bold))
        #e9.setAlignment(Qt.AlignCenter)
        e10 = QLabel()
        hbox_1.addWidget(e8)
        hbox_1.addWidget(self.surgeons)
        hbox_1.addWidget(e10)
        hbox_1.setSpacing(10)
        #hbox_1.setAlignment(Qt.AlignLeft)
        group2.setLayout(hbox_1)
        group2.setAlignment(Qt.AlignLeft)
        group3 = QGroupBox()
        group3.setObjectName("ReportGroup")
        group3.setStyleSheet("QGroupBox#ReportGroup{border:0;}")
        hbox_2 = QHBoxLayout()
        hbox_2.setObjectName("ReportLayout")
        self.reportButton = QPushButton("Generate Report")
        self.reportButton.setObjectName("ReportButton")
        self.reportButton.setFont(QFont("Arial", 16))
        self.reportButton.setStyleSheet("QPushButton"  
                                        "{"
                                        "background-color: green;"
                                        "color: white;"
                                        "padding: 5px 15px;"
                                        "margin-top: 10px;"
                                        "outline: 1px;"
                                        "min-width: 8em;"
                                        "}")
        self.reportButton.clicked.connect(self.generateReport)
        hbox_2.addWidget(self.reportButton)
        hbox_2.setAlignment(Qt.AlignCenter)
        group3.setLayout(hbox_2)
        egrid.addWidget(group1, 0, 0)
        egrid.addWidget(group2, 1, 0)
        egrid.addWidget(group3, 2, 0)
        widget.setLayout(egrid)
    
    def setHLayout(self,name, percent):
        widget = QtWidgets.QLabel(name.title())
        widget.setObjectName(name.title().replace(' ', '')+'Title')
        widget.setFont(QFont('Arial', 18, QFont.Bold))
        widget.setStyleSheet("color: blue;")
        self.horizontalLayout.addWidget(widget, int(percent/5))
        widget1 = QtWidgets.QWidget()
        widget1.setObjectName(name.title().replace(' ', ''))
        widget1.setAttribute(Qt.WA_StyledBackground, True)
        widget1.setStyleSheet(f"QWidget#{name.title().replace(' ', '')}"+"{background-color: #dee0e3; border-radius:5px;}")
        self.horizontalLayout.addWidget(widget1, int(percent*4/5))
    
    def setVLayout(self, name, percent):
        widget = QtWidgets.QLabel(name.title())
        widget.setObjectName(name.title().replace(' ', '')+'Title')
        widget.setFont(QFont('Arial', 18, QFont.Bold))
        widget.setStyleSheet("color: blue;")
        self.verticalLayout.addWidget(widget, 4)
        widget1 = QtWidgets.QWidget()
        widget1.setObjectName(name.title().replace(' ', ''))
        widget1.setAttribute(Qt.WA_StyledBackground, True)
        widget1.setStyleSheet(f"QWidget#{name.title().replace(' ', '')}"+"{background-color: #dee0e3; border-radius:5px;}")
        self.verticalLayout.addWidget(widget1, 21)
    
    def init_status(self):
        self.WORKING = False
        self.INIT = False
        self.TRAINEE = "NONE"
        self.PAUSE_times = 0
        self.INDEPENDENT = True
        self.HELP = False
        self.STATUS = "--"
    
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("iPhaser", "iPhaser"))
    
    def get_frame_size(self):
        capture = cv2.VideoCapture(0)   # TODO: change camera

        # Default resolutions of the frame are obtained (system dependent)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = int(capture.get(cv2.CAP_PROP_FPS))
        fps = 30
        capture.release()
        return frame_width, frame_height, fps
       
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-s", default=False,  action='store_true', help="Whether save predictions")
    parse.add_argument("-q", default=False, action='store_true', help="Display video")
    parse.add_argument("--cfg", default="test_camera", type=str)

    cfg = parse.parse_args()
    cfg = ParserUse(cfg.cfg, "camera").add_args(cfg)
    
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    ui = Ui_iPhaser()
    ui.setupUi(cfg)
    ui.show()
    app.installEventFilter(ui)
    sys.exit(app.exec_())