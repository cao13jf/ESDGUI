import os
import torch
import cv2
import sys
sys.setswitchinterval(0.000001)  # IMPORTANT: Small value for high FPS
import time
import warnings
import argparse
import numpy as np
import pandas as pd



from PyQt5.QtCore import *
import torch
import random
from datetime import datetime


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import ticker

from utils.guis import PhaseCom, draw_segmentation, add_text, CustomTableDelegate
from utils.parser import ParserUse
from utils.gui_parts import *
from utils.report_tools import generate_report
from canvas import Canvas

now = datetime.now()

warnings.filterwarnings("ignore")
from utils.guis import DEFAULT_STYLE, COMBOBOX, CustomTableDelegate
from utils.threads import ImageProcessingThread, VideoReadThread, ReportThread, PlotCurveThread


class Ui_iPhaser(QMainWindow):
    def __init__(self):
        super(Ui_iPhaser, self).__init__()

    def __init_variables(self):
        self.redo = False
        self.point_size = 1
        self.cbFlag = 0
        self.camera_frame = None
        self.save_folder = "../Records"
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
        self.down_ratio = 1  # cfg.down_ratio

        self.init_status()

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
        # self.date_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S.%f")
        self.fps = 0
        self.selected_tool = 0
        self.f_image = None

        self.pred = "--"
        self.preds = []
        self.pred_phases = []
        self.nt_indexes = [0]
        self.transitions = []
        self.pop_image = {}
        self.pop_idx = []
        self.total_diff = []
        self.pop_image_count = []
        self.seg_pred = torch.zeros([4, 224, 224])  # .numpy()
        self.log_data = []
        self.actionList = {}
        self.last_idx = []
        self.count_image = {}
        self.image_count = []
        self.ru = 0
        self.aaa = 0
        self.bbb = 0
        self.prev_second = 0
        self.index2phase = {0: "idle", 1: "marking", 2: "injection", 3: "dissection"}


        self.video = False
        self.disply_width = 1080
        self.display_height = 720
        self.start_x = 0
        self.end_x = -1
        self.start_y = 530
        # self.start_y = 450
        self.end_y = -1
        # cv_img[0:1150, 450:1800]
        self.save_folder = os.path.join("../Records")
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
        self.down_ratio = 1  # cfg.down_ratio
        self.start_time = "--:--:--"
        self.trainee_name = "--"
        self.manual_set = "--"
        self.updating = True

        # model prediction
        # self.centralwidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.setCentralWidget(self.usbVideo)
        # self.setCentralWidget(self.centralwidget)

    def setupUi(self, cfg):
        self.__init_variables()
        self.setObjectName("iPhaser")
        self.resize(1825, 1175)
        self.setMinimumHeight(965)
        self.setMinimumWidth(965)
        self.setStyleSheet("QWidget#iPhaser{background-color: rgb(28, 69, 135)}")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        old_pos = self.frameGeometry().getRect()
        curr_x = old_pos[2]
        curr_y = old_pos[3]

        self.mQImage = QPixmap('./images/test.jpg')
        self.size = QSize(curr_x - 25 - 500, curr_y - 65 - 250)
        self.old_pos = self.frameGeometry().getRect()

        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.resizeEvent = self.windowResized

        self.startButton = QPushButton(self)
        self.startButton.setEnabled(True)
        # self.startButton.setFont(QFont("Arial",12, QFont.Bold))
        self.startButton.setGeometry(QtCore.QRect(500, self.height() - 200, 80, 80))
        self.startButton.setStyleSheet("background-color: DarkGreen;")
        self.startButton.setObjectName('StartButton')
        self.start_pixmap = QtGui.QIcon()
        self.start_pixmap.addFile("./images/start.png", QtCore.QSize(80, 80), QtGui.QIcon.Active, QtGui.QIcon.On)
        self.startButton.setIcon(self.start_pixmap)
        self.startButton.setIconSize(QtCore.QSize(150, 150))
        self.startButton.pressed.connect(self.onButtonClickStart)

        self.stopButton = QPushButton(self)
        self.stopButton.setEnabled(False)
        # self.stopButton.setFont(QFont("Arial",12, QFont.Bold))
        self.stopButton.setGeometry(QtCore.QRect(600, self.height() - 200, 80, 80))
        self.stopButton.setStyleSheet("background-color:DarkGrey;")
        self.stopButton.setObjectName('StopButton')
        self.stop_pixmap = QtGui.QIcon()
        self.stop_pixmap.addFile("./images/stop.png", QtCore.QSize(80, 80), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.stopButton.setIcon(self.stop_pixmap)
        self.stopButton.setIconSize(QtCore.QSize(150, 150))
        self.stopButton.clicked.connect(self.onButtonClickStop)

        self.VideoCanvas = QtWidgets.QLabel(self.centralwidget)
        self.VideoCanvas.setScaledContents(True)
        self.VideoCanvas.setStyleSheet("background-color: black;")
        self.VideoCanvas.setText("")
        self.VideoCanvas.setObjectName("DisplayVideo")
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.VideoCanvas.setSizePolicy(size_policy)

        # =============================
        # Case info layout
        # =============================
        self.trainining_session_components = QtWidgets.QVBoxLayout()
        self.trainining_session_components.setObjectName("VLayout1")

        self.train_title = QtWidgets.QLabel('Training Session')
        self.train_title.setObjectName("TrainLabelTitle")
        self.train_title.setStyleSheet("color:white;")
        self.train_title.setFont(QFont("Arial", 18, QFont.Bold))
        self.trainining_session_components.addWidget(self.train_title, 16)

        self.trainLabel = QtWidgets.QWidget()
        self.trainLabel.setObjectName("TrainLabel")
        self.trainLabel.setAttribute(Qt.WA_StyledBackground, True)
        self.trainLabel.setStyleSheet("background-color: rgb(98, 154, 202); border-radius:5px")
        self.trainining_session_components.addWidget(self.trainLabel, 84)

        self.layoutWidget1 = QtWidgets.QWidget(self)
        self.layoutWidget1.setGeometry(QtCore.QRect(500, 45, 440, 225))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout1 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.verticalLayout1.setObjectName("verticalLayout1")
        self.verticalLayout1.addLayout(self.trainining_session_components)
        

        self.case_info_widget = QtWidgets.QWidget(self)
        self.case_info_widget.setGeometry(QtCore.QRect(30, 45, 440, 270))
        self.case_info_widget.setObjectName("layoutWidget3")
        self.case_info_components = QtWidgets.QVBoxLayout(self.case_info_widget)
        self.case_info_components.setObjectName("verticalLayout3")


        # ==============================
        # Prediction results layout
        # ==============================
        self.phase_pred_widget = QtWidgets.QWidget(self)
        self.phase_pred_widget.setGeometry(QtCore.QRect(30, 275, 440, 400))
        self.phase_pred_widget.setObjectName("layoutWidget4")
        self.phase_pred_info = QtWidgets.QVBoxLayout(self.phase_pred_widget)
        self.phase_pred_info.setObjectName("verticalLayout4")

        # ==============================
        # Online analysis results
        # ==============================
        self.online_analysis_widget = QtWidgets.QWidget(self)
        self.online_analysis_widget.setGeometry(QtCore.QRect(500, 275, 440, 400))
        self.online_analysis_widget.setObjectName("layoutWidget5")
        self.online_analysis_info = QtWidgets.QVBoxLayout(self.online_analysis_widget)
        self.online_analysis_info.setObjectName("verticalLayout5")
        # end of training session

        # start of summary report
        report_tex_widget = QtWidgets.QWidget(self)
        egrid = QGridLayout()
        group1 = QGroupBox()
        group1.setObjectName("DurationGroup")
        group1.setStyleSheet("QGroupBox#DurationGroup{border:0;}")
        hbox = QHBoxLayout()
        hbox.setObjectName("DurationLayout")
        e1 = QLabel("Duration:")
        e1.setObjectName("Duration")
        e1.setFont(QFont("Arial", 16, QFont.Bold))
        e1.setStyleSheet("color: white;")
        self.flag = False
        self.Timer = QtCore.QTimer()
        self.Timer.timeout.connect(self.countTime)
        self.Timer.start(1)
        self.hour = 0
        self.minute = 0
        self.second = 0
        self.curve_array = None
        e2 = QLabel('{:02d}'.format(self.hour))
        e2.setAlignment(Qt.AlignCenter)
        e2.setObjectName("DurationHour")
        e2.setFont(QFont("Arial", 16, QFont.Bold))
        e2.setStyleSheet("background-color: #336699; color: white;")
        self.duraHour = e2
        e3 = QLabel("hrs:")
        e3.setObjectName("DurationHourUnit")
        e3.setFont(QFont("Arial", 16, QFont.Bold))
        e3.setStyleSheet("color: white;")
        e4 = QLabel('{:02d}'.format(self.minute))
        e4.setAlignment(Qt.AlignCenter)
        e4.setObjectName("DurationMinute")
        e4.setFont(QFont("Arial", 16, QFont.Bold))
        e4.setStyleSheet("background-color: #336699;  color: white;")
        self.duraMinute = e4
        e5 = QLabel("min:")
        e5.setObjectName("DurationMinuteUnit")
        e5.setFont(QFont("Arial", 16, QFont.Bold))
        e5.setStyleSheet("color: white;")
        e6 = QLabel('{:02d}'.format(self.second))
        e6.setAlignment(Qt.AlignCenter)
        e6.setObjectName("DurationSecond")
        e6.setFont(QFont("Arial", 16, QFont.Bold))
        e6.setStyleSheet("background-color: #336699;  color: white;")
        self.duraSecond = e6
        e7 = QLabel("sec")
        e7.setObjectName("DurationSecondUnit")
        e7.setFont(QFont("Arial", 16, QFont.Bold))
        e7.setStyleSheet("color: white;")
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
        e8.setStyleSheet("color: white;  color: white；")
        # e9.setAlignment(Qt.AlignCenter)

        # Surgeon info
        self.surgeons = QComboBox()
        self.surgeons.setObjectName("SurgeonsName")
        self.surgeons.setStyleSheet(COMBOBOX)
        self.surgeons.setEditable(True)
        self.surgeons.lineEdit().setAlignment(Qt.AlignCenter)
        self.surgeons.lineEdit().setFont(QFont("Arial", 16, QFont.Bold))
        self.surgeons.lineEdit().setReadOnly(True)
        self.surgeons.setCurrentIndex(-1)

        e10 = QLabel()
        hbox_1.addWidget(e8)
        hbox_1.addWidget(self.surgeons)
        hbox_1.addWidget(e10)
        hbox_1.setSpacing(10)
        # hbox_1.setAlignment(Qt.AlignLeft)
        group2.setLayout(hbox_1)
        group2.setAlignment(Qt.AlignLeft)
        group3 = QGroupBox()
        group3.setObjectName("ReportGroup")
        group3.setStyleSheet("QGroupBox#ReportGroup{border:0;}")
        hbox_2 = QHBoxLayout()
        hbox_2.setObjectName("ReportLayout")
        self.reportButton = QPushButton("Generate AI Report")
        self.reportButton.setObjectName("ReportButton")
        self.reportButton.setFont(QFont("Arial", 18, QFont.Bold))
        self.reportButton.setStyleSheet("QPushButton"
                                        "{"
                                        "background-color: green;"
                                        "color: white;"
                                        "padding: 5px 15px;"
                                        "margin-top: 10px;"
                                        "outline: 1px;"
                                        "min-width: 8em;"
                                        "}")
        self.reportButton.setFixedSize(280, 70)
        self.reportButton.clicked.connect(self.stop_thread)
        self.reportButton.clicked.connect(self.generateReport)
        hbox_2.addWidget(self.reportButton)
        hbox_2.setAlignment(Qt.AlignCenter)
        group3.setLayout(hbox_2)
        egrid.addWidget(group1, 0, 0)
        egrid.addWidget(group2, 1, 0)
        upperLeftWidget = QtWidgets.QWidget(self)
        upperLeftWidget.setLayout(egrid)
        upperHlayout = QtWidgets.QHBoxLayout()
        upperHlayout.addWidget(upperLeftWidget)
        # self.canvas_bar = FigureCanvas(Figure(figsize=(400/80, 50/80), dpi=80))  # Update canvas
        # self.canvas_table = FigureCanvas(Figure(figsize=(400/80, 200/80), dpi=80))

        # self.summaryReportOutput3 = QtWidgets.QWidget(self)
        # self.summaryReportOutput3.setStyleSheet("background-color: white;")
        # self.summaryReportOutput3.setFixedSize(400, 50)
        upperHlayout.addWidget(self.reportButton)
        # self.ax_bar = self.canvas_bar.figure.subplots()
        # self.ax_bar.set_axis_off()
        # self.phase_colors = {"idle": "blue", "marking": "green", "injection": "yellow", "dissection": "red"}

        report_tex_widget.setLayout(upperHlayout)

        self.report_components = QtWidgets.QVBoxLayout()
        self.report_components.setObjectName("VLayout2")

        self.summaryReportTitle = QtWidgets.QLabel('Summary Report')
        self.summaryReportTitle.setObjectName("SummaryReportTitle")
        self.summaryReportTitle.setStyleSheet("color:white;")
        self.summaryReportTitle.setFont(QFont("Arial", 18, QFont.Bold))
        self.report_components.addWidget(self.summaryReportTitle, 16)

        self.summaryReportContent = QtWidgets.QWidget()
        self.summaryReportContent.setObjectName("summaryReportContent")
        self.summaryReportContent.setAttribute(Qt.WA_StyledBackground, True)
        self.summaryReportContent.setStyleSheet("background-color: rgb(98, 154, 202); border-radius:5px")
        self.ContentVerticalLayout = QtWidgets.QVBoxLayout()
        self.summaryReportContent.setLayout(self.ContentVerticalLayout)
        self.report_components.addWidget(self.summaryReportContent, 84)

        ContentLowerWidget = QtWidgets.QWidget(self)
        LowerVLayout = QtWidgets.QVBoxLayout()
        ContentLowerWidget.setLayout(LowerVLayout)

        self.LowerUpperWidget = QtWidgets.QWidget(self)
        LowerUpperLayout = QtWidgets.QHBoxLayout()
        self.LowerUpperWidget.setLayout(LowerUpperLayout)
        LowerVLayout.addWidget(self.LowerUpperWidget)

        self.LowerLowerWidget = QtWidgets.QWidget(self)
        LowerLowerLayout = QtWidgets.QHBoxLayout()
        self.LowerLowerWidget.setLayout(LowerLowerLayout)

        self.canvas_nt = QLabel(self.centralwidget)
        self.canvas_nt.setScaledContents(True)
        self.canvas_nt.setStyleSheet("background-color: lightgray;")
        self.canvas_nt.setFixedSize(440, 350)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.canvas_nt.setSizePolicy(size_policy)
        LowerLowerLayout.addWidget(self.canvas_nt)
        self.table = QTableWidget(self)
        self.table.setStyleSheet("""
        QTableWidget::item:selected {
            background-color: transparent;
        }
        """)
        self.table.setItemDelegate(CustomTableDelegate())
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setStyleSheet("QTableView { background-color: lightgrey; border: none}")
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Phase Name", "Duration", "Proportion"])
        header = self.table.horizontalHeader()
        header.setStyleSheet("background-color: darkgrey; color: black")
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setFixedHeight(50)
        self.table.setFixedSize(390, 330)
        self.table.setRowCount(5)
        self.table.setItem(0, 0, QTableWidgetItem("Marking"))
        self.table.setItem(1, 0, QTableWidgetItem("Injection"))
        self.table.setItem(2, 0, QTableWidgetItem("Dissection"))
        self.table.setItem(3, 0, QTableWidgetItem("Idle"))
        self.table.setItem(4, 0, QTableWidgetItem("Total"))
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            item.setFont(font)
            item.setTextAlignment(Qt.AlignCenter)
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(12)
        for column in range(self.table.columnCount()):
            header_item = self.table.horizontalHeaderItem(column)
            header_item.setFont(header_font)
            header_item.setTextAlignment(Qt.AlignCenter)
        LowerLowerLayout.addWidget(self.table)
        LowerVLayout.addWidget(self.LowerLowerWidget)
        LowerVLayout.setAlignment(Qt.AlignCenter)

        # LowerRightVLayout.addWidget(self.fullReportButton)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: white;")

        self.ContentVerticalLayout.addWidget(report_tex_widget)
        self.ContentVerticalLayout.addWidget(line)
        self.ContentVerticalLayout.addWidget(ContentLowerWidget)

        self.layoutWidget2 = QtWidgets.QWidget(self)
        self.layoutWidget2.setGeometry(QtCore.QRect(30, 620, 910, 600))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.verticalLayout2 = QtWidgets.QVBoxLayout(self.layoutWidget2)
        self.verticalLayout2.setObjectName("verticalLayout2")
        self.verticalLayout2.addLayout(self.report_components)
        # end of summary report


        # start of Case information
        name = "case information"
        case_information_widget = QtWidgets.QLabel(name.title())
        case_information_widget.setObjectName(name.title().replace(' ', '') + 'Title')
        case_information_widget.setFont(QFont('Arial', 18, QFont.Bold))
        case_information_widget.setStyleSheet("color: white;")
        self.case_info_components.addWidget(case_information_widget, 4)
        widget1 = QtWidgets.QWidget()
        widget1.setObjectName(name.title().replace(' ', ''))
        widget1.setAttribute(Qt.WA_StyledBackground, True)
        widget1.setStyleSheet(
            f"QWidget#{name.title().replace(' ', '')}" + "{background-color: rgb(98, 154, 202); border-radius:5px;}")
        self.case_info_components.addWidget(widget1, 21)
        gapWidget = QtWidgets.QWidget()
        gapWidget.setFixedWidth(50)  # Set the desired width for the gap
        gapWidget.setObjectName(name.title().replace(' ', '') + 'Gap')
        self.case_info_components.addWidget(gapWidget, 5)

        vlayout = QVBoxLayout(widget1)
        vlayout.setObjectName("CaseInformationVlayout")
        current_datetime = datetime.now()
        year = current_datetime.year
        month = current_datetime.month
        month_names = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec"
        }
        month_name = month_names[month]
        day = current_datetime.day
        self.e1 = QLabel('Patient ID:')
        self.e1.setObjectName("PID")
        self.e1.setFont(QFont("Arial", 16, QFont.Bold))
        self.e1.setStyleSheet("color:white;")
        vlayout.addWidget(self.e1)
        hlayout = QHBoxLayout()
        e2 = QLineEdit()
        e2.setFixedHeight(35)
        e2.setFixedWidth(180)
        e2.setObjectName("PID1")
        e2.setStyleSheet("background: #336699;border-radius:5px;color: white")
        e2.setAlignment(Qt.AlignCenter)
        e2.setFont(QFont("Arial", 14))
        e2.setText("Jenny")
        e2.setEnabled(False)
        hlayout.addWidget(e2)
        e3 = QFrame()
        e3.setFrameShape(QFrame.HLine)
        e3.setStyleSheet("color: white;")
        e3.setFrameShadow(QFrame.Plain)
        e3.setLineWidth(2)
        e3.setObjectName("PIDSpace")
        hlayout.addWidget(e3)
        e4 = QLineEdit()
        e4.setEnabled(False)
        e4.setFixedHeight(35)
        e4.setFixedWidth(180)
        e4.setObjectName("PID2")
        e4.setStyleSheet("background: #336699;border-radius:5px;color: white")
        e4.setAlignment(Qt.AlignCenter)
        e4.setFont(QFont("Arial", 14))
        e4.setText("798xxx(x)")
        hlayout.addWidget(e4)
        hlayout.setSpacing(15)
        vlayout.addLayout(hlayout)
        e5 = QLabel('Date:')
        e5.setObjectName("PIDDate")
        e5.setFont(QFont("Arial", 16, QFont.Bold))
        e5.setStyleSheet("color:white;")
        vlayout.addWidget(e5)
        hlayout1 = QHBoxLayout()
        e6 = QLineEdit()
        e6.setEnabled(False)
        e6.setFixedHeight(35)
        e6.setFixedWidth(105)
        e6.setObjectName("PIDDateYear")
        e6.setStyleSheet("background-color: #336699;border-radius:5px;color: white")
        e6.setAlignment(Qt.AlignCenter)
        e6.setFont(QFont("Arial", 14))
        e6.setText(str(year))
        hlayout1.addWidget(e6)
        e7 = QFrame()
        e7.setFrameShape(QFrame.HLine)
        e7.setStyleSheet("color: white;")
        e7.setFrameShadow(QFrame.Plain)
        e7.setLineWidth(2)
        e7.setObjectName("PIDDateSpace")
        hlayout1.addWidget(e7)
        e8 = QLineEdit()
        e8.setEnabled(False)
        e8.setFixedHeight(35)
        e8.setFixedWidth(105)
        e8.setObjectName("PIDDateMonth")
        e8.setStyleSheet("background-color: #336699;border-radius:5px;color: white")
        e8.setAlignment(Qt.AlignCenter)
        e8.setFont(QFont("Arial", 14))
        e8.setText(month_name)
        hlayout1.addWidget(e8)
        e9 = QFrame()
        e9.setFrameShape(QFrame.HLine)
        e9.setStyleSheet("color: white;")
        e9.setFrameShadow(QFrame.Plain)
        e9.setLineWidth(2)
        e9.setObjectName("PIDDateSpace1")
        hlayout1.addWidget(e9)
        e10 = QLineEdit()
        e10.setEnabled(False)
        e10.setFixedHeight(35)
        e10.setFixedWidth(105)
        e10.setObjectName("PIDDateDay")
        e10.setStyleSheet("background-color: #336699;border-radius:5px;color: white")
        e10.setAlignment(Qt.AlignCenter)
        e10.setFont(QFont("Arial", 14))
        e10.setText(str(day))
        hlayout1.addWidget(e10)
        hlayout1.setSpacing(15)
        vlayout.addLayout(hlayout1)
        # end of case information

        # start of phase recognition
        name = "phase recognition"
        phase_recognition_widget = QtWidgets.QLabel(name.title())
        phase_recognition_widget.setObjectName(name.title().replace(' ', '') + 'Title')
        phase_recognition_widget.setFont(QFont('Arial', 18, QFont.Bold))
        phase_recognition_widget.setStyleSheet("color: white;")
        self.phase_pred_info.addWidget(phase_recognition_widget, 4)
        widget2 = QtWidgets.QWidget()
        widget2.setObjectName(name.title().replace(' ', ''))
        widget2.setAttribute(Qt.WA_StyledBackground, True)
        widget2.setStyleSheet(
            f"QWidget#{name.title().replace(' ', '')}" + "{background-color: rgb(98, 154, 202); border-radius:5px;}")
        self.phase_pred_info.addWidget(widget2, 21)
        gapWidget = QtWidgets.QWidget()
        gapWidget.setFixedWidth(50)  # Set the desired width for the gap
        gapWidget.setObjectName(name.title().replace(' ', '') + 'Gap')
        self.phase_pred_info.addWidget(gapWidget, 5)

        widget3 = QtWidgets.QWidget(self)
        e1 = QLabel('Idle')
        e1.setObjectName("Idle")
        e1.setFont(QFont("Arial", 16, QFont.Bold))
        e1.setStyleSheet("color:white;")
        self.phase1_state = QRadioButton()
        self.phase1_state.setChecked(False)
        # e2.setTristate(True)
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
        self.phase1_prob.setValue(7)
        self.phase1_prob.setTextVisible(False)
        e4 = QLabel('Marking')
        e4.setObjectName("Marking")
        e4.setFont(QFont("Arial", 16, QFont.Bold))
        e4.setStyleSheet("color:white;")
        self.phase2_state = QRadioButton()
        self.phase2_state.setChecked(False)
        # e2.setTristate(True)
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
        e7.setStyleSheet("color:white;")
        self.phase3_state = QRadioButton()
        self.phase3_state.setChecked(True)
        # e2.setTristate(True)
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
        e10.setStyleSheet("color:white;")
        self.phase4_state = QRadioButton()
        self.phase4_state.setChecked(False)
        # e2.setTristate(True)
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
        widget3.setLayout(egrid)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: white;")

        self.a1 = QLabel("Predicted phase")
        self.a1.setFont(QFont("Arial", 16, QFont.Bold))
        self.a1.setStyleSheet("color: white")
        self.a2 = QLineEdit()
        self.a2.setAlignment(Qt.AlignCenter)
        self.a2.setEnabled(False)
        self.a2.setFont(QFont("Arial", 26, QFont.Bold))
        self.a2.setStyleSheet("color: #ff9900; background-color: #336699;")
        self.VLayout3 = QtWidgets.QVBoxLayout()
        self.VLayout3.addWidget(widget3)
        self.VLayout3.addWidget(line)
        self.VLayout3.addWidget(self.a1, alignment=Qt.AlignCenter)
        self.VLayout3.addWidget(self.a2, alignment=Qt.AlignCenter)
        widget2.setLayout(self.VLayout3)
        # end of phase recognition

        # start of online analytics
        name = "online analytics"
        online_analytics_widget = QtWidgets.QLabel(name.title())
        online_analytics_widget.setObjectName(name.title().replace(' ', '') + 'Title')
        online_analytics_widget.setFont(QFont('Arial', 18, QFont.Bold))
        online_analytics_widget.setStyleSheet("color: white;")
        self.online_analysis_info.addWidget(online_analytics_widget, 4)
        widget4 = QtWidgets.QWidget()
        widget4.setObjectName(name.title().replace(' ', ''))
        widget4.setAttribute(Qt.WA_StyledBackground, True)
        widget4.setStyleSheet(
            f"QWidget#{name.title().replace(' ', '')}" + "{background-color: rgb(98, 154, 202); border-radius:5px;}")
        self.online_analysis_info.addWidget(widget4, 21)
        gapWidget = QtWidgets.QWidget()
        gapWidget.setFixedWidth(50)  # Set the desired width for the gap
        gapWidget.setObjectName(name.title().replace(' ', '') + 'Gap')
        self.online_analysis_info.addWidget(gapWidget, 5)

        # Create the gray rectangles
        self.rect1 = QLineEdit()
        self.rect1.setEnabled(False)
        self.rect1.setStyleSheet("background-color: #336699; color: white;")
        self.rect1.setFixedWidth(300)
        self.rect1.setFixedHeight(30)
        self.rect2 = QLineEdit()
        self.rect2.setEnabled(False)
        self.rect2.setStyleSheet("background-color: #336699; color: white;")
        self.rect2.setFixedWidth(300)
        self.rect2.setFixedHeight(30)
        self.rect3 = QLineEdit()
        self.rect3.setEnabled(False)
        self.rect3.setStyleSheet("background-color: #336699; color: white;")
        self.rect3.setFixedWidth(300)
        self.rect3.setFixedHeight(30)
        self.rect4 = QLineEdit()
        self.rect4.setEnabled(False)
        self.rect4.setStyleSheet("background-color: #336699; color: white;")
        self.rect4.setFixedWidth(300)
        self.rect4.setFixedHeight(30)

        self.rect1.setAlignment(Qt.AlignCenter)
        self.rect1.setFont(QFont("Arial", 16, QFont.Bold))
        self.rect2.setAlignment(Qt.AlignCenter)
        self.rect2.setFont(QFont("Arial", 16, QFont.Bold))
        self.rect3.setAlignment(Qt.AlignCenter)
        self.rect3.setFont(QFont("Arial", 16, QFont.Bold))
        self.rect4.setAlignment(Qt.AlignCenter)
        self.rect4.setFont(QFont("Arial", 16, QFont.Bold))

        # Create the labels
        e1 = QLabel('Time:')
        e1.setObjectName("Time")
        e1.setFont(QFont("Arial", 16, QFont.Bold))
        e1.setStyleSheet("color:white;")
        e3 = QLabel('Mentor:')
        e3.setObjectName("Mentor")
        e3.setFont(QFont("Arial", 16, QFont.Bold))
        e3.setStyleSheet("color:white;")
        e5 = QLabel('Trainee:')
        e5.setObjectName("Trainee")
        e5.setFont(QFont("Arial", 16, QFont.Bold))
        e5.setStyleSheet("color:white;")
        e7 = QLabel('NT-index:')
        e7.setObjectName("NT-index")
        e7.setFont(QFont("Arial", 16, QFont.Bold))
        e7.setStyleSheet("color:white;")

        # Create the layout for each row
        row1_layout = QHBoxLayout()
        row1_layout.addWidget(e1)
        row1_layout.addWidget(self.rect1)
        row2_layout = QHBoxLayout()
        row2_layout.addWidget(e3)
        row2_layout.addWidget(self.rect2)
        row3_layout = QHBoxLayout()
        row3_layout.addWidget(e5)
        row3_layout.addWidget(self.rect3)
        row4_layout = QHBoxLayout()
        row4_layout.addWidget(e7)
        row4_layout.addWidget(self.rect4)

        # Create the main vertical layout
        VLayout = QVBoxLayout()
        VLayout.addLayout(row1_layout)
        VLayout.addLayout(row2_layout)
        VLayout.addLayout(row3_layout)
        VLayout.addLayout(row4_layout)

        # Set the layout for the widget
        widget4.setLayout(VLayout)

        # end of online analytics

        self.imageLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageLabel.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft)  # Align to the left bottom corner
        self.imageLabel.setObjectName("CUHK_logol")
        self.imageLabel.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        # Load the image
        image_path = "images/CUHK_logol.png"  # Replace with the actual path to your image
        image = QtGui.QPixmap(image_path)
        # Resize the image to fit within the available space while maintaining the aspect ratio
        scaled_image = image.scaled(125, 100)
        # Set the scaled image as the pixmap for the image label
        self.imageLabel.setPixmap(scaled_image)
        # Adjust the position and size of the image label when the main window is resized
        self.centralwidget.resizeEvent = self.windowResized

        self.AiEndoLabel = QtWidgets.QLabel(self.centralwidget)
        self.AiEndoLabel.setFixedSize(300, 100)
        self.AiEndoLabel.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft)  # Align to the left bottom corner
        self.AiEndoLabel.setObjectName("AiEndoLabel")
        # Set the size policy of the image label to Ignored
        # self.AiEndoLabel.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        # Load the image
        AiEndo_image_path = "images/AI-Endo.jpg"  # Replace with the actual path to your image
        Ai_Endo_image = QtGui.QPixmap(AiEndo_image_path)
        # Resize the image to fit within the available space while maintaining the aspect ratio
        scaled_Ai_Endo_Image = Ai_Endo_image.scaled(300, 100)
        # Set the scaled image as the pixmap for the image label
        self.AiEndoLabel.setPixmap(scaled_Ai_Endo_Image)
        # Adjust the position and size of the image label when the main window is resized
        self.centralwidget.resizeEvent = self.windowResized

        self.setupTrainer()
        # self.setupPhaseRecog()
        # self.setupAnalytics()
        self.setCentralWidget(self.centralwidget)
        self.retranslateUi()

        self.process_frames = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(30)

        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # self.camera_thread = VideoReadThread()
        # self.camera_thread.frame_data.connect(self.update_camera_frame)
        # self.camera_thread.moveToThread(QCoreApplication.instance().thread())
        # self.camera_thread.start()

        self.canvas_draw_thread = None
        self.frame_index = 0
        self.processing_thread = ImageProcessingThread(start_x=self.start_x,
                                                       end_x=self.end_x,
                                                       start_y=self.start_y,
                                                       end_y=self.end_y,
                                                       cfg=cfg)
        self.processing_thread.processed_frame.connect(self.update_pred)
        self.processing_thread.moveToThread(QCoreApplication.instance().thread())
        self.processing_thread.start()

        self.plot_thread = PlotCurveThread()
        self.plot_thread.ploted_curve_array.connect(self.update_plot)
        self.plot_thread.moveToThread(QCoreApplication.instance().thread())
        self.plot_thread.start()

        self.report_thread = ReportThread("../Records")
        self.report_thread.out_file_path.connect(self.enableReport)
        self.report_thread.moveToThread(QCoreApplication.instance().thread())
        # self.report_thread.start()

    def windowResized(self, event):
        # Get the current window size
        width = self.centralwidget.width()
        height = self.centralwidget.height()

        # Calculate the new size and position of the video widget
        videoWidth = min(width - 970 - 25, (height - 50)/13*15)
        videoHeight = min(height - 50, (width - 970 - 25)/15*13)

        # videoWidth = width - 970 - 25
        # videoHeight = height - 25



        # Set the new geometry of the video widget
        self.VideoCanvas.setGeometry(QtCore.QRect(970, 25, videoWidth, videoHeight))

        # Calculate the new position of the image label
        imageWidth = self.imageLabel.pixmap().width()
        imageHeight = self.imageLabel.pixmap().height()
        imageX = 50  # Align to the left side of the window
        imageY = height - imageHeight - 20  # Align to the bottom of the window

        # Set the new geometry of the image label
        self.imageLabel.setGeometry(QtCore.QRect(imageX, imageY, imageWidth, imageHeight))
        self.AiEndoLabel.setGeometry(QtCore.QRect(imageX+175, imageY, imageWidth, imageHeight))

        self.startButton.setGeometry(QtCore.QRect(650, self.height() - 110, 80, 80))
        self.stopButton.setGeometry(QtCore.QRect(800, self.height() - 110, 80, 80))

    def update_image(self):
        """Convert from an opencv image to QPixmap"""
        # Collect settings of functional keys
        # cv_img = cv_img[30:1050, 695:1850]
        ret, frame = self.camera.read()
        # print(self.frame_index)
        if frame is not None and self.updating:
            frame = frame[self.start_x:self.end_x, self.start_y:self.end_y]
            if self.WORKING:
                self.processing_thread.add_frame(frame)
                self.update_table()

                self.plot_thread.add_data([np.array(self.nt_indexes), int(time.time()) - self.start_second])
                self.display_frame(frame)
                if self.curve_array is not None:
                    self.display_curve(self.curve_array)

        # update the online analytics box
                self.current_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
                self.rect1.setText(self.current_time)
                self.rect2.setText(self.mentor.text())
                self.rect3.setText(self.trainee.text())
                self.rect4.setText("{:.3f}".format(self.nt_indexes[-1]))

            else:
                self.display_frame(frame)

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.manual_frame = self.manual_frame - 1
        if self.manual_frame <= 0:
            self.manual_frame = 0
            self.manual_set = "--"
        if self.INIT:
            self.date_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
            if self.manual_frame > 0:
                self.pred = self.manual_set
            rgb_image = add_text(self.date_time, self.pred, self.trainee.text(), self.nt_indexes[-1], rgb_image)
        if self.WORKING:
            # print('write', rgb_image.shape)
            self.date_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
            rbg_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            rgb_image = add_text(self.date_time, self.pred, self.trainee.text(), self.nt_indexes[-1], rgb_image)
            # self.output_video.write(rbg_image)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.size.width(), self.size.height(), Qt.KeepAspectRatio)
        p = QPixmap.fromImage(p)
        self.VideoCanvas.setPixmap(p)

    def display_curve(self, curve_frame):
        cirve_frame = cv2.cvtColor(curve_frame, cv2.COLOR_RGB2BGR)
        h, w, ch = cirve_frame.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(cirve_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        # p = convert_to_Qt_format.scaled(self.size.width(), self.size.height(), Qt.KeepAspectRatio)
        p = QPixmap.fromImage(convert_to_Qt_format)
        p = p.scaled(3200, 2400)
        self.canvas_nt.setPixmap(p)
        self.canvas_nt.setContentsMargins(30, 5, 5, 30)

    def update_camera_frame(self, camera_frame, frame_index):
        self.camera_frame = camera_frame
        self.frame_index = frame_index

    def generate_random_numbers(self, i):
        max_index = i - 1
        random_numbers = np.random.random(3) * 0.2
        remaining_sum = 1 - random_numbers.sum()
        random_numbers = np.insert(random_numbers, max_index, remaining_sum)
        # random_numbers[max_index] = random.uniform(0.5, 1.0)

        return random_numbers
    def update_pred(self, pred):

        self.manual_frame = self.manual_frame - 1
        pred_index = np.argmax(pred)
        prob = np.exp(pred) / sum(np.exp(pred))
        self.pred = self.index2phase[pred_index]
        self.a2.setText(self.pred)
        add_log = [datetime.now(), self.trainee.text(), self.mentor.text(), self.bed.text(), self.pred]
        add_log += prob.tolist()
        self.log_data.append(add_log)
        pred_percentages = ((np.exp(pred) / np.exp(pred).sum()) * 100).tolist()
        self.phase1_prob.setValue(pred_percentages[0])
        self.phase2_prob.setValue(pred_percentages[1])
        self.phase3_prob.setValue(pred_percentages[2])
        self.phase4_prob.setValue(pred_percentages[3])
        states = [False] * 4
        states[pred_index] = True
        self.phase1_state.setChecked(states[0])
        self.phase2_state.setChecked(states[1])
        self.phase3_state.setChecked(states[2])
        self.phase4_state.setChecked(states[3])

        if self.prev_second == 0:
            self.prev_second = int(time.time())
        self.cur_sceond = int(time.time())
        if len(self.log_data) > 1:
            multi_el = 1 # self.cur_sceond - self.prev_second
            self.prev_second = self.cur_sceond
            prev_pred = self.log_data[-2][-5]
            cur_pred = self.log_data[-1][-5]
            self.pred_phases += [cur_pred] * multi_el
            self.preds.append(self.pred)
            if prev_pred != cur_pred:
                self.transitions += [self.transitions[-1] + 1] * multi_el
            else:
                self.transitions += [self.transitions[-1]] * multi_el
        else:
            self.transitions = [0]
        print("*"*10, self.transitions[-1])
        self.nt_indexes.append(self.transitions[-1] / 2.0 / len(self.transitions) * 10)


    def update_plot(self, plotted_cuvre_array):

        self.curve_array = plotted_cuvre_array


    def update_table(self):
        if self.WORKING and len(self.pred_phases) > 0:
            total_seconds = int(time.time()) - self.start_second
            phase_ratio = self.pred_phases.count("marking") / len(self.pred_phases)
            num_ratio = QTableWidgetItem("{:>.2%}".format(phase_ratio))
            num = QTableWidgetItem("{} s".format(int(phase_ratio * total_seconds)))
            num.setFont(QFont("", 16))
            num_ratio.setFont(QFont("", 16))
            num.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            num_ratio.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.table.setItem(0, 1, num)
            self.table.setItem(0, 2, num_ratio)

            phase_ratio = self.pred_phases.count("injection") / len(self.pred_phases)
            num_ratio = QTableWidgetItem("{:>.2%}".format(phase_ratio))
            num = QTableWidgetItem("{} s".format(int(phase_ratio * total_seconds)))
            num.setFont(QFont("", 16))
            num_ratio.setFont(QFont("", 16))
            num.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            num_ratio.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.table.setItem(1, 1, num)
            self.table.setItem(1, 2, num_ratio)

            phase_ratio = self.pred_phases.count("dissection") / len(self.pred_phases)
            num_ratio = QTableWidgetItem("{:>.2%}".format(phase_ratio))
            num = QTableWidgetItem("{} s".format(int(phase_ratio * total_seconds)))
            num.setFont(QFont("", 16))
            num_ratio.setFont(QFont("", 16))
            num.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            num_ratio.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.table.setItem(2, 1, num)
            self.table.setItem(2, 2, num_ratio)

            phase_ratio = self.pred_phases.count("idle") / len(self.pred_phases)
            num_ratio = QTableWidgetItem("{:>.2%}".format(phase_ratio))
            num = QTableWidgetItem("{} s".format(int(phase_ratio * total_seconds)))
            num.setFont(QFont("", 16))
            num_ratio.setFont(QFont("", 16))
            num.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            num_ratio.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.table.setItem(3, 1, num)
            self.table.setItem(3, 2, num_ratio)

            num = QTableWidgetItem("{} s".format(int(total_seconds)))
            num_ratio = QTableWidgetItem("   /   ")
            num.setFont(QFont("", 16))
            num_ratio.setFont(QFont("", 16))
            num.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            num_ratio.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.table.setItem(4, 1, num)
            self.table.setItem(4, 2, num_ratio)

    def onButtonClickStart(self):
        self.startButton.setStyleSheet("background-color: DarkGrey;")
        self.startButton.setEnabled(False)
        self.stopButton.setStyleSheet("background-color: DarkRed;")
        self.stopButton.setEnabled(True)
        self.flag = True
        self.WORKING = True
        self.video = False
        self.log_data = []
        self.pred_phases = []
        # video_file_name = os.path.join(self.save_folder, self.e1.text().replace(":", "_").replace(" ",
        #                                                                                           "-") + "_" + self.start_time.replace(
        #     ":", "-") + ".avi")
        # self.output_video = cv2.VideoWriter(video_file_name, self.CODEC, self.stream_fps,
        #                                     (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        self.start_time = datetime.now().strftime("%H:%M:%S")
        self.start_second = int(time.time())
        self.log_file = os.path.join(self.save_folder, self.trainee.text() + "_" + self.start_time.replace(":",
                                                                                                           "-") + ".csv")
        self.startTime = datetime.now()
        lineEdits = self.trainLabel.findChildren(QLineEdit)
        for lineEdit in lineEdits:
            if lineEdit.objectName() == 'TraineeInput':
                if self.surgeons.findText(lineEdit.text()) == -1 and lineEdit.text() != '':
                    self.surgeons.addItem(lineEdit.text())
                    self.surgeons.setCurrentIndex(-1)

    def onButtonClickStop(self):
        self.startButton.setStyleSheet("background-color: DarkGreen;")
        self.startButton.setEnabled(True)
        self.stopButton.setStyleSheet("background-color: DarkGrey;")
        self.stopButton.setEnabled(False)
        self.flag = False
        self.WORKING = False
        # self.video = False
        # self.output_video.release()
        # self.thread.terminate()
        # self.thread.wait(1)

        self.pred = "--"
        self.NT_index = 0
        self.init_status()
        self.save_log_data()
        # self.DisplayTrainee.setText("--")

    def init_status(self):
        self.WORKING = False
        self.INIT = False
        self.TRAINEE = "NONE"
        self.PAUSE_times = 0
        self.INDEPENDENT = True
        self.HELP = False
        self.STATUS = "--"
        self.nt_indexes = [0]
        self.transitions = [0]
        self.pred_phases = []

    def resizeEvent(self, event):
        old_pos = self.frameGeometry().getRect()
        curr_x = old_pos[2]
        curr_y = old_pos[3]
        self.size = QSize(curr_x - 25 - 500, curr_y - 65 - 250)
        self.pos = QSize(500, 250)
        # self.usbVideo.setGeometry(QtCore.QRect(500, 250, curr_x-25-500, curr_y-65-250))
        # self.canvas.label_mask.scaled(curr_x-25-500, curr_y-65-250, QtCore.Qt.KeepAspectRatio)

    def setupTrainer(self):
        e1 = QLabel('Mentor:')
        e1.setObjectName("Mentor")
        e1.setFont(QFont("Arial", 16, QFont.Bold))
        e1.setStyleSheet("color:white;")
        e2 = QLabel('Lesion Location:')
        e2.setObjectName("LesionLocation")
        e2.setFont(QFont("Arial", 16, QFont.Bold))
        e2.setStyleSheet("color:white;")
        e3 = QLineEdit()
        e3.setEnabled(False)
        e3.setFixedHeight(35)
        e3.setObjectName("MentorInput")
        e3.setStyleSheet("background-color: #336699;color: white")
        e3.setFont(QFont("Arial", 14))
        e3.setText("Jeffery")
        e3.setAlignment(Qt.AlignCenter)
        self.mentor = e3
        e4 = QLineEdit()
        e4.setEnabled(False)
        e4.setFixedHeight(35)
        e4.setObjectName("LesionLocationInput")
        e4.setStyleSheet("background-color: #336699;color: white")
        e4.setFont(QFont("Arial", 14))
        e4.setText("Stomach")
        e4.setAlignment(Qt.AlignCenter)
        self.lesion = e4
        e5 = QLabel("Trainee:")
        e5.setObjectName("Trainee")
        e5.setFont(QFont("Arial", 16, QFont.Bold))
        e5.setStyleSheet("color:white;")
        e6 = QLabel('Bed:')
        e6.setObjectName("Bed")
        e6.setFont(QFont("Arial", 16, QFont.Bold))
        e6.setStyleSheet("color:white;")
        e7 = QLineEdit()
        e7.setEnabled(False)
        e7.setFixedHeight(35)
        e7.setObjectName("TraineeInput")
        e7.setStyleSheet("background-color: #336699;color: white")
        e7.setFont(QFont("Arial", 14))
        e7.setText("John")
        e7.setAlignment(Qt.AlignCenter)
        self.trainee = e7
        e8 = QLineEdit()
        e8.setEnabled(False)
        e8.setFixedHeight(35)
        e8.setObjectName("BedInput")
        e8.setStyleSheet("background-color: #336699;color: white")
        e8.setFont(QFont("Arial", 14))
        e8.setText("1")
        e8.setAlignment(Qt.AlignCenter)
        self.bed = e8
        e = QGridLayout()
        e.addWidget(e1, 0, 0)
        e.addWidget(e2, 0, 1)
        e.addWidget(self.mentor, 1, 0)
        e.addWidget(self.lesion, 1, 1)
        e.addWidget(e5, 2, 0)
        e.addWidget(e6, 2, 1)
        e.addWidget(self.trainee, 3, 0)
        e.addWidget(self.bed, 3, 1)
        self.trainLabel.setLayout(e)

    def countTime(self):
        if self.flag:
            current_time = datetime.now()
            diff_time = (current_time - self.startTime).total_seconds()
            hour = int(diff_time // 3600)
            minute = int(diff_time % 3600 // 60)
            second = int(diff_time % 60)
            self.duraHour.setText('{:02d}'.format(hour))
            self.duraMinute.setText('{:02d}'.format(minute))
            self.duraSecond.setText('{:02d}'.format(second))

    def generateReport(self):
        self.report_thread.start()
        self.reportButton.setEnabled(False)
        self.reportButton.setStyleSheet("QPushButton"
                                        "{"
                                        "background-color: lightgrey;"
                                        "color: white;"
                                        "padding: 5px 15px;"
                                        "margin-top: 10px;"
                                        "outline: 1px;"
                                        "min-width: 8em;"
                                        "}")

        self.report_thread.report_signal(generate_flag=True)

    def stop_thread(self):
        self.plot_thread.stop()
        self.processing_thread.stop()
        self.updating = False


    def enableReport(self, report_file_path):
        self.reportButton.setEnabled(True)
        self.reportButton.setStyleSheet("QPushButton"
                                        "{"
                                        "background-color: green;"
                                        "color: white;"
                                        "padding: 5px 15px;"
                                        "margin-top: 10px;"
                                        "outline: 1px;"
                                        "min-width: 8em;"
                                        "}")
    def setupAnalytics(self):
        num_widget = self.verticalLayout.count()
        while num_widget > 0:
            widget = self.verticalLayout.itemAt(num_widget - 1).widget()
            if widget.objectName() == 'OnlineAnalytics':
                break
            num_widget -= 1
        # Create the gray rectangles
        self.rect1 = QLineEdit()
        self.rect1.setStyleSheet("background-color: gray; color: white;")
        self.rect1.setFixedWidth(300)
        self.rect1.setFixedHeight(30)
        self.rect2 = QLineEdit()
        self.rect2.setStyleSheet("background-color: gray; color: white;")
        self.rect2.setFixedWidth(300)
        self.rect2.setFixedHeight(30)
        self.rect3 = QLineEdit()
        self.rect3.setStyleSheet("background-color: gray; color: white;")
        self.rect3.setFixedWidth(300)
        self.rect3.setFixedHeight(30)
        self.rect4 = QLineEdit()
        self.rect4.setStyleSheet("background-color: gray; color: white;")
        self.rect4.setFixedWidth(300)
        self.rect4.setFixedHeight(30)

        self.rect1.setAlignment(Qt.AlignCenter)
        self.rect1.setFont(QFont("Arial", 16, QFont.Bold))
        self.rect2.setAlignment(Qt.AlignCenter)
        self.rect2.setFont(QFont("Arial", 16, QFont.Bold))
        self.rect3.setAlignment(Qt.AlignCenter)
        self.rect3.setFont(QFont("Arial", 16, QFont.Bold))
        self.rect4.setAlignment(Qt.AlignCenter)
        self.rect4.setFont(QFont("Arial", 16, QFont.Bold))

        # Create the labels
        e1 = QLabel('Time:')
        e1.setObjectName("Time")
        e1.setFont(QFont("Arial", 16, QFont.Bold))
        e1.setStyleSheet("color:white;")
        e3 = QLabel('Mentor:')
        e3.setObjectName("Mentor")
        e3.setFont(QFont("Arial", 16, QFont.Bold))
        e3.setStyleSheet("color:white;")
        e5 = QLabel('Trainee:')
        e5.setObjectName("Trainee")
        e5.setFont(QFont("Arial", 16, QFont.Bold))
        e5.setStyleSheet("color:white;")
        e7 = QLabel('NT-index:')
        e7.setObjectName("NT-index")
        e7.setFont(QFont("Arial", 16, QFont.Bold))
        e7.setStyleSheet("color:white;")

        # Create the layout for each row
        row1_layout = QHBoxLayout()
        row1_layout.addWidget(e1)
        row1_layout.addWidget(self.rect1)
        row2_layout = QHBoxLayout()
        row2_layout.addWidget(e3)
        row2_layout.addWidget(self.rect2)
        row3_layout = QHBoxLayout()
        row3_layout.addWidget(e5)
        row3_layout.addWidget(self.rect3)
        row4_layout = QHBoxLayout()
        row4_layout.addWidget(e7)
        row4_layout.addWidget(self.rect4)

        # Create the main vertical layout
        VLayout = QVBoxLayout()
        VLayout.addLayout(row1_layout)
        VLayout.addLayout(row2_layout)
        VLayout.addLayout(row3_layout)
        VLayout.addLayout(row4_layout)

        # Set the layout for the widget
        widget.setLayout(VLayout)

    def setVLayout(self, name, percent):
        widget = QtWidgets.QLabel(name.title())
        widget.setObjectName(name.title().replace(' ', '') + 'Title')
        widget.setFont(QFont('Arial', 18, QFont.Bold))
        widget.setStyleSheet("color: white;")
        self.verticalLayout.addWidget(widget, 4)
        widget1 = QtWidgets.QWidget()
        widget1.setObjectName(name.title().replace(' ', ''))
        widget1.setAttribute(Qt.WA_StyledBackground, True)
        widget1.setStyleSheet(
            f"QWidget#{name.title().replace(' ', '')}" + "{background-color: rgb(98, 154, 202); border-radius:5px;}")
        self.verticalLayout.addWidget(widget1, 21)
        gapWidget = QtWidgets.QWidget()
        gapWidget.setFixedWidth(50)  # Set the desired width for the gap
        gapWidget.setObjectName(name.title().replace(' ', '') + 'Gap')
        self.verticalLayout.addWidget(gapWidget, 5)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("iPhaser", "AI-Endo"))

    def get_frame_size(self):
        capture = cv2.VideoCapture(0)

        # Default resolutions of the frame are obtained (system dependent)
        # frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = int(capture.get(cv2.CAP_PROP_FPS))
        # fps = 30
        frame_width = 1920
        frame_height = 1080
        fps = 50
        capture.release()
        return frame_width, frame_height, fps

    def save_log_data(self):
        datas = zip(*self.log_data)
        data_dict = {}
        # [datetime.now(), self.trainee.text(), self.mentor.text(), self.bed.text(), self.pred]
        names = ["Time", "Trainee", "Trainer", "Bed", "Prediction", "Phase idle", "Phase marking", "Phase injection",
                 "Phase dissection"]
        for name, data in zip(names, datas):
            data_dict[name] = list(data)
        pd_log = pd.DataFrame.from_dict(data_dict)
        curent_date_time = "_" + datetime.now().strftime("%H-%M-%S") + ".csv"
        pd_log.to_csv(self.log_file.replace(".csv", curent_date_time), index=False, header=True)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-s", default=False, action='store_true', help="Whether save predictions")
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