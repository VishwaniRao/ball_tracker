from PyQt6 import QtGui
from PyQt6.QtGui import QIcon, QFont,  QPixmap, QPainter,QPen,QColor,QIntValidator,QDoubleValidator,QPolygon
from PyQt6.QtCore import QDir, Qt, QUrl, QSize,QEvent,QPoint
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QStyleFactory,QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QStatusBar,QLineEdit,QGridLayout,QScrollArea,QComboBox)
# from tracker import BallTracker
import cv2
import requests
import numpy as np 
import json

bgs_algos = {

    0:"FrameDifference",
    1:"StaticFrameDifference",
    2:"WeightedMovingMean",
    3:"WeightedMovingVariance",
    4:"AdaptiveBackgroundLearning",
    5:"AdaptiveSelectiveBackgroundLearning",
    6:"MixtureOfGaussianV2",
    7:"PixelBasedAdaptiveSegmenter",
    8:"SigmaDelta",
    9:"SuBSENSE",
    10:"LOBSTER",
    11:"PAWCS",
    12:"TwoPoints",
    13:"ViBe",
    14:"CodeBook"

}


class DisplayAllContours(QWidget):

    def __init__(self, *args, **kwargs):
        super(DisplayAllContours, self).__init__(*args, **kwargs)
        
        self.curveTraced = []
        self.allChosenPoints = []
        self.allChosenRadii = []

        self.Layout =QVBoxLayout()
        self.Layout.setContentsMargins(0,0,0,0)
        self.Layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.label = QLabel()
       
        self.canvas = QPixmap(500, 500)
        self.canvas.fill(Qt.GlobalColor.white)

        self.label.setPixmap(self.canvas)

        '''
        adding a reset button and save button
        '''
        self.reset = QPushButton("reset")
        self.reset.setEnabled(False)
        self.reset.setFixedHeight(24)
        self.reset.setFixedWidth(150)
        self.reset.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogDiscardButton))
        self.reset.clicked.connect(self.resetStuff)
        self.reset.setEnabled(True)

        self.save = QPushButton("save")
        self.save.setEnabled(False)
        self.save.setFixedHeight(24)
        self.save.setFixedWidth(150)
        self.save.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save.clicked.connect(self.saveFunc)
        self.save.setEnabled(True)

        self.create = QPushButton("createTraj")
        self.create.setEnabled(False)
        self.create.setFixedHeight(24)
        self.create.setFixedWidth(150)
        self.create.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.create.clicked.connect(self.createTraj)
        self.create.setEnabled(True)        

        self.white = QPushButton("White")
        self.white.setEnabled(False)
        self.white.setFixedHeight(24)
        self.white.setFixedWidth(150)
        self.white.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.white.clicked.connect(self.removeAllPts)
        self.white.setEnabled(True)                

        self.Layout.addWidget(self.label)
        self.Layout.addWidget(self.reset)
        self.Layout.addWidget(self.save)
        self.Layout.addWidget(self.create)
        self.Layout.addWidget(self.white)
      
        self.setLayout(self.Layout)




    def pushPoints(self,pts,rect,cam_pos,path,finalPerspectiveCoords,rootDir,trackerIP,radii,filename):
        self.cam_pos = cam_pos
        self.path = path
        self.perspectiveCoords = finalPerspectiveCoords
        self.rootDir = rootDir
        self.trackerIP = trackerIP
        self.radii = radii
        self.filename = filename
        self.allChosenPoints = []
        self.allChosenRadii =[]
 
        
        w,h = rect[1][0] - rect[0][0],rect[1][1] - rect[0][1]
        self.rect = rect
        self.pts = pts
        # print(self.radii)
        # print(self.pts)

        # self.Layout =QVBoxLayout()
        # self.Layout.setContentsMargins(0,0,0,0)
        # self.Layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # self.label = QLabel()
        
        # self.canvas = QPixmap(500, 500)
        # self.canvas.fill(Qt.GlobalColor.white)
        
        self.canvas.fill(Qt.GlobalColor.white)
        painter = QPainter(self.canvas)

        pen =QPen()
        pen.setWidth(4)
        pen.setColor(QColor("#EB5160"))
        painter.setPen(pen)
      
        # print("xxxxxxx")
        # print(self.pts)
        for pt in self.pts:
            painter.drawPoint(pt[0],pt[1])

        painter.end()
        self.label.setPixmap(self.canvas)


    def createTraj(self):
        body = {"rootDir":self.rootDir,"cameraPos":self.cam_pos,"path":self.path,"roi":self.rect,"pitch":self.perspectiveCoords,"pts":self.allChosenPoints}
        print(body)
        x = requests.post(self.trackerIP + 'finalContours', json = body)
        print(x.json())
        

    def saveFunc(self):
        data = {}
        count = 0
        self.allChosenPoints = np.unique(np.asarray(self.allChosenPoints),axis=0)
        for i,pt in enumerate(self.pts):
            data[int(count)] = {}
            data[int(count)]['key'] = [pt[0],pt[1],pt[2]]
            # data[int(count)]['rad'] = self.radii[i]
            if pt in self.allChosenPoints:
                data[int(count)]['val'] = 1
            else:
                data[int(count)]['val'] = 0
            count = count + 1
        
        split_ = self.filename.split('/')
        file_ = self.rootDir+'/records/'+ split_[-2]  + '-' + split_[-1].split('.')[0] + '.json'
        # with open(file_, "r") as file:
        #     entries = json.load(file)
        # entries.append(data)
        with open(file_, "w") as file:
            json.dump(data, file,indent=4,sort_keys=True)

    def resetStuff(self):
        self.allChosenPoints = []
        self.allChosenRadii = []
        self.canvas.fill(Qt.GlobalColor.white)
        painter = QPainter(self.canvas)

        pen =QPen()
        pen.setWidth(4)
        pen.setColor(QColor("#EB5160"))
        painter.setPen(pen)
      
        for pt in self.pts:
            painter.drawPoint(pt[0],pt[1])

        painter.end()
        self.label.setPixmap(self.canvas)
        # self.pushPoints(self.pts,self.rect)
        
    def removeAllPts(self):
        self.allChosenPoints = []
        self.canvas.fill(Qt.GlobalColor.white)
        self.label.setPixmap(self.canvas)
        
        # layout =QVBoxLayout()
        # layout.setContentsMargins(0,0,0,0)
        # layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # layout.addWidget(self.label)
        # layout.addWidget(self.reset)
        # layout.addWidget(self.save)
        # layout.addWidget(self.create)
        # layout.addWidget(self.white)
        self.setLayout(self.Layout)


    def mouseMoveEvent(self,e):
        self.curveTraced.append((e.pos().x(),e.pos().y()))

    def mouseReleaseEvent(self,e):
        print("mouse released" + str((e.pos().x(),e.pos().y())))
        painter = QPainter(self.canvas)

        x,y = [i[0] for i in self.curveTraced],[i[1] for i in self.curveTraced]
        p1 = np.asarray([x[0],y[0]])
        p2 = np.asarray([x[-1],y[-1]])
        # print(p1,p2)
        # p3 = np.asarray(self.pts)
        p3 = np.asarray([(pt[0],pt[1]) for pt in self.pts])
        # print(p3)
        # print(p3.shape )
        d=abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))
        best_index = np.where(d<10)[0]
        # print(best_index)
   
        chosen_pts = []
        chosen_radii = []
        for i in best_index:
            if self.pts[i][1] > p1[1] and self.pts[i][1] < p2[1]: #self.pts[i][0] > p1[0] and self.pts[i][0] < p2[0] and self.pts[i][1] > p1[1] and self.pts[i][1] < p2[1]:
                chosen_pts.append((self.pts[i][0],self.pts[i][1]))
                chosen_radii.append(self.pts[i][2])
                self.allChosenPoints.append(self.pts[i])
                self.allChosenRadii.append(self.pts[i][2])


        # print(chosen_pts)
        # print("yyyyyyyy")
        # print(chosen_radii)

        pen =QPen()
        pen.setWidth(4)
        pen.setColor(QColor(100,100,100))
        painter.setPen(pen)
      
        for i,pt in enumerate(chosen_pts):
            painter.drawPoint(pt[0],pt[1])
            xmin = int(pt[0] - chosen_radii[i])
            ymin = int(pt[1] - chosen_radii[i])
            painter.drawEllipse(xmin,ymin,2*chosen_radii[i] ,2*chosen_radii[i])

        painter.end()
        self.label.setPixmap(self.canvas)
        self.curveTraced=[]



        

class DisplayPerspectiveImage(QWidget):
    def __init__(self,path,w,h,layout,resize,perspectiveCoords):
        super().__init__()


        self.orig_w = w
        self.orig_h = h
        self.resize_val = resize
        self.w = int(w/resize)
        self.h = int(h/resize)
        self.globalLayout = layout
        self.path = path
        self.coordinates = perspectiveCoords

        self.setGeometry(0, 0, w,h)
        self.label = QLabel(self)
        self.pixmap = QPixmap(path)
        self.label.setPixmap(self.pixmap)
        

    def mousePressEvent(self, e):
        x = e.pos().x()
        y = e.pos().y()


        # print("clicked Positions are :: " + str(x) + ' ' + str(y) + " width :: " + str(self.w) + " height :: " + str(self.h) + ' len :: ' + str(len(self.coordinates)))
        if x < self.w and y < self.h:

            # if len(self.coordinates) > 4:
            #     self.coordinates = []


            self.coordinates.append((x,y))
            painter = QPainter(self.pixmap)

            pen =QPen()
            pen.setWidth(4)
            pen.setColor(QColor("#EB5160"))
            painter.setPen(pen)

            if len(self.coordinates) < 5:
                for pt in self.coordinates:
                    painter.drawPoint(pt[0],pt[1])
            if len(self.coordinates) == 4:
                points = QPolygon([
                    QPoint(self.coordinates[0][0],self.coordinates[0][1]),
                    QPoint(self.coordinates[1][0],self.coordinates[1][1]),
                    QPoint(self.coordinates[2][0],self.coordinates[2][1]),
                    QPoint(self.coordinates[3][0],self.coordinates[3][1])
                ])
                painter.drawPolygon(points)

            painter.end()
            self.label.setPixmap(self.pixmap)

            # label.setObjectName("PerspectiveImage")
            # self.globalLayout.addWidget(label)

            
            




class DisplayROIImage(QWidget):
    def __init__(self,path,w,h,layout,resize,roi_size):
        super().__init__()


        self.orig_w = w
        self.orig_h = h
        self.resize_val = resize
        self.w = int(w/resize)
        self.h = int(h/resize)
        self.globalLayout = layout
        self.path = path
        self.roi_size = int(roi_size/self.resize_val)

        self.setGeometry(0, 0, w,h)
        label = QLabel(self)
        pixmap = QPixmap(path)
        label.setPixmap(pixmap)
        

    def getRect(self):
        initial = (self.x*self.resize_val,self.y*self.resize_val)
        final =  (self.resize_val*(self.x+self.roi_size),self.resize_val*(self.y+self.roi_size))

        return([initial,final])
    
    def mousePressEvent(self, e):
        x = e.pos().x()
        y = e.pos().y()
        print("clicked Positions are :: " + str(x) + ' ' + str(y) + " width :: " + str(self.w) + " height :: " + str(self.h))
        if x < self.w and y < self.h:
            items = [self.globalLayout.itemAt(i).widget() for i in range(self.globalLayout.count())]
            for  item in items:
                if item != None:
                    if  item.objectName() == "ExtractImage":
                        self.globalLayout.removeWidget(item)

            
            self.setGeometry(0, 0, self.w,self.h)
            label = QLabel(self)
            pixmap = QPixmap(self.path)
            painter = QPainter(pixmap)

            pen =QPen()
            pen.setWidth(4)
            pen.setColor(QColor("#EB5160"))
            painter.setPen(pen)
            painter.drawRoundedRect(x, y, self.roi_size, self.roi_size, 0, 0)
            painter.end()
            label.setPixmap(pixmap)

            label.setObjectName("ExtractImage")
            self.globalLayout.addWidget(label)

            self.x = x 
            self.y = y



def createVideo(name,hideCallback,playCallback,setPos):
        '''
        Video
        '''

        btnSize = QSize(16, 16)
        posSlider = QSlider(Qt.Orientation.Horizontal)
        posSlider.setRange(0, 0)
        posSlider.sliderMoved.connect(setPos)
        posSlider.setObjectName('slider-'+name)
    

        playButton = QPushButton("Play")
        playButton.setEnabled(True)
        playButton.setFixedHeight(24)
        playButton.setFixedWidth(150)
        playButton.setIconSize(btnSize)
        playButton.setObjectName('play-'+name)
        playButton.clicked.connect(playCallback)


        hideVideo = QPushButton("hide")
        hideVideo.setEnabled(True)
        hideVideo.setFixedHeight(24)
        hideVideo.setFixedWidth(150)
        hideVideo.setIconSize(btnSize)
        hideVideo.setObjectName('hide-'+name)
        # self.hideVideo.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        hideVideo.clicked.connect(hideCallback)


        controlLayout = QGridLayout()
        controlLayout.setSpacing(1)
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(playButton,0,0)
        controlLayout.addWidget(posSlider,0,1)
        controlLayout.addWidget(hideVideo,0,2)
        
        return posSlider,controlLayout


class VideoPlayer(QWidget):

    def __init__(self, parent=None):
        super(VideoPlayer, self).__init__(parent)
        
        self.reset()
        self.ROI_val = None
        self.cam_pos = None
        self.resize_val = 4
        self.perspectiveCoords = []
        self.fileName = None
        self.minArea = None
        self.maxArea = None
        self.refinedMinArea =None
        self.trackerIP = 'http://localhost:800/'
        # self.rootDir = '/Users/muck27/Downloads/qt-stuff/trial/'

        self.rootDir = '/Volumes/Share/'
       


    def reset(self):

        '''
        Video
        '''
        self.statusBar = QStatusBar()
        self.statusBar.setFont(QFont("Noto Sans", 7))
        self.statusBar.setFixedHeight(14)

        self.mediaPlayer = QMediaPlayer()
        self.videoWidget = QVideoWidget()
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.playbackStateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.errorChanged.connect(self.handleError)
        self.videoWidget.setObjectName("RawVideo")
        self.statusBar.showMessage("Ready")

        btnSize = QSize(16, 16)
        self.positionSlider = QSlider(Qt.Orientation.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)


        openButton = QPushButton("Open Video")   
        openButton.setToolTip("Open Video File")
        openButton.setStatusTip("Open Video File")
        openButton.setFixedHeight(24)
        openButton.setIconSize(btnSize)
        openButton.setFont(QFont("Noto Sans", 8))
        openButton.setIcon(QIcon.fromTheme("document-open", QIcon("D:/_Qt/img/open.png")))
        openButton.clicked.connect(self.abrir)

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setFixedHeight(24)
        self.playButton.setIconSize(btnSize)
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.hideVideo = QPushButton("hide")
        self.hideVideo.setEnabled(False)
        self.hideVideo.setFixedHeight(24)
        self.hideVideo.setIconSize(btnSize)
        # self.hideVideo.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.hideVideo.clicked.connect(self.hideVideoCallback)


        self.controlLayout = QHBoxLayout()
        self.controlLayout.setContentsMargins(0, 0, 0, 0)
        self.controlLayout.addWidget(openButton)
        self.controlLayout.addWidget(self.playButton)
        self.controlLayout.addWidget(self.positionSlider)
        self.controlLayout.addWidget(self.hideVideo)

        '''
        GET ROI
        '''

        self.resetButton = QPushButton("Hide / Reset Image")
        self.resetButton.setEnabled(False)
        self.resetButton.setFixedHeight(24)
        self.resetButton.setFixedWidth(150)
        self.resetButton.setIconSize(btnSize)
        self.resetButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton))
        self.resetButton.clicked.connect(self.resetExtract)


        roiValue = QLineEdit()
        roiValue.setValidator(QDoubleValidator(0,500,2))
        roiValue.setFixedWidth(150)
        roiValue.textChanged.connect(self.getROI)
        roiValue.setText("ROI width/height ")

        resizeVal = QLineEdit()
        resizeVal.setValidator(QDoubleValidator(0,500,2))
        resizeVal.setFixedWidth(150)
        resizeVal.textChanged.connect(self.getResize)
        resizeVal.setText("resize value")
        

        self.extractImage = QPushButton("Set ROI")
        self.extractImage.setEnabled(False)
        self.extractImage.setFixedWidth(150)
        self.extractImage.setFixedHeight(24)
        self.extractImage.setIconSize(btnSize)
        self.extractImage.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
        self.extractImage.clicked.connect(self.setROI)
        self.extractImage.setEnabled(False)


        self.controlImageLayout = QGridLayout()
        self.controlImageLayout.addWidget(self.extractImage,0,0)
        self.controlImageLayout.addWidget(roiValue,0,1)
        self.controlImageLayout.addWidget(resizeVal,0,2)
        self.controlImageLayout.addWidget(self.resetButton,0,3)
        



        '''
        GET Perspective
        '''

        self.resetPerspective = QPushButton("Hide / Reset Image")
        self.resetPerspective.setEnabled(False)
        self.resetPerspective.setFixedHeight(24)
        self.resetPerspective.setFixedWidth(150)
        self.resetPerspective.setIconSize(btnSize)
        self.resetPerspective.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton))
        self.resetPerspective.clicked.connect(self.resetPerspectiveLayout)
        self.resetPerspective.setEnabled(True)

        self.savePerspective = QPushButton("Save")
        self.savePerspective.setEnabled(False)
        self.savePerspective.setFixedHeight(24)
        self.savePerspective.setFixedWidth(150)
        self.savePerspective.setIconSize(btnSize)
        self.savePerspective.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton))
        self.savePerspective.clicked.connect(self.savePerspectiveCoords)
        self.savePerspective.setEnabled(True)

        self.perscImage = QPushButton("Set Perspective")
        self.perscImage.setEnabled(False)
        self.perscImage.setFixedWidth(150)
        self.perscImage.setFixedHeight(24)
        self.perscImage.setIconSize(btnSize)
        self.perscImage.clicked.connect(self.setPerspective)
        self.perscImage.setEnabled(True)


        self.perspectiveLayout = QGridLayout()
        self.perspectiveLayout.addWidget(self.perscImage,0,0)
        self.perspectiveLayout.addWidget(self.resetPerspective,0,1)
        self.perspectiveLayout.addWidget(self.savePerspective,0,2)

        # '''
        # BGS the video
        # '''
        # self.crop = QPushButton("crop+bgs")
        # self.crop.setEnabled(False)
        # self.crop.setFixedWidth(150)
        # self.crop.setFixedHeight(24)
        # self.crop.setIconSize(btnSize)
        # self.crop.clicked.connect(self.requestCrop)
        # self.crop.setEnabled(True)

        # self.bgsPath = QLineEdit()
        # self.bgsPath.setObjectName("BgsPath")
        # self.bgsPath.setFixedWidth(150)
        # self.bgsPath.setReadOnly(True)

        # self.bgsLayout = QGridLayout()
        # self.bgsLayout.addWidget(self.crop,0,0)
        # self.bgsLayout.addWidget(self.bgsPath,0,1)


        # '''
        # Contours of Video
        # '''
        # self.contours = QPushButton("contours")
        # self.contours.setEnabled(False)
        # self.contours.setFixedWidth(150)
        # self.contours.setFixedHeight(24)
        # self.contours.setIconSize(btnSize)
        # self.contours.clicked.connect(self.requestContours)
        # self.contours.setEnabled(True)

        # minArea = QLineEdit()
        # minArea.setValidator(QDoubleValidator(0,500,2))
        # minArea.setFixedWidth(150)
        # minArea.textChanged.connect(self.setMinArea)
        # minArea.setText("min area")

        # maxArea = QLineEdit()
        # maxArea.setValidator(QDoubleValidator(0,500,2))
        # maxArea.setFixedWidth(150)
        # maxArea.textChanged.connect(self.setMaxArea)
        # maxArea.setText("max area")

        # refinedMinArea = QLineEdit()
        # refinedMinArea.setValidator(QDoubleValidator(0,500,2))
        # refinedMinArea.setFixedWidth(150)
        # refinedMinArea.textChanged.connect(self.setRefinedMinArea)
        # refinedMinArea.setText("refined min area")

        # self.contoursLayout = QGridLayout()
        # self.contoursLayout.addWidget(self.contours,0,0)
        # self.contoursLayout.addWidget(minArea,0,1)
        # self.contoursLayout.addWidget(maxArea,0,2)
        # self.contoursLayout.addWidget(refinedMinArea,0,3)


        '''
        Debug Contours 
        '''
        self.debugcontours = QPushButton("debug contours")
        self.debugcontours.setEnabled(False)
        self.debugcontours.setFixedWidth(150)
        self.debugcontours.setFixedHeight(24)
        self.debugcontours.setIconSize(btnSize)
        self.debugcontours.clicked.connect(self.requestDebugContours)
        self.debugcontours.setEnabled(True)

        minArea = QLineEdit("Rows n Cols")
        # minArea.setValidator(QDoubleValidator(0,500,2))
        minArea.setFixedWidth(150)
        minArea.textChanged.connect(self.setMinArea)
        minArea.setText("rows n cols")

        maxArea = QLineEdit("Max Area")
        maxArea.setValidator(QDoubleValidator(0,500,2))
        maxArea.setFixedWidth(150)
        maxArea.textChanged.connect(self.setMaxArea)
        maxArea.setText("max area")

        refinedMinArea = QLineEdit("Refined Area")
        refinedMinArea.setValidator(QDoubleValidator(0,500,2))
        refinedMinArea.setFixedWidth(150)
        refinedMinArea.textChanged.connect(self.setRefinedMinArea)
        refinedMinArea.setText("refined min area")

        camPos = QLineEdit("Dexterity")
        camPos.setValidator(QDoubleValidator(0,1,2))
        camPos.setFixedWidth(150)
        camPos.textChanged.connect(self.selectCamPos)
        camPos.setText("0 : bowler left, 1 : bowler right")
       
        bgs_dropdown = []
        for bgs in bgs_algos.keys():
            bgs_dropdown.append(bgs_algos[bgs])

        self.bgs_dropdown = QComboBox()
        self.bgs_dropdown.addItems(bgs_dropdown)

        self.debugcontoursLayout = QGridLayout()
        self.debugcontoursLayout.addWidget(self.debugcontours,0,0)
        self.debugcontoursLayout.addWidget(minArea,0,1)
        self.debugcontoursLayout.addWidget(maxArea,0,2)
        self.debugcontoursLayout.addWidget(refinedMinArea,0,3)
        self.debugcontoursLayout.addWidget(camPos,0,4)
        self.debugcontoursLayout.addWidget(self.bgs_dropdown,0,5)

        '''
        More Video 
        '''
        self.mp1 = QMediaPlayer()
        self.vw1 = QVideoWidget()
        self.mp1.setVideoOutput(self.vw1)
        self.mp1.playbackStateChanged.connect(self.mediaStateChanged1)
        self.mp1.positionChanged.connect(self.positionChanged1)
        self.mp1.durationChanged.connect(self.durationChanged1)
        self.mp1.errorChanged.connect(self.handleError1)
        self.vw1.setObjectName("vw1")
        self.posSlider1, self.videoLayout1 = createVideo('vw1',self.hideVideoCallback1,self.play1,self.setPos1)
  

        # items = [self.videoLayout1.itemAt(i).widget() for i in range(self.videoLayout1.count())]
        # for item in items:
        #     print(item.objectName())


        self.mp2 = QMediaPlayer()
        self.vw2 = QVideoWidget()
        self.mp2.setVideoOutput(self.vw2)
        self.mp2.playbackStateChanged.connect(self.mediaStateChanged2)
        self.mp2.positionChanged.connect(self.positionChanged2)
        self.mp2.durationChanged.connect(self.durationChanged2)
        self.mp2.errorChanged.connect(self.handleError2)
        self.vw2.setObjectName("vw2")
        # self.statusBar.showMessage("Ready")

        self.posSlider2, self.videoLayout2 = createVideo('vw2',self.hideVideoCallback2,self.play2,self.setPos2)
        # items = [self.videoLayout2.itemAt(i).widget() for i in range(self.videoLayout2.count())]
        # for item in items:
        #     print(item.objectName())


        self.mp3 = QMediaPlayer()
        self.vw3 = QVideoWidget()
        self.mp3.setVideoOutput(self.vw3)
        self.mp3.playbackStateChanged.connect(self.mediaStateChanged3)
        self.mp3.positionChanged.connect(self.positionChanged3)
        self.mp3.durationChanged.connect(self.durationChanged3)
        self.mp3.errorChanged.connect(self.handleError3)
        self.vw3.setObjectName("vw3")
        # self.statusBar.showMessage("Ready")

        self.posSlider3, self.videoLayout3  = createVideo('vw3',self.hideVideoCallback3,self.play3,self.setPos3)
        # items = [self.videoLayout3.itemAt(i).widget() for i in range(self.videoLayout3.count())]
        # for item in items:
        #     print(item.objectName())


        self.allcontours = QLabel(self)

        self.interactiveContours = DisplayAllContours()
        self.interactiveContours.setObjectName("interactiveContours")
        

        self.allcontours_btn = QPushButton("hide-contours")
        # self.allcontours_btn.setEnabled(False)
        self.allcontours_btn.setFixedWidth(150)
        self.allcontours_btn.setFixedHeight(24)
        self.allcontours_btn.setIconSize(btnSize)
        self.allcontours_btn.clicked.connect(self.hideContours)

        self.contourLayout = QHBoxLayout()
        self.contourLayout.addWidget(self.allcontours)
        self.contourLayout.addWidget(self.interactiveContours)
        self.contourLayout.addWidget(self.allcontours_btn)

         
        '''
        Push in the layout
        '''


        self.layout = QGridLayout()
        self.layout.columnStretch(5)
        self.layout.addWidget(self.videoWidget,0,0) ### 0th widget
        self.layout.addLayout(self.controlLayout,1,0)
        self.layout.addLayout(self.controlImageLayout,2,0)
        self.layout.addLayout(self.perspectiveLayout,3,0)
        # self.layout.addLayout(self.bgsLayout)
        # self.layout.addLayout(self.contoursLayout)
        self.layout.addLayout(self.debugcontoursLayout,4,0)
        
        self.layout.addWidget(self.vw1,5,0)
        self.layout.addLayout(self.videoLayout1,6,0)
        self.layout.addWidget(self.vw2,7,0)
        self.layout.addLayout(self.videoLayout2,8,0)
        self.layout.addWidget(self.vw3,9,0)
        self.layout.addLayout(self.videoLayout3,10,0)
        self.layout.addLayout(self.contourLayout,11,0)
        # self.layout.addWidget(self.allcontours,11,0)
        # self.layout.addWidget(self.allcontours_btn,11,1)
        # self.layout.addWidget(self.interactiveContours,12,0)

        self.layout.setSpacing(0)
        self.setLayout(self.layout)
    

    def selectCamPos(self,text):
        try:
            self.cam_pos = int(text)
        except:
            pass

    def getResize(self,text):
        try:
            self.resize_val = int(text)
            self.extractImage.setEnabled(True)
            # print(self.resize_val)
        except:
            pass

    def hideContours(self):
        # print(self.allcontours.isVisible())
        if self.allcontours.isVisible():
            self.allcontours.hide()
        else:
            self.allcontours.show()

        if self.interactiveContours.isVisible():
            self.interactiveContours.hide()
        else:
            self.interactiveContours.show()

    def requestDebugContours(self):
        for key_ in bgs_algos.keys():
            if bgs_algos[key_] == self.bgs_dropdown.currentText():
                bgs_key = key_
        
        # self.interactiveContours.pushPoints([],self.roiWidget.getRect(),self.cam_pos,localFile,self.finalPerspectiveCoords,self.rootDir,self.trackerIP)
        localFile = self.fileName
        body = {"rootDir":self.rootDir,"cameraPos":self.cam_pos,"path":localFile,"roi":self.roiWidget.getRect(),"pitch":self.finalPerspectiveCoords,'minArea':self.minArea,'maxArea':self.maxArea,'refinedMinArea':self.refinedMinArea,'bgs':bgs_key}
        print(body)
        try:
            x = requests.post(self.trackerIP + 'getDebugContours', json = body)

            debug_path = self.rootDir+'/indiContours/'+x.json()['bgs'].split('/')[-1]
            bgs_path = self.rootDir+'/bgs/'+x.json()['bgs'].split('/')[-1]
            contours_path = self.rootDir+'/contours/'+x.json()['bgs'].split('/')[-1]
            debug_path_img = self.rootDir+'/indiContours/'+x.json()['bgs'].split('/')[-1].split('.')[0] + '.jpg'
            centers= x.json()['pts']
            radii = x.json()['radius']
            print(x.json())
            # print(len(centers))
            # print(radii)
            self.interactiveContours.pushPoints(centers,self.roiWidget.getRect(),self.cam_pos,localFile,self.finalPerspectiveCoords,self.rootDir,self.trackerIP,radii,self.fileName)

            # localFile = '/Users/muck27/Downloads/qt-stuff/trial/bro1-5.mp4'
            # self.interactiveContours.pushPoints([],[(0,0),(200,200)],self.cam_pos,localFile,[],self.rootDir,self.trackerIP)
            # centers = [[157, 103], [141, 147], [159, 165], [151, 136], [157, 109], [138, 67], [138, 152], [142, 174], [141, 137], [156, 117], [43, 40], [61, 68], [74, 94], [86, 117], [95, 138], [103, 155], [117, 189], [111, 173], [123, 204], [129, 165], [174, 140]]
            # bgs_path = self.rootDir + '/bgs/' + 'bro1-5.avi'
            # debug_path = self.rootDir + '/indiContours/' + 'bro1-5.avi'
            # contours_path = self.rootDir + '/contours/' + 'bro1-5.avi'
            # debug_path_img = self.rootDir + '/indiContours/' + 'bro1-5.jpg'
            # self.interactiveContours.pushPoints(centers,[(0,0),(200,200)],self.cam_pos,localFile,[],self.rootDir,self.trackerIP)


            a = cv2.imread(debug_path_img)
            a = cv2.resize(a,(500,500))
            cv2.imwrite(debug_path_img,a)
        

            self.mp1.setSource(QUrl.fromLocalFile(bgs_path))
            self.mp1.play()
            self.mp2.setSource(QUrl.fromLocalFile(debug_path))
            self.mp2.play()
            self.mp3.setSource(QUrl.fromLocalFile(contours_path))
            self.mp3.play()

            pixmap = QPixmap(debug_path_img)
            self.allcontours.setPixmap(pixmap)
        except:
            print("bad response")
        # self.debugcontoursPath.setText(contours_path)
        

    def setMinArea(self,text):
        try:
            print(text)
            r,c = text.split(',')
            print(r,c)
            self.minArea = [int(r),int(c)]
        except:
            pass

    def setMaxArea(self,text):
        try:
            self.maxArea = int(text)
        except:
            pass

    def setRefinedMinArea(self,text):
        try:
            self.refinedMinArea = int(text)
        except:
            pass

    def requestContours(self):
        localFile = self.fileName.split('/')[-1]
        body = {"path":localFile,"roi":self.roiWidget.getRect(),"pitch":self.finalPerspectiveCoords}
        x = requests.post(self.trackerIP + 'getContours', json = body)
        contours_path = './contours/'+x.json()['path'].split('/')[-1]
        self.contoursPath.setText(contours_path)
        

    def requestCrop(self):
        localFile = self.fileName.split('/')[-1]
        body = {"path":localFile,"roi":self.roiWidget.getRect(),"pitch":self.finalPerspectiveCoords}
        x = requests.post(self.trackerIP + 'cropROI', json = body)
        bgs_path = './bgs/'+x.json()['path'].split('/')[-1]

        self.bgsPath.setText(bgs_path)
        self.contours.setEnabled(True)



    def savePerspectiveCoords(self):
        self.finalPerspectiveCoords = []
        for pt in self.perspectiveCoords:
            self.finalPerspectiveCoords.append((pt[0]*self.resize_val,pt[1]*self.resize_val))
        # print(self.finalPerspectiveCoords)

    def removeObjectPyQT(self,name):
        items = [self.layout.itemAt(i).widget() for i in range(self.layout.count())]
        for  item in items:
            if item != None:
                if  item.objectName() == name:
                    print(name)
                    self.layout.removeWidget(item)

    def resetPerspectiveLayout(self):
      
        self.perspectiveCoords = []
        items = [self.layout.itemAt(i).widget() for i in range(self.layout.count())]
        for  item in items:
            if item != None:
                # print(item.objectName())
                if  item.objectName() == "PerspectiveImage":
                    self.layout.removeWidget(item)
       
    def resetExtract(self):
        
        items = [self.layout.itemAt(i).widget() for i in range(self.layout.count())]
        for  item in items:
            if item != None:
                if  item.objectName() == "ExtractImage":
                    self.layout.removeWidget(item)

    def getROI(self,text):
        try:
            self.ROI_val = int(text)
            self.extractImage.setEnabled(True)
        except:
            pass

    def getImage(self):
        print(self.fileName)
        cap = cv2.VideoCapture(self.fileName)
        ret,frame_ = cap.read()
        self.path = './outputs/fullImage.png'
        self.frame_h,self.frame_w,c = frame_.shape
        frame_ = cv2.resize(frame_, (int(self.frame_w/self.resize_val), int(self.frame_h/self.resize_val)),interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(self.path,frame_)
        cap.release()

    def setROI(self):

        self.getImage()

        ROILayout = QGridLayout()
        ROILayout.setRowStretch(1,1)
        self.roiWidget  = DisplayROIImage(self.path,self.frame_w,self.frame_h,self.layout,self.resize_val,self.ROI_val)
        roiLabel = QLabel("ROI: Click on Image for ROI")
        ROILayout.addWidget(roiLabel,0,0)
        ROILayout.addWidget(self.roiWidget,1,0)

        widget = QWidget()
        widget.setObjectName("ExtractImage")
        widget.setLayout(ROILayout)

        self.layout.addWidget(widget)
        self.setLayout(self.layout)


    def setPerspective(self):

        self.getImage()

        ROILayout = QGridLayout()
        ROILayout.setRowStretch(1,1)


        perspectiveWidget  = DisplayPerspectiveImage(self.path,self.frame_w,self.frame_h,self.layout,self.resize_val,self.perspectiveCoords)
        persplectiveLabel = QLabel("tl->tr->bl->br")
        ROILayout.addWidget(persplectiveLabel,0,1)
        ROILayout.addWidget(perspectiveWidget,1,1)

        widget = QWidget()
        widget.setObjectName("PerspectiveImage")
        widget.setLayout(ROILayout)

        self.layout.addWidget(widget)
        self.setLayout(self.layout)   




    def abrir(self):
       
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Media",
                ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")

        if fileName != '':
            print(fileName)
            # self.extractImage.setEnabled(True)
            self.resetButton.setEnabled(True)
            self.mediaPlayer.setSource(QUrl.fromLocalFile(fileName))
            self.playButton.setEnabled(True)
            self.statusBar.showMessage(fileName)
            self.fileName = fileName
            self.play()
            self.hideVideo.setEnabled(True)

    def hideVideoCallback(self):
      
        if self.videoWidget.isHidden():
            self.videoWidget.show()
        else:
            self.videoWidget.hide()

    def play(self):
        if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()


    def hideVideoCallback3(self):
        # print("hide 1")
        if self.vw3.isHidden():
            self.vw3.show()
        else:
            self.vw3.hide()

    def play3(self):
        # print("play 1")
        if self.mp3.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.mp3.pause()
        else:
            self.mp3.play()


    def hideVideoCallback1(self):
        # print("hide 1")
        if self.vw1.isHidden():
            self.vw1.show()
        else:
            self.vw1.hide()

    def play1(self):
        # print("play 1")
        if self.mp1.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.mp1.pause()
        else:
            self.mp1.play()

    def hideVideoCallback2(self):
        # print("hide 2")
        if self.vw2.isHidden():
            self.vw2.show()
        else:
            self.vw2.hide()

    def play2(self):
        # print("play 2")
        if self.mp2.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.mp2.pause()
        else:
            self.mp2.play()


    def mediaStateChanged(self, state):
        self.setWindowTitle(self.fileName)
        if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.statusBar.showMessage("Error: " + self.mediaPlayer.errorString())


    '''
    functions for all videos
    '''
    def mediaStateChanged1(self, state):
        print('mediachanged')
        # if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
        #     self.playButton.setIcon(
        #             self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        # else:
        #     self.playButton.setIcon(
        #             self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def positionChanged1(self, position):
        
        self.posSlider1.setValue(position)

    def durationChanged1(self, duration):
        self.posSlider1.setRange(0, duration)

    def setPos1(self, position):
        self.mp1.setPosition(position)

    def handleError1(self):
        print("handle error for vid 1 ")
        # self.playButton.setEnabled(False)
        # self.statusBar.showMessage("Error: " + self.mediaPlayer.errorString())


    def mediaStateChanged2(self, state):
        print('mediachanged')
        # if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
        #     self.playButton.setIcon(
        #             self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        # else:
        #     self.playButton.setIcon(
        #             self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def positionChanged2(self, position):
        
        self.posSlider2.setValue(position)

    def durationChanged2(self, duration):
        self.posSlider2.setRange(0, duration)

    def setPos2(self, position):
        self.mp2.setPosition(position)

    def handleError2(self):
        print("handle error for vid 2 ")
        # self.playButton.setEnabled(False)
        # self.statusBar.showMessage("Error: " + self.mediaPlayer.errorString())

    def mediaStateChanged3(self, state):
        print('mediachanged')
        # if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
        #     self.playButton.setIcon(
        #             self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        # else:
        #     self.playButton.setIcon(
        #             self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def positionChanged3(self, position):
        
        self.posSlider3.setValue(position)

    def durationChanged3(self, duration):
        self.posSlider3.setRange(0, duration)

    def setPos3(self, position):
        self.mp3.setPosition(position)

    def handleError3(self):
        print("handle error for vid 3 ")
        # self.playButton.setEnabled(False)
        # self.statusBar.showMessage("Error: " + self.mediaPlayer.errorString())

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    player = VideoPlayer()
    # # player.setWindowTitle("Player")
    # player.connectNotifyset
    player.resize(900, 600)
    player.show()
    sys.exit(app.exec())
