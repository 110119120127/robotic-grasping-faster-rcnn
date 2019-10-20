# https://stackoverflow.com/questions/6783194/background-thread-with-qthread-in-pyqt

# worker.py
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
import time
import numpy as np

import sys
sys.path.append('./devices')
from devices.cam_rs415 import RealSenseD400
from devices.controller import Controller
from devices.gripper import Gripper

from predict import PoseEst

class Worker(QObject):
    # This signal emits when a new frame comes
    newFrame = pyqtSignal(object)

    
    def __init__(self, delay, cam=None, predictor=None, robot=None, gripper=None):
        super(Worker, self).__init__()
        
        self.delay = delay
        self.predictor = predictor
        self.cam = cam
        
        self.robot = None
        
        if gripper:
            self.gripper = gripper
            #self.gripper.gripper_reset()
            self.robot = robot
            #self.robot.power_on()
            
            self.robot_speed = 1500
            #self.robot.move_robot_pos('-27370', '-234660', '69030', 
                            #'-1729152', '-11155', '-206430', self.robot_speed)
            #QThread.msleep(2000)
            

            self.trans_mat = np.array([
                [0.6219, -0.0021, 0.],
                [-0.0028, -0.6218, 0.],
                [-337.3547, -163.6015, 1.0]
            ])
        
        #self.image = self.cam.start_stream()
        self.__bRobotMoveFinished = True
        self.__abort = False

    @pyqtSlot()
    def generateVideo(self):
        while True:
            if self.__abort:
                break
                
            image = self.cam.start_stream()
            # the newFrame signal emits an object (image)   
            self.newFrame.emit(image)
            app.processEvents()
            QThread.msleep(self.delay)
   
    @pyqtSlot()
    def predict(self):
        box3_pos = ['-501159', '-305128', '55016', '-1698058', '-700', '-1051343']
        z_plane = str(0)
        z_box3 = str(9000)
        speed = 2000
        pos = self.robot.get_robot_pos()
        
        print('Outside loop')
        
        while True:
            if self.__abort:
                break
            if self.__bRobotMoveFinished:
                img = self.cam.start_stream()
                im = np.copy(img) 
                start = time.perf_counter()
                pose, imgBRG, imgRGB = self.predictor.pose_est(im)
                end = time.perf_counter()
                print(end - start)
                
                if len(pose) < 1:
                    continue
                
                # the newFrame signal emits an object (im)   
                self.newFrame.emit(imgBRG)
                QThread.msleep(500)
                     
                xc = pose[2] + 275
                yc = pose[3] + 165
                angle = -(90 - pose[4]) * 10000

                img_pos = np.array([xc, yc, 1]).reshape(1, 3)
                new_pos = np.dot(img_pos, self.trans_mat)
                new_pos = np.int32(new_pos*1000)
                print(new_pos)
                
                
                pos[0] = str(new_pos[0][0])
                pos[1] = str(new_pos[0][1])
                pos[5] = str(angle)
                
                QThread.msleep(1000)
                # move to obj pos                    
                self.robot.move_robot_pos(pos[0], pos[1], pos[2], 
                                pos[3], pos[4], pos[5], speed)
                QThread.msleep(2000)
                self.robot.move_robot_pos(pos[0], pos[1], pos[2], 
                                pos[3], pos[4], pos[5], speed)                
                QThread.msleep(1000)
                # move down to the working plane
                self.robot.move_robot_pos(pos[0], pos[1], z_plane, 
                                pos[3], pos[4], pos[5], speed)
                QThread.msleep(1000)
                self.robot.move_robot_pos(pos[0], pos[1], z_plane, 
                                pos[3], pos[4], pos[5], speed)                
                QThread.msleep(500)
                # close the gripper
                self.gripper.gripper_off()
                QThread.msleep(500)
                # move up
                self.robot.move_robot_pos(pos[0], pos[1], pos[2], 
                                pos[3], pos[4], pos[5], speed)
                QThread.msleep(500)
                self.robot.move_robot_pos(pos[0], pos[1], pos[2], 
                                pos[3], pos[4], pos[5], speed)
                QThread.msleep(500)
                # move to the box pos
                self.robot.move_robot_pos(box3_pos[0], box3_pos[1], box3_pos[2], 
                                    box3_pos[3], box3_pos[4], box3_pos[5], speed)                    
                QThread.msleep(1000)
                self.robot.move_robot_pos(box3_pos[0], box3_pos[1], box3_pos[2], 
                                    box3_pos[3], box3_pos[4], box3_pos[5], speed) 
                QThread.msleep(500)
                # move down a little bit
                self.robot.move_robot_pos(box3_pos[0], box3_pos[1], z_box3, 
                                    box3_pos[3], box3_pos[4], box3_pos[5], speed) 
                QThread.msleep(1000)
                self.robot.move_robot_pos(box3_pos[0], box3_pos[1], z_box3, 
                                    box3_pos[3], box3_pos[4], box3_pos[5], speed) 
                QThread.msleep(1000)
                # open the gripper
                self.gripper.gripper_on()
                QThread.msleep(500)
                # move up a little bit
                self.robot.move_robot_pos(box3_pos[0], box3_pos[1], box3_pos[2], 
                                    box3_pos[3], box3_pos[4], box3_pos[5], speed) 
                QThread.msleep(500)
                self.robot.move_robot_pos(box3_pos[0], box3_pos[1], box3_pos[2], 
                                    box3_pos[3], box3_pos[4], box3_pos[5], speed) 
                QThread.msleep(500)
                pos = self.robot.get_robot_pos()

                #self.__bRobotMoveFinished = False
                
                
            app.processEvents()
            QThread.msleep(self.delay) 
                
            
                            
    def abort(self):
        self.__abort = True
    def robot_finish(self):
        self.__bRobotMoveFinished = True       
                      
# main.py
#from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap
import sys


class Form(QWidget):
    sig_abort_workers = pyqtSignal()
    
    def __init__(self):
        super(Form, self).__init__()
        
        self.rs400 = RealSenseD400()
        self.robot = Controller()
        self.gripper = Gripper()
        self.predictor = PoseEst()
        
        self.initUI()

    def initUI(self):
        self.lbl_video = QLabel(self)
        #self.lbl_video.setGeometry(160, 100, 640, 480)
        
        self.lbl_image = QLabel(self)
        #self.lbl_image.setGeometry(820, 620, 640, 480)
        
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.lbl_video)
        hbox1.addWidget(self.lbl_image)
        
        self.btn_init = QPushButton()
        self.btn_init.clicked.connect(self.init)
        self.btn_init.setText('Init')
        
        self.btn_start = QPushButton()
        self.btn_start.clicked.connect(self.start)
        self.btn_start.setText('Start')
        
        self.btn_stop = QPushButton()
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setText('Stop')
        
        self.btn_get_pos = QPushButton()
        self.btn_get_pos.clicked.connect(self.getPos)
        self.btn_get_pos.setText('Get pos')
        
        hbox2 = QHBoxLayout()
        hbox2.addStretch(1)
        hbox2.addWidget(self.btn_init)
        hbox2.addStretch(1)
        hbox2.addWidget(self.btn_get_pos)
        hbox2.addStretch(1)
        hbox2.addWidget(self.btn_start)
        hbox2.addStretch(1)
        hbox2.addWidget(self.btn_stop)
        hbox2.addStretch(1)
        
        vbox = QVBoxLayout()
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox1)

        self.setLayout(vbox)
        
        #self.setGeometry(100, 100, 1600, 1000)
        self.setWindowTitle('Robot Grasping using Multi-task Faster R-CNN')
        self.show()
        
    @pyqtSlot(object)
    def updateVideo(self, img):
        self.image = np.copy(img)
        h, w, c = self.image.shape
        bytesPerLine = 3 * w
        qImg = QImage(self.image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.qImg = qImg.rgbSwapped()
        self.lbl_video.setPixmap(QPixmap(self.qImg)) 
    
    @pyqtSlot(object)
    def updatePredict(self, img):
        im = np.copy(img) 
        #pose, imgBRG, imgRGB = self.predictor.pose_est(im)
        h, w, c = im.shape
        bytesPerLine = 3 * w
        qImg = QImage(im.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.lbl_image.setPixmap(QPixmap(qImg))               
          

    @pyqtSlot()
    def init(self):
        self.robot.power_on()
        QThread.msleep(500)
        self.robot.move_robot_pos('-27370', '-234660', '69030', 
                            '-1729152', '-11155', '-206430', 2000)
        QThread.msleep(2000)
        self.gripper.gripper_reset()
        
        self.btn_init.setDisabled(True) 
        
    @pyqtSlot()
    def start(self):
        # thread 1
        # 1 - create Worker and Thread inside the Form
        self.worker_video = Worker(33, self.rs400)  # no parent!
        self.thread_video = QThread()  # no parent!
        
        # connect the newFrame signal to the updateVideo slot
        # because the newFrame signal emits an image, so the updateVideo func 
        # receives that image
        self.worker_video.newFrame.connect(self.updateVideo)
         
        # move the worker to the main thread
        self.worker_video.moveToThread(self.thread_video)
        
        self.sig_abort_workers.connect(self.worker_video.abort)
        
        self.thread_video.started.connect(self.worker_video.generateVideo)
        self.thread_video.start()
              
        # thread 2
        self.worker_predict = Worker(500, self.rs400, self.predictor, 
                                    robot=self.robot , gripper=self.gripper)
        self.thread_predict = QThread()  # no parent!
        self.worker_predict.newFrame.connect(self.updatePredict)
        
        self.worker_predict.moveToThread(self.thread_predict)
        
        self.sig_abort_workers.connect(self.worker_predict.abort)
        
        self.thread_predict.started.connect(self.worker_predict.predict)
        self.thread_predict.start()
        
 
        self.btn_start.setDisabled(True) 
    
    def getPos(self):   
        pos = self.robot.get_robot_pos()
        print('Current position: {}'.format(pos))
    
    @pyqtSlot()
    def stop(self):
        self.sig_abort_workers.emit()
        self.thread_video.quit()
        self.thread_video.wait() 
        self.thread_predict.quit()
        self.thread_predict.wait()
        self.rs400.stop()
        self.robot.power_off()
        self.btn_stop.setDisabled(True)  
        
if __name__ == "__main__":
    app = QApplication([])

    form = Form()
    form.show()

    sys.exit(app.exec_())
    
    
