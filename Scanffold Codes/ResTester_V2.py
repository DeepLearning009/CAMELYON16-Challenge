'''
Created on 22 Jun 2016

@author: hjlin
'''
import sys
import os
#from boto.sdb.db.test_db import test_list
caffe_root = '/home/hjlin/caffe-master/'
sys.path.insert(0,caffe_root+'python')
import caffe
import cv2
import numpy as np
import openslide

caffe.set_mode_gpu()
caffe.set_device(0)

class CrossSlideDetecter:
    def __init__(self):
        '''
        Constructor
        '''
        self._win_size = 227
        self._step_size = 227
        self._workingLevel = 0
        self._VGGcsvDir = ""
        self._outputDir = ""
        self._OV_level = 4
        self._OV_win_size = (self._win_size/2**(self._OV_level-self._workingLevel)) * 2**(3-self._workingLevel-1)
        self._OV_step_size = self._OV_win_size
        self._avg_list = [[135,85,164],[155,100,135],[165,125,151],[187,155,180],[188,130,160],[145,88,122],[110,75,115],[67,28,120],[100,47,140],[100, 64, 110]]
        
        self._TP = self._TN = self._FP = self._FN =0
        
        self._m_threshold = 0.5
    
    def InitializeDetecter(self):
        self._net = caffe.Classifier( self._NET_FILE, self._TRAINED_MODEL_FILE )
        if os.path.exists(self._outputDir) is False:
            os.system('mkdir '+self._outputDir)  
        
    def _GetPatch(self, TIFFImg, start_w, start_h, windowShape, workingLevel):
        tile = np.array(TIFFImg.read_region((start_w, start_h), workingLevel, windowShape)) 
        return tile;
    
    def SetWinSize(self, winSize):
        '''Set the size of window.
           Window size will be 1/winProportion of Img height
           Default winProportion is 24
        '''
        self._win_size = winSize
        
        
    def SetStepSize(self, stepSize):
        '''Set the step size of window.
        '''
        self._step_size = stepSize
    
    def SetWorkingLevel(self, workingLevel):
        '''Set the level of slide to fetch the data.
           Default winProportion is 1
        '''
        self._workingLevel = workingLevel
    
    def SetNetDeployFile(self, netDeployFile):
        '''Set the level of slide to fetch the data.
           Default winProportion is 1
        '''
        self._NET_FILE = netDeployFile
    
    def SetTrainedModelFile(self, trainedModelFile):
        '''Set the level of slide to fetch the data.
           Default winProportion is 1
        '''
        self._TRAINED_MODEL_FILE = trainedModelFile
        
    def SetVGGcsvDir(self, VGGcsvDir):
        '''Set the size of window.
           Window size will be 1/winProportion of Img height
           Default winProportion is 24
        '''
        self._VGGcsvDir = VGGcsvDir 

    def SetOutputDir(self, outputDir):
        '''Set the size of window.
           Window size will be 1/winProportion of Img height
           Default winProportion is 24
        '''
        self._outputDir = outputDir    
    
    def ResetTFPN(self):
        self._TP = self._TN = self._FP = self._FN =0
     
    def PrintAccuracy(self):
        if self._TP + self._TN + self._FP + self._FN == 0:
            print "No Data to present!!"
        file = open(self._outputDir + "/Accuracy_Overall.txt",'w')
        if not file:
            print "Cannot open the file %s for writing" %(self._outputDir + "/Accuracy_Overall.txt")
        
        prt1 = "Overall Accuracy: %f" %(float(self._TP+self._TN)/(self._TP + self._TN + self._FP + self._FN))
        prt2 = "Overall True Positive: %f" %(float(self._TP)/(self._TP + self._FP + self._FN))
        print prt1
        print prt2
        file.write(prt1+'\n')
        file.write(prt2+'\n')
        file.close()
    
    def DetectWithMask(self, slidePath, maskPath, DbatchSize = 60):
        slideFileName = slidePath.split('/')[-1]
        dataName = slideFileName.split('.tif')[0]
        
        slide = openslide.open_slide(slidePath)
        mask = openslide.open_slide(maskPath)
        
        window_H = window_W = self._win_size
        windowShape = (window_H, window_W)
        EXTwindow_H = EXTwindow_W = self._win_size +int(self._win_size*1.0)
        EXTwindowShape = (EXTwindow_H, EXTwindow_W)
        
        step = self._step_size * 2**self._workingLevel
        L0_win_size = self._win_size * 2**self._workingLevel
        
        coordinate_list = []
        
        if os.path.exists(self._VGGcsvDir + "/" +dataName+".csv"):
            #read coordinates from .txt file
            coorPath = self._VGGcsvDir + "/" + dataName + ".csv"
            file = open( coorPath,'r')
            coor_lines = file.readlines()
            for i in xrange(len(coor_lines)):
                line = coor_lines[i]
                elems = line.rstrip().split(',')
                #labelCoor = int(elems[0])
                WCoor = int(elems[1])
                HCoor = int(elems[2]) 

                maskTile  = self._GetPatch(mask, WCoor- windowShape[0]/2, HCoor- windowShape[1]/2, windowShape, self._workingLevel)
                r2,g2,b2,a2 = cv2.split(maskTile)
                maskTile = cv2.merge([r2])                 

                if maskTile[maskTile.shape[0]/2][maskTile.shape[1]/2] > 100:
                    coordinate_list.append([WCoor, HCoor, 1])
                else:
                    coordinate_list.append([WCoor, HCoor, 0])
        else:
            print "Failure to find the coor file: " + self._VGGcsvDir + "/" +dataName+".csv"
            return False
        
        m_iter = 0
        t_iter = 0
        all_iter = len(coordinate_list)
        interval = all_iter/20.0
        
        TN=TP=FN=FP=0
        
        dz = 0
        t_coor_list = []
        img_list = []
        Positive_coords = []
        print "Begin Testing...%d pixels to be predict" %all_iter
        
        for i in xrange(all_iter):
            m_iter = m_iter + 1
            t_iter = t_iter + 1
            if t_iter > interval:
                t_iter = 0
                print "Detecting Complete: %.2f %%" %(float(m_iter)*100/all_iter)    
            
            WCoor = coordinate_list[i][0] 
            HCoor = coordinate_list[i][1] 
            labelCoor = coordinate_list[i][2]
            coor = [WCoor, HCoor, labelCoor]
            
            slideTile = self._GetPatch(slide, WCoor- EXTwindowShape[0]/2, HCoor- EXTwindowShape[1]/2, EXTwindowShape, self._workingLevel)
            slideTile = slideTile.astype('float32')
            r,g,b,a = cv2.split(slideTile)
            slideTile_sw = cv2.merge([r-185, g-50, b-185])
            
            #Four corner and one center patches to predict
            img_list.append(slideTile_sw[0:windowShape[0], 0:windowShape[1], :])
            img_list.append(slideTile_sw[EXTwindowShape[0]-windowShape[0]:EXTwindowShape[0], 0:windowShape[1], :])
            img_list.append(slideTile_sw[0:windowShape[0], EXTwindowShape[1]-windowShape[1]:EXTwindowShape[1], :])
            img_list.append(slideTile_sw[EXTwindowShape[0]-windowShape[0]:EXTwindowShape[0], EXTwindowShape[1]-windowShape[1]:EXTwindowShape[1], :])
            img_list.append(slideTile_sw[slideTile_sw.shape[0]/2-windowShape[0]/2:slideTile_sw.shape[0]/2-windowShape[0]/2+windowShape[0], slideTile_sw.shape[1]/2-windowShape[1]/2:slideTile_sw.shape[1]/2-windowShape[1]/2+windowShape[1], :] )
            t_coor_list.append(coor)
            
            dz = dz + 1
            
            if dz > DbatchSize-1:
                batch = img_list
                preds = self._net.predict(batch,False)
            
                for j in xrange(len(t_coor_list)):
                    pre_i = 0 
                    pre_neg_value = 0.0
                    pre_pos_value = 0.0
                    while pre_i<4:
                        pre_neg_value = pre_neg_value + preds[j*5+pre_i,0]*0.15
                        pre_pos_value = pre_pos_value + preds[j*5+pre_i,1]*0.15
                        pre_i = pre_i + 1
                    pre_neg_value = pre_neg_value + preds[j*5+pre_i,0]*0.4
                    pre_pos_value = pre_pos_value + preds[j*5+pre_i,1]*0.4
                        
                    if pre_neg_value > 1 - self._m_threshold: # Predict as Negative
                        if t_coor_list[j][2] == 0:
                            TN = TN + 1
                        else:
                            FN = FN + 1
                    if pre_pos_value >= self._m_threshold:
                        Positive_coords.append([t_coor_list[j][0],t_coor_list[j][1],pre_pos_value])
                        if t_coor_list[j][2] == 1:
                            TP = TP + 1
                        else:
                            FP = FP + 1
                del batch[:]
                del img_list[:]
                del t_coor_list[:]
                dz = 0
                
        if len(img_list)>0:
            batch = img_list
            preds = self._net.predict(batch,False)
            for j in xrange(len(t_coor_list)):
                pre_i = 0 
                pre_neg_value = 0.0
                pre_pos_value = 0.0
                while pre_i<4:
                    pre_neg_value = pre_neg_value + preds[j*5+pre_i,0]*0.15
                    pre_pos_value = pre_pos_value + preds[j*5+pre_i,1]*0.15
                    pre_i = pre_i + 1
                pre_neg_value = pre_neg_value + preds[j*5+pre_i,0]*0.4
                pre_pos_value = pre_pos_value + preds[j*5+pre_i,1]*0.4
                        
                if pre_neg_value > 1 - self._m_threshold: # Predict as Negative
                    if t_coor_list[j][2] == 0:
                        TN = TN + 1
                    else:
                        FN = FN + 1
                if pre_pos_value >= self._m_threshold:
                    Positive_coords.append([t_coor_list[j][0],t_coor_list[j][1],pre_pos_value])
                    if t_coor_list[j][2] == 1:
                        TP = TP + 1
                    else:
                        FP = FP + 1
            del batch[:]
            del img_list[:]
            del t_coor_list[:]
            dz = 0
        
        present_level = 5
        width, height = slide.level_dimensions[ present_level ]
        pred_Result = np.zeros((height, width, 3))
        
        csvPath = self._outputDir + "/" + dataName + ".csv"
        file_csv = open(csvPath,'w')
        if not file_csv:
            print "Cannot open the file_csv %s for writing" %csvPath
        
        file_csv.write(str(0.0000)+ ","+ str(int(0)) + "," + str(int(0))+ "\n")
        
        for i in xrange(len(Positive_coords)):
            P_X = float(Positive_coords[i][0]) 
            P_Y = float(Positive_coords[i][1]) 
        
            if Positive_coords[i][2] >= 0.9:
                pred_Result[ int(P_Y/2**present_level+0.5), int(P_X/2**present_level+0.5), 0] = pred_Result[ int(P_Y/2**present_level+0.5), int(P_X/2**present_level+0.5), 1] = pred_Result[ int(P_Y/2**present_level+0.5), int(P_X/2**present_level+0.5), 2] =  255
            if Positive_coords[i][2] < 0.9 and Positive_coords[i][2] >= 0.8:
                pred_Result[ int(P_Y/2**present_level+0.5), int(P_X/2**present_level+0.5), 2] =  255
            if Positive_coords[i][2] < 0.8 and Positive_coords[i][2] >=0.5:
                pred_Result[ int(P_Y/2**present_level+0.5), int(P_X/2**present_level+0.5), 0] =  255
            file_csv.write(str(Positive_coords[i][2])+ ","+ str(int(P_X+0.5)) + "," + str(int(P_Y+0.5))+ "\n")
        
        file_csv.close()
        
        slideFileName = slidePath.split('/')[-1]
        dataName = slideFileName.split('.tif')[0]
        cv2.imwrite( self._outputDir + "/" + dataName + "_pred.jpg", pred_Result )
    
        self._TP = self._TP + TP
        self._TN = self._TN + TN
        self._FP = self._FP + FP
        self._FN = self._FN + FN
        
        if TP + TN + FP + FN == 0:
            print "No Data to present!!"
            
        file = open(self._outputDir + "/%s_Accuracy.txt" %dataName,'w')
        if not file:
            print "Cannot open the file %s for writing" %(self._outputDir + "/Accuracy_%s.txt"%dataName)
        
        prt1 = "Accuracy: %f" %(float(TP + TN)/float(TP + TN + FP + FN))
        prt2 = "TFPN Accuracy: %f" %(float(TP)/float(TP + FP + FN))
        prt3 = "Tumour(Positive) Accuracy: %f" %(float(TP)/(TP + FN))
        prt4 = "Background(Negtive) Accuracy: %f" %(float(TN)/(TN + FP))
        
        print prt1
        print prt2
        print prt3
        print prt4
        file.write(prt1+'\n')
        file.write(prt2+'\n')
        file.write(prt3+'\n')
        file.write(prt4+'\n')
        file.close() 
        
        print "Finish Detecting %s " %slideFileName
        
if __name__ == '__main__':
    NET_FILE =           "/media/hjlin/CULab1/zHJProgram/Camelyon_VGGRes_L1/Res-152-deploy.prototxt"
    TRAINED_MODEL_FILE = "/media/hjlin/CULab1/zHJProgram/Camelyon_VGGRes_L1/ResDetecter_L0_V2/Camelyon_Res_L0.caffemodel"
    VGGcsvDir =          "/media/hjlin/CULab1/zHJProgram/Camelyon_VGGRes_L1/ResDetecter_L0_V2/VGGcsvDir"
    outputDir =          "/media/hjlin/CULab1/zHJProgram/Camelyon_VGGRes_L1/ResDetecter_L0_V2/ResResults"

    tester = CrossSlideDetecter()
    tester.SetNetDeployFile(NET_FILE)
    tester.SetTrainedModelFile(TRAINED_MODEL_FILE)
    tester.SetVGGcsvDir(VGGcsvDir)
    tester.SetOutputDir(outputDir)
    tester.ResetTFPN()
    tester.InitializeDetecter()
    
    Ex_list = [15, 18, 20, 29, 33, 44, 46, 51, 54, 55, 79, 92, 95]
    Test_list = [1, 10, 17, 24, 28, 30, 97]
    #T_list = [49 ]
    
    for i in xrange(2,41):#T_list:
        if True:#i not in Ex_list and i not in Test_list:
            slidePath = "/media/hjlin/CULab1/2016ISBI/CAMELYON16/TrainingData/Train_Tumor/Tumor_%03d.tif" %i
            maskPath =  "/media/hjlin/CULab1/2016ISBI/CAMELYON16/TrainingData/Ground_Truth/Mask/Tumor_%03d_Mask.tif"%i
            slideName = slidePath.split("/")[-1]
            print "Start Predicting: %s" %slideName
            tester.DetectWithMask(slidePath, maskPath, DbatchSize = 500)
            print "Finish Predicting: %s"%slideName
    tester.PrintAccuracy()
    print "Finish All Prediction"
    
        
        