'''
Created on 22 Jun 2016

@author: hjlin
'''
import sys
import os
from scipy import ndimage
import scipy
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
        self._step_size = 224
        self._workingLevel = 0
        self._VGGcsvDir = ""
        self._outputDir = ""
        self._OV_level = 4
        self._OV_win_size = (self._win_size/2**(self._OV_level-self._workingLevel)) * 2**(3-self._workingLevel-1)
        self._OV_step_size = self._OV_win_size
        self._avg_list = [[135,85,164],[155,100,135],[165,125,151],[187,155,180],[188,130,160],[145,88,122],[110,75,115],[67,28,120],[100,47,140],[100, 64, 110]]
        
        self._TP = self._TN = self._FP = self._FN =0
        
        self._m_threshold = 0.5
        self._recomputeSize = 9
    
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
        
        
        step = self._step_size * 2**self._workingLevel
        L0_win_size = self._win_size * 2**self._workingLevel
        
        coordinate_list = []
        maskM = np.zeros([1100,1100])
        
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
                    wcoor_t = int(float(WCoor)/self._step_size+0.5)
                    hcoor_t = int(float(HCoor)/self._step_size+0.5)
                    maskM[hcoor_t, wcoor_t] = 3
                else:
                    coordinate_list.append([WCoor, HCoor, 0])
                    wcoor_t = int(float(WCoor)/self._step_size+0.5)
                    hcoor_t = int(float(HCoor)/self._step_size+0.5)
                    maskM[hcoor_t, wcoor_t] = -3
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
        Positive_probs = []
        print "Begin Testing... %d pixels to be predict in the first step" %all_iter
        
        for i in xrange(all_iter):
            m_iter = m_iter + 1
            t_iter = t_iter + 1
            if t_iter > interval:
                t_iter = 0
                print "    First Detecting Complete: %.2f %%" %(float(m_iter)*100/all_iter)    
            
            WCoor = coordinate_list[i][0] 
            HCoor = coordinate_list[i][1] 
            labelCoor = coordinate_list[i][2]
            coor = [WCoor, HCoor, labelCoor]
            
            slideTile = self._GetPatch(slide, WCoor- windowShape[0]/2, HCoor- windowShape[1]/2, windowShape, self._workingLevel)
            slideTile = slideTile.astype('float32')
            r,g,b,a = cv2.split(slideTile)
            slideTile_sw = cv2.merge([r-185, g-50, b-185])
            img_list.append(slideTile_sw)
            t_coor_list.append(coor)
            dz = dz + 1
            
            if dz > DbatchSize-1:
                batch = img_list
                preds = self._net.predict(batch,False)
            
                for j in xrange(preds.shape[0]):
                    if preds[j,1] >= self._m_threshold:
                        Positive_coords.append([t_coor_list[j][0],t_coor_list[j][1]])
                        Positive_probs.append(preds[j,1])
                del batch[:]
                del img_list[:]
                del t_coor_list[:]
                dz = 0
                
        if len(img_list)>0:
            batch = img_list
            preds = self._net.predict(batch,False)
            for j in xrange(preds.shape[0]):
                if preds[j,1] >= self._m_threshold:
                    Positive_coords.append([t_coor_list[j][0],t_coor_list[j][1]])
                    Positive_probs.append(preds[j,1])
            del batch[:]
            del img_list[:]
            del t_coor_list[:]
            dz = 0
        print "Finish the First Step Detection"
        
        
        imgM = np.zeros([1100,1100])
        probM = np.zeros([1100,1100])
        coor_minia = []    #coordinate of miniature
        prob_minia = []    #prob of miniature
        for i in xrange(len(Positive_coords)):
            WCoor = float(Positive_coords[i][0])
            HCoor = float(Positive_coords[i][1])
            probCoor = float(Positive_probs[i])
            wcoor_t = int(float(WCoor)/self._step_size+0.5)
            hcoor_t = int(float(HCoor)/self._step_size+0.5)
            if probCoor > self._m_threshold:
                probM[hcoor_t, wcoor_t] = probCoor
            coor_minia.append([wcoor_t, hcoor_t])
            prob_minia.append(probCoor)
            
        imgM[probM>0.5] = 1
        labeled, nr_objects = ndimage.label(imgM)
        sizes = ndimage.sum(imgM, labeled, range(1, nr_objects+1))
        
        recompute_list = []
        for i in xrange(1,nr_objects+1):
            if sizes[i-1] < self._recomputeSize:
                posRecompute = np.where(labeled == i)
                for pos_i in xrange(posRecompute[0].shape[0]):
                    WCoor = posRecompute[1][pos_i] * self._step_size
                    HCoor = posRecompute[0][pos_i] * self._step_size
                    recompute_list.append([WCoor, HCoor])
        
        dz = 0
        EXTwindow_H = EXTwindow_W = self._win_size +int(self._win_size*1.0)
        EXTwindowShape = (EXTwindow_H, EXTwindow_W)
        t_coor_list = []
        
        m_iter = 0
        t_iter = 0
        all_iter = len(recompute_list)
        interval = all_iter/20.0
        print "Recomputing... %d pixels to be recomputed in the second step" %all_iter
        
        for i in xrange(all_iter):
            m_iter = m_iter + 1
            t_iter = t_iter + 1
            if t_iter > interval:
                t_iter = 0
                print "    Detecting Complete: %.2f %%" %(float(m_iter)*100/all_iter)
            
            WCoor = recompute_list[i][0] 
            HCoor = recompute_list[i][1] 
            coor = [WCoor, HCoor]
            
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
                    pre_pos_value = 0.0
                    while pre_i<4:
                        pre_pos_value = pre_pos_value + preds[j*5+pre_i,1]*0.125
                        pre_i = pre_i + 1
                    pre_pos_value = pre_pos_value + preds[j*5+pre_i,1]*0.5
                    
                    wcoor_t = int(float(t_coor_list[j][0])/self._step_size+0.5)
                    hcoor_t = int(float(t_coor_list[j][1])/self._step_size+0.5)
                    index_t = coor_minia.index( [wcoor_t, hcoor_t] )
                    
                    #Update the prediction results
                    if pre_pos_value >= self._m_threshold:   
                        prob_minia[index_t] = pre_pos_value
                    else:
                        del coor_minia[index_t]
                        del prob_minia[index_t]
                del batch[:]
                del img_list[:]
                del t_coor_list[:]
                dz = 0
        
        if len(img_list)>0:
            batch = img_list
            preds = self._net.predict(batch,False)
            for j in xrange(len(t_coor_list)):
                pre_i = 0 
                pre_pos_value = 0.0
                while pre_i<4:
                    pre_pos_value = pre_pos_value + preds[j*5+pre_i,1]*0.125
                    pre_i = pre_i + 1
                pre_pos_value = pre_pos_value + preds[j*5+pre_i,1]*0.5
                    
                wcoor_t = int(float(t_coor_list[j][0])/self._step_size+0.5)
                hcoor_t = int(float(t_coor_list[j][1])/self._step_size+0.5)
                index_t = coor_minia.index( [wcoor_t, hcoor_t] )
                    
                #Update the prediction results
                if pre_pos_value >= self._m_threshold:   
                    prob_minia[index_t] = pre_pos_value
                else:
                    del coor_minia[index_t]
                    del prob_minia[index_t]
            del batch[:]
            del img_list[:]
            del t_coor_list[:]
            dz = 0
        print "Finish the Second Step Recomputing"       
                
        del Positive_coords[:]
        del Positive_probs[:]
        Positive_coords = []
        Positive_probs = []    
        
        slideM = np.zeros([1100,1100])
        for i_mi in xrange(len(coor_minia)):
            WCoor = coor_minia[i_mi][0] * self._step_size
            HCoor = coor_minia[i_mi][1] * self._step_size
            Positive_coords.append([WCoor, HCoor])
            Positive_probs.append(prob_minia[i_mi]) 
            slideM[coor_minia[i_mi][1], coor_minia[i_mi][0]] = 1
        
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
        
            if Positive_probs[i] >= 0.9:
                pred_Result[ int(P_Y/2**present_level+0.5), int(P_X/2**present_level+0.5), 0] = pred_Result[ int(P_Y/2**present_level+0.5), int(P_X/2**present_level+0.5), 1] = pred_Result[ int(P_Y/2**present_level+0.5), int(P_X/2**present_level+0.5), 2] =  255
            if Positive_probs[i] < 0.9 and Positive_probs[i] >= 0.8:
                pred_Result[ int(P_Y/2**present_level+0.5), int(P_X/2**present_level+0.5), 2] =  255
            if Positive_probs[i] < 0.8 and Positive_probs[i] >=0.5:
                pred_Result[ int(P_Y/2**present_level+0.5), int(P_X/2**present_level+0.5), 0] =  255
            file_csv.write(str(Positive_probs[i])+ ","+ str(int(P_X+0.5)) + "," + str(int(P_Y+0.5))+ "\n")
        
        file_csv.close()
        
        slideFileName = slidePath.split('/')[-1]
        dataName = slideFileName.split('.tif')[0]
        cv2.imwrite( self._outputDir + "/" + dataName + "_pred.jpg", pred_Result )
        
        #TP calculation
        tMat1 = np.zeros(slideM.shape)
        tMat1[(slideM + maskM)!=4] = 0
        tMat1[(slideM + maskM)==4] = 1
        TP = tMat1.sum()
        #FP calculation
        tMat2 = np.zeros(slideM.shape)
        tMat2[(slideM+maskM)!=-2] = 0
        tMat2[(slideM+maskM)==-2] = 1
        FP = tMat2.sum()
        #FN calculation
        tMat3 = np.zeros(slideM.shape)
        tMat3[(maskM+slideM)!=3] = 0
        tMat3[(maskM+slideM)==3] = 1
        FN = tMat3.sum()
        #TN calculation
        tMat4 = np.zeros(slideM.shape)
        tMat4[(maskM+slideM)!=-3] = 0
        tMat4[(maskM+slideM)==-3] = 1
        TN = tMat4.sum()
        
    
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
    NET_FILE =           "/home/hjlin/Desktop/Working/Res-152-deploy.prototxt"
    TRAINED_MODEL_FILE = "/home/hjlin/Desktop/Working/Camelyon_Res_L0.caffemodel"
    VGGcsvDir =          "/home/hjlin/Desktop/Working/VGGcsvDir"
    outputDir =          "/home/hjlin/Desktop/Working/ResResults"

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
    
    for i in xrange(1,11):#T_list:
        if True:#i not in Ex_list and i not in Test_list:
            slidePath = "/media/hjlin/CULab1/2016ISBI/CAMELYON16/TrainingData/Train_Tumor/Tumor_%03d.tif" %i
            maskPath =  "/media/hjlin/CULab1/2016ISBI/CAMELYON16/TrainingData/Ground_Truth/Mask/Tumor_%03d_Mask.tif"%i
            slideName = slidePath.split("/")[-1]
            print "Start Predicting: %s" %slideName
            tester.DetectWithMask(slidePath, maskPath, DbatchSize = 500)
            print "Finish Predicting: %s"%slideName
    #tester.PrintAccuracy()
    print "Finish All Prediction"
    
        
        