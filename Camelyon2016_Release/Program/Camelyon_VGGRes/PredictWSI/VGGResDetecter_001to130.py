'''
Created on 15 Jul 2016

@author: hjlin
'''
import sys
import os
import datetime
#from boto.sdb.db.test_db import test_list
caffe_root = '/home/hjlin/caffe-master/'
sys.path.insert(0,caffe_root+'python')
import caffe
import cv2
import numpy as np
import openslide

import itertools
from scipy import ndimage
import scipy

caffe.set_mode_gpu()
caffe.set_device(0)

class VGGResDetecter:
    def __init__(self):
        '''
        Constructor
        '''
        self._outputDir = "."
        self._cacheDir = "."
        
        self._avg_list = [[135,85,164],[155,100,135],[165,125,151],[187,155,180],[188,130,160],[145,88,122],[110,75,115],[67,28,120],[100,47,140],[100, 64, 110]]
        
        self._VGG_threshold = 0.5    #Threshold of Positive
        self._VGG_win_size = 224
        self._VGG_step_size = 112
        self._VGG_workingLevel = 1
        self._VGG_OV_level = 4
        self._VGG_OV_win_size = (self._VGG_win_size/2**(self._VGG_OV_level-self._VGG_workingLevel)) * 2**(3-self._VGG_workingLevel-1)
        self._VGG_OV_step_size = self._VGG_OV_win_size
        
        self._node_Distance = self._VGG_step_size * 2**self._VGG_workingLevel
        
        self._Res_threshold = 0.5        
        self._Res_win_size = 227
        self._Res_workingLevel = 0
        self._Res_recomputeSize = 4
        
              
    def SetOutputDir(self, outputDir):
        '''Set the Output Dir
        '''
        self._outputDir = outputDir 
        if os.path.exists(self._outputDir) == False:
            os.system("mkdir " + self._outputDir)
    
    def SetCacheDir(self, cacheDir):
        '''Set the Cache Dir
           It will speed up the re-calculation
        '''
        self._cacheDir = cacheDir 
        if os.path.exists(self._cacheDir) == False:
            os.system("mkdir " + self._cacheDir)
    
    def SetVGGThreshold(self, VGGThreshold):
        '''Set the positive Threshold of VGG Result .
        '''
        self._VGG_threshold = VGGThreshold
        
    def SetVGGWinSize(self, VGGwinSize):
        '''Set the window size of the VGG net.
           The recommend value is 224
        '''
        self._VGG_win_size = VGGwinSize    
        self._VGG_OV_win_size = (self._VGG_win_size/2**(self._VGG_OV_level-self._VGG_workingLevel)) * 2**(3-self._VGG_workingLevel-1)
        self._VGG_OV_step_size = self._VGG_OV_win_size
        
    def SetVGGOVLevel(self, VGGOVLevel):
        '''Set the overview scan level in preprocessing.
           The recommended value is 4
           This strategy aim at remove unuseful area rapidly. It will speed up the progress to a large extent
        '''
        self._VGG_OV_level = VGGOVLevel
        self._VGG_OV_win_size = (self._VGG_win_size/2**(self._VGG_OV_level-self._VGG_workingLevel)) * 2**(3-self._VGG_workingLevel-1)
        self._VGG_OV_step_size = self._VGG_OV_win_size
        
    def SetVGGWorkingLevel(self, VGGworkingLevel):
        '''Set the working level of the VGG net.
           The recommended VGG working is 1
        '''
        self._VGG_workingLevel = VGGworkingLevel        
        self._VGG_OV_win_size = (self._VGG_win_size/2**(self._VGG_OV_level-self._VGG_workingLevel)) * 2**(3-self._VGG_workingLevel-1)
        self._VGG_OV_step_size = self._VGG_OV_win_size
        self._node_Distance = self._VGG_step_size * 2**self._VGG_workingLevel
        
    def SetVGGStepSize(self, VGGstepSize):
        '''Set the step size of VGG slide-window on the working-level.
           The recommended step size is 112 (half of recommended VGG-window)
        '''
        self._VGG_step_size = VGGstepSize 
        self._node_Distance = self._VGG_step_size * 2**self._VGG_workingLevel
        
    def SetVGGNetDeployFile(self, VGGnetDeployFile):
        '''Set the path of VGG deploy file (.prototxt)
        '''
        self._VGG_NET_FILE = VGGnetDeployFile    
    
    def SetVGGTrainedModelFile(self, VGGtrainedModelFile):
        '''Set the path of trained VGG model file (.caffemodel)
        '''
        self._VGG_TRAINED_MODEL_FILE = VGGtrainedModelFile
        
    def SetResThreshold(self, ResThreshold):
        '''Set the positive Threshold of Res Result .
        '''
        self._Res_threshold = ResThreshold
    
    def SetResWinSize(self, ReswinSize):
        '''Set the window size of the Res net.
           The recommend value is 227
        '''
        self._Res_win_size = ReswinSize    
        
    def SetResWorkingLevel(self, ResworkingLevel):
        '''Set the working level of the Res-Net.
           The recommended Res working is 1
        '''
        self._Res_workingLevel = ResworkingLevel       
    
    def SetResRecomputeSize(self, ResrecomputeSize):
        '''Set the recompute-size of the small area after premary processing by Res-Net.
           The recommended Res working is 9 or 25
        '''
        self._Res_recomputeSize = ResrecomputeSize  
        
    def SetResNetDeployFile(self, ResnetDeployFile):
        '''Set the path of Res deploy file (.prototxt)
        '''
        self._Res_NET_FILE = ResnetDeployFile    
    
    def SetResTrainedModelFile(self, RestrainedModelFile):
        '''Set the path of trained Res model file (.caffemodel)
        '''
        self._Res_TRAINED_MODEL_FILE = RestrainedModelFile
    
    def InitializeVGGDetecter(self):
        '''Initialize the VGG Detecter 
           ***IMPORTANT*** 
           This function must be evoked before using VGGDetect()
        '''
        self._VGG_net = caffe.Classifier( self._VGG_NET_FILE, self._VGG_TRAINED_MODEL_FILE )     
        
    def InitializeResDetecter(self):
        '''Initialize the Res Detecter 
           ***IMPORTANT*** 
           This function must be evoked before using ResDetect()
        '''
        self._Res_net = caffe.Classifier( self._Res_NET_FILE, self._Res_TRAINED_MODEL_FILE )     
    
        
    def _GetPatch(self, TIFFImg, start_w, start_h, windowShape, workingLevel):
        '''Get Patch from WSI
           _GetPatch(self, TIFFImg, start_w, start_h, windowShape, workingLevel)
        '''
        tile = np.array(TIFFImg.read_region((start_w, start_h), workingLevel, windowShape)) 
        return tile;    
    
    def TxtMake(self, slidePath ):
        '''
           TxtMake( slidePath )
           Extract the coordinates of every interest patches, and write them into a .txt file...
        '''
        avg_list = self._avg_list
        
        slide = openslide.open_slide(slidePath)
        max_level = slide.level_count - 1
        if(self._VGG_workingLevel>max_level or self._VGG_workingLevel<0):
            print "the level to fetch data is out of the range of TIFF image"
            return 0;
        
        zero_level_size = slide.level_dimensions[0]
        
        OVwindow_W = OVwindow_H = self._VGG_OV_win_size
        OVwindowShape = (OVwindow_W, OVwindow_H)
    
        h = w = 0
        L0_win_size = self._VGG_win_size * 2**self._VGG_workingLevel
        OVstep = self._VGG_OV_step_size * 2**self._VGG_OV_level
        OV_L0_win_size = self._VGG_OV_win_size * 2**self._VGG_OV_level  
        
        slideFileName = slidePath.split('/')[-1]
        dataName = slideFileName.split('.tif')[0]
        
        coorTxtDir = self._cacheDir + "/coorTxtDir"  
        if os.path.exists(coorTxtDir) == False:
            os.system("mkdir " + coorTxtDir)
        
        if os.path.exists(coorTxtDir + "/" +dataName+"_coor.txt"):
            #read coordinates from .txt file
            print coorTxtDir + "/" +dataName+"_coor.txt" + "  Exists!!! Skip the coorinates Generation"
            return False
        else:
            #calculate the coordinate list and save to .txt
            print "Begin Overview Image Calculation..."
            m_iter = (i for i in itertools.count(0,1)) 
            t_iter = (i for i in itertools.count(0,1))
            total = (float(zero_level_size[0])/OVstep) * (float(zero_level_size[1])/OVstep)
            interval = total / 20
            h = w = 0
            OV_node_list = []
            coor_list = []
            stop_flag_1 = False
            while(w<zero_level_size[0] and stop_flag_1 == False):
                stop_flag_2 = False
                while(h<zero_level_size[1] and stop_flag_2 == False):
                    if ( w + OV_L0_win_size > zero_level_size[0] ):
                        w = zero_level_size[0] - OV_L0_win_size -1
                        stop_flag_1 = True
                    if ( h + OV_L0_win_size > zero_level_size[1] ):
                        h = zero_level_size[1] - OV_L0_win_size -1
                        stop_flag_2 = True 
            
                    #tracing the procedure...
                    m_value = m_iter.next()
                    t_value = t_iter.next()
                    if t_value > interval:
                        t_iter = (i for i in itertools.count(0,1))
                        print "    Overview complete: %.2f %%" %(m_value*100.0/total)

                    OVslideTile = self._GetPatch(slide, w, h, OVwindowShape, self._VGG_workingLevel)           
                    r,g,b,a = cv2.split(OVslideTile)
                    OVslideTile = cv2.merge([r,g,b])
                    OVslideTile = cv2.resize(OVslideTile,(self._VGG_win_size,self._VGG_win_size)).astype('float32')

                    #Average of tissue is RGB=[135,85,164] or....
                    t_thres = 10
                    t_list = []
                    t_tissue = np.zeros(b.shape)
                    for i in xrange(len(avg_list)):
                        t1 = np.zeros(b.shape)
                        t2 = np.zeros(b.shape)
                        t3 = np.zeros(g.shape)
                        t4 = np.zeros(g.shape)
                        t5 = np.zeros(r.shape)
                        t6 = np.zeros(r.shape)
                        t1[r>avg_list[i][0]-t_thres] = 1
                        t2[r<avg_list[i][0]+t_thres] = 1
                        t3[g>avg_list[i][1]-t_thres] = 1
                        t4[g<avg_list[i][1]+t_thres] = 1
                        t5[b>avg_list[i][2]-t_thres] = 1
                        t6[b<avg_list[i][2]+t_thres] = 1
                        t7 = t1+t2+t3+t4+t5+t6
                        t_tmp = np.zeros(r.shape)
                        t_tmp[t7 > 5.5] = 1
                        t_list.append(t_tmp)  

                    for i in xrange(len(t_list)):
                        t_tissue = t_tissue + t_list[i]
                    t_tissue[t_tissue>=1] = 1  

                    #Judge White Image
                    tr = np.zeros(r.shape)
                    tg = np.zeros(g.shape)
                    tb = np.zeros(b.shape)
                    tr[r>245] = 1
                    tg[g>245] = 1
                    tb[b>245] = 1
                    t_overall = tr+tg+tb
                    t_white = np.zeros(r.shape)
                    t_white[t_overall > 2.5] = 1
                    thres = b.shape[0]*b.shape[1]*(12.0/16.0)
                    
                    if r.mean() > 200 and b.mean()>180 and g.mean()>200:
                        white_flag = True
                    else:
                        white_flag = False

                    if(t_tissue.sum()>5 and white_flag == False ):
                        OV_node_list.append([w + OV_L0_win_size/2, h + OV_L0_win_size/2])
                        
                    h = h + OVstep
                h = 0
                w = w + OVstep
            print "Finish Overview Image Calculation..."

            #Extract coordinates from previous Overview Slide
            print "Generating Coordinate .txt file..."
            step = self._VGG_step_size * 2**self._VGG_workingLevel
            L0_win_size = self._VGG_win_size * 2**self._VGG_workingLevel
            window_H = window_W = self._VGG_win_size
            windowShape = (window_H, window_W)

            m_iter = (i for i in itertools.count(0,1)) 
            t_iter = (i for i in itertools.count(0,1))
            total = len(OV_node_list)
            interval = total / 20   

            for i in xrange(len(OV_node_list)):
                #tracing the procedure...
                m_value = m_iter.next()
                t_value = t_iter.next()
                if t_value > interval:
                    t_iter = (i for i in itertools.count(0,1))
                    print "    Coordinate complete: %.2f %%" %(m_value*100.0/total)
                
                OV_WCoor = OV_node_list[i][0]
                OV_HCoor = OV_node_list[i][1]
                W_Bond_Min = OV_WCoor - OV_L0_win_size/2
                W_Bond_Max = OV_WCoor + OV_L0_win_size/2
                H_Bond_Min = OV_HCoor - OV_L0_win_size/2
                H_Bond_Max = OV_HCoor + OV_L0_win_size/2

                stop_flag_1 = False
                if L0_win_size == step:
                    w = W_Bond_Min #- L0_win_size/2
                else:
                    w = W_Bond_Min - L0_win_size/2
                    if w < 0:
                        w = 0 
                while(w < W_Bond_Max and stop_flag_1 == False):
                    stop_flag_2 = False
                    if L0_win_size == step:
                        h = H_Bond_Min #- L0_win_size/2
                    else:
                        h = H_Bond_Min - L0_win_size/2
                    if h < 0:
                        h = 0                

                    while(h < H_Bond_Max and stop_flag_2 == False):
                        if ( w + L0_win_size > W_Bond_Max ):
                            w = W_Bond_Max - L0_win_size
                            stop_flag_1 = True
                        if ( h + L0_win_size > H_Bond_Max ):
                            h = H_Bond_Max - L0_win_size
                            stop_flag_2 = True 
                
                        #Remove the duplicated elements
                        if [w + L0_win_size/2,h + L0_win_size/2] in coor_list:
                            h = h + step
                            continue
                        
                        slideTile = self._GetPatch(slide, w, h, windowShape, self._VGG_workingLevel)
                        r,g,b,a = cv2.split(slideTile)
                        slideTile = cv2.merge([r,g,b])
                        slideTile = cv2.resize(slideTile,(self._VGG_win_size, self._VGG_win_size)).astype('float32')
                        
                        #Average of tissue is RGB=[135,85,164] or....
                        t_thres = 10
                        t_list = []
                        t_tissue = np.zeros(b.shape)
                        for i_t in xrange(len(avg_list)):
                            t1 = np.zeros(b.shape)
                            t2 = np.zeros(b.shape)
                            t3 = np.zeros(g.shape)
                            t4 = np.zeros(g.shape)
                            t5 = np.zeros(r.shape)
                            t6 = np.zeros(r.shape)
                            t1[r>avg_list[i_t][0]-t_thres] = 1
                            t2[r<avg_list[i_t][0]+t_thres] = 1
                            t3[g>avg_list[i_t][1]-t_thres] = 1
                            t4[g<avg_list[i_t][1]+t_thres] = 1
                            t5[b>avg_list[i_t][2]-t_thres] = 1
                            t6[b<avg_list[i_t][2]+t_thres] = 1
                            t7 = t1+t2+t3+t4+t5+t6
                            t_tmp = np.zeros(r.shape)
                            t_tmp[t7 > 5.5] = 1
                            t_list.append(t_tmp)                                   

                        for i_t in xrange(len(t_list)):
                            t_tissue = t_tissue + t_list[i_t]
                        t_tissue[t_tissue>=1] = 1              
            
                        #Judge White Image
                        tr = np.zeros(r.shape)
                        tg = np.zeros(g.shape)
                        tb = np.zeros(b.shape)
                        tr[r>245] = 1
                        tg[g>245] = 1
                        tb[b>245] = 1
                        t_overall = tr+tg+tb
                        t_white = np.zeros(r.shape)
                        t_white[t_overall > 2.5] = 1
                        thres = b.shape[0]*b.shape[1]*(15.0/16.0)
                        
                        if( t_white.sum()<thres ):
                            coor_list.append([w + L0_win_size/2, h + L0_win_size/2]) 
                            
                        h = h + step
                    w = w + step                       
                        
            coorPath = coorTxtDir + "/" + dataName + "_coor.txt"                        
            file1 = open(coorPath ,'w')             
            if not file1:
                print "Cannot open the file %s for writing" %coorPath                        
            for i in xrange(len(coor_list)):
                file1.write( "-1,"+str(coor_list[i][0]) + ", " + str(coor_list[i][1]) + "\n" )                        
        
            del OV_node_list[:]   
            del coor_list[:]     
            return True
        
    def VGGDetect(self, slidePath, DbatchSize = 800):
        '''Detect the patches by VGG net to find out the candidates
        '''
        slideFileName = slidePath.split('/')[-1]
        dataName = slideFileName.split('.tif')[0]
        
        VGGcsvDir = self._cacheDir + "/VGGcsvDir"
        VGGcsvPath = VGGcsvDir + "/" + dataName + ".csv"
        if os.path.exists(VGGcsvPath):
            print "%s Exist!!! Slip the VGG Detection"%VGGcsvPath
            return True
        
        slide = openslide.open_slide(slidePath)
        window_H = window_W = self._VGG_win_size
        windowShape = (window_H, window_W)
        
        step = self._VGG_step_size * 2**self._VGG_workingLevel
        L0_win_size = self._VGG_win_size * 2**self._VGG_workingLevel
        
        coordinate_list = []
        coorTxtDir = self._cacheDir + "/coorTxtDir"
        coorPath = coorTxtDir + "/" + dataName + "_coor.txt"
        if os.path.exists(coorPath):
            #read coordinates from .txt file
            file1 = open( coorPath,'r')
            coor_lines = file1.readlines()
            for i in xrange(len(coor_lines)):
                line = coor_lines[i]
                elems = line.rstrip().split(',')
                WCoor = int(elems[1])
                HCoor = int(elems[2]) 
                coordinate_list.append([WCoor, HCoor])
        else:
            print "Failure to find the coor file: " + coorPath
            return False      
        
        m_iter = 0
        t_iter = 0
        all_iter = len(coordinate_list)
        interval = all_iter/20.0

        dz = 0
        t_coor_list = []
        img_list = []
        Positive_coords = []
        print "Begin VGG Detection on level-%d....%d pixels to be predict: %s" %(self._VGG_workingLevel, all_iter, slideFileName)
        
        for i in xrange(all_iter):
            m_iter = m_iter + 1
            t_iter = t_iter + 1
            if t_iter > interval:
                t_iter = 0
                print "    Detecting Complete: %.2f %%" %(float(m_iter)*100/all_iter)    
            
            WCoor = coordinate_list[i][0] 
            HCoor = coordinate_list[i][1] 
            coor = [WCoor, HCoor]

            slideTile = self._GetPatch(slide, WCoor- windowShape[0]/2, HCoor- windowShape[1]/2, windowShape, self._VGG_workingLevel)
            slideTile = slideTile.astype('float32')
            r,g,b,a = cv2.split(slideTile)
            slideTile_sw = cv2.merge([r-185, g-50, b-185])
            img_list.append(slideTile_sw)
            t_coor_list.append(coor)
            dz = dz + 1
            #Predict when collect enough patches
            if dz > DbatchSize-1:
                batch = img_list
                preds = self._VGG_net.predict(batch,False)
            
                for j in xrange(preds.shape[0]):
                    if preds[j,1] >= self._VGG_threshold:
                        Positive_coords.append([t_coor_list[j][0],t_coor_list[j][1],preds[j,1]])
                del batch[:]
                del img_list[:]
                del t_coor_list[:]
                dz = 0            
            
        if len(img_list)>0:
            batch = img_list
            preds = self._VGG_net.predict(batch,False)
            for j in xrange(preds.shape[0]):
                if preds[j,1] >= self._VGG_threshold:
                    Positive_coords.append([t_coor_list[j][0],t_coor_list[j][1],preds[j,1]])
            del batch[:]
            del img_list[:]
            del t_coor_list[:]
            dz = 0                    

        present_level = 5
        width, height = slide.level_dimensions[ present_level ]
        pred_Result = np.zeros((height, width, 3))

        VGGcsvDir = self._cacheDir + "/VGGcsvDir"
        if os.path.exists(VGGcsvDir) is False:
            os.system('mkdir '+ VGGcsvDir)   
        
        VGGcsvPath = VGGcsvDir + "/" + dataName + ".csv"
        file_csv = open(VGGcsvPath, 'w')
        if not file_csv:
            print "Cannot open the file_csv %s for writing" %VGGcsvPath
        #Make sure the .csv file is not empty...
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
        
        if os.path.exists(VGGcsvDir+"/pictures") == False:
            os.system("mkdir " + VGGcsvDir +"/pictures") 
        cv2.imwrite( VGGcsvDir + "/pictures/" + dataName + "_pred.jpg", pred_Result )
        
        print "Finish VGG Detection: %s " %slideFileName
 
    def ResDetect(self, slidePath, DbatchSize = 800):  
        '''Detect the patches by VGG net to find out the candidates
        '''
        slideFileName = slidePath.split('/')[-1]
        dataName = slideFileName.split('.tif')[0]   
        
        RescsvDir = self._cacheDir + "/RescsvDir"
        RescsvPath = RescsvDir + "/" + dataName + ".csv"
        if os.path.exists(RescsvPath):
            print "%s Exist!!! Slip the Res Detection"%RescsvPath
            return True
        
        slide = openslide.open_slide(slidePath)
    
        window_H = window_W = self._Res_win_size
        windowShape = (window_H, window_W)    
    
        coordinate_list = []
        
        VGGcsvDir = self._cacheDir + "/VGGcsvDir"
        VGGcsvPath = VGGcsvDir + "/" + dataName + ".csv"
        if os.path.exists(VGGcsvPath):
            #read coordinates from .txt file
            file1 = open( VGGcsvPath,'r')
            coor_lines = file1.readlines()
            for i in xrange(1,len(coor_lines)):
                line = coor_lines[i]
                elems = line.rstrip().split(',')
                WCoor = int(elems[1])
                HCoor = int(elems[2]) 
                coordinate_list.append([WCoor, HCoor])                                     
        else:
            print "Failure to find the coor file: " + VGGcsvDir + "/" +dataName+".csv"
            return False        
        
        m_iter = 0
        t_iter = 0
        all_iter = len(coordinate_list)
        interval = all_iter/20.0
        
        dz = 0
        t_coor_list = []
        img_list = []
        Positive_coords = []
        Positive_probs = []
        print "Begin Res Detection on level-%d... %d pixels to be predict in the this step: %s" %(self._Res_workingLevel, all_iter, slideFileName)
        
        for i in xrange(all_iter):
            m_iter = m_iter + 1
            t_iter = t_iter + 1
            if t_iter > interval:
                t_iter = 0
                print "    First Detecting Complete: %.2f %%" %(float(m_iter)*100/all_iter) 
                   
            WCoor = coordinate_list[i][0] 
            HCoor = coordinate_list[i][1] 
            coor = [WCoor, HCoor]        

            slideTile = self._GetPatch(slide, WCoor- windowShape[0]/2, HCoor- windowShape[1]/2, windowShape, self._Res_workingLevel)
            slideTile = slideTile.astype('float32')
            r,g,b,a = cv2.split(slideTile)
            slideTile_sw = cv2.merge([r-185, g-50, b-185])
            img_list.append(slideTile_sw)
            t_coor_list.append(coor)
            dz = dz + 1    
            
            if dz > DbatchSize-1:
                batch = img_list
                preds = self._Res_net.predict(batch,False)
            
                for j in xrange(preds.shape[0]):
                    if preds[j,1] >= self._Res_threshold:
                        Positive_coords.append([t_coor_list[j][0],t_coor_list[j][1]])
                        Positive_probs.append(preds[j,1])
                del batch[:]
                del img_list[:]
                del t_coor_list[:]
                dz = 0
                
        if len(img_list)>0:
            batch = img_list
            preds = self._Res_net.predict(batch,False)
            for j in xrange(preds.shape[0]):
                if preds[j,1] >= self._Res_threshold:
                    Positive_coords.append([t_coor_list[j][0],t_coor_list[j][1]])
                    Positive_probs.append(preds[j,1])
            del batch[:]
            del img_list[:]
            del t_coor_list[:]
            dz = 0
        print "Finish the primary Res Detection of %s " %slideFileName

        imgM = np.zeros([1100,1100])
        probM = np.zeros([1100,1100])
        coor_minia = []    #coordinate of miniature
        prob_minia = []    #prob of miniature                           
        for i in xrange(len(Positive_coords)):
            WCoor = float(Positive_coords[i][0])
            HCoor = float(Positive_coords[i][1])
            probCoor = float(Positive_probs[i])
            wcoor_t = int(float(WCoor)/self._node_Distance+0.5)
            hcoor_t = int(float(HCoor)/self._node_Distance+0.5)
            if probCoor > self._Res_threshold:
                probM[hcoor_t, wcoor_t] = probCoor
            coor_minia.append([wcoor_t, hcoor_t])
            prob_minia.append(probCoor)
            
        imgM[probM>0.5] = 1
        labeled, nr_objects = ndimage.label(imgM)
        sizes = ndimage.sum(imgM, labeled, range(1, nr_objects+1))
        
        recompute_list = []
        for i in xrange(1,nr_objects+1):
            if sizes[i-1] < self._Res_recomputeSize:
                posRecompute = np.where(labeled == i)
                for pos_i in xrange(posRecompute[0].shape[0]):
                    WCoor = posRecompute[1][pos_i] * self._node_Distance
                    HCoor = posRecompute[0][pos_i] * self._node_Distance
                    recompute_list.append([WCoor, HCoor])  
        
        dz = 0
        EXTwindow_H = EXTwindow_W = self._Res_win_size +int(self._Res_win_size*1.0)
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

            slideTile = self._GetPatch(slide, WCoor- EXTwindowShape[0]/2, HCoor- EXTwindowShape[1]/2, EXTwindowShape, self._Res_workingLevel)
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
                preds = self._Res_net.predict(batch,False)
                for j in xrange(len(t_coor_list)):
                    pre_i = 0 
                    pre_pos_value = 0.0
                    while pre_i<4:
                        pre_pos_value = pre_pos_value + preds[j*5+pre_i,1]*0.125
                        pre_i = pre_i + 1
                    pre_pos_value = pre_pos_value + preds[j*5+pre_i,1]*0.5
                    
                    wcoor_t = int(float(t_coor_list[j][0])/self._node_Distance+0.5)
                    hcoor_t = int(float(t_coor_list[j][1])/self._node_Distance+0.5)
                    index_t = coor_minia.index( [wcoor_t, hcoor_t] )
                    
                    #Update the prediction results
                    if pre_pos_value >= self._Res_threshold:   
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
            preds = self._Res_net.predict(batch,False)
            for j in xrange(len(t_coor_list)):
                pre_i = 0 
                pre_pos_value = 0.0
                while pre_i<4:
                    pre_pos_value = pre_pos_value + preds[j*5+pre_i,1]*0.125
                    pre_i = pre_i + 1
                pre_pos_value = pre_pos_value + preds[j*5+pre_i,1]*0.5
                    
                wcoor_t = int(float(t_coor_list[j][0])/self._node_Distance+0.5)
                hcoor_t = int(float(t_coor_list[j][1])/self._node_Distance+0.5)
                index_t = coor_minia.index( [wcoor_t, hcoor_t] )
                    
                #Update the prediction results
                if pre_pos_value >= self._Res_threshold:   
                    prob_minia[index_t] = pre_pos_value
                else:
                    del coor_minia[index_t]
                    del prob_minia[index_t]
            del batch[:]
            del img_list[:]
            del t_coor_list[:]
            dz = 0
        print "Finish the Res Recomputing..."                       
 
        del Positive_coords[:]
        del Positive_probs[:]
        Positive_coords = []
        Positive_probs = []     
        
        #Transform the results from miniature back to Level 0
        slideM = np.zeros([1100,1100])
        for i_mi in xrange(len(coor_minia)):
            WCoor = coor_minia[i_mi][0] * self._node_Distance
            HCoor = coor_minia[i_mi][1] * self._node_Distance
            Positive_coords.append([WCoor, HCoor])
            Positive_probs.append(prob_minia[i_mi]) 
            slideM[coor_minia[i_mi][1], coor_minia[i_mi][0]] = 1        
        
        present_level = 5
        width, height = slide.level_dimensions[ present_level ]
        pred_Result = np.zeros((height, width, 3))   
             
        RescsvDir = self._cacheDir + "/RescsvDir"  
        if os.path.exists(RescsvDir) == False:
            os.system("mkdir " + RescsvDir)                   
        RescsvPath = RescsvDir + "/" + dataName + ".csv"
        file_csv = open(RescsvPath,'w')
        if not file_csv:
            print "Cannot open the file_csv %s for writing" %RescsvPath    
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
        
        if os.path.exists(RescsvDir+"/pictures") == False:
            os.system("mkdir " + RescsvDir +"/pictures")  
        cv2.imwrite( RescsvDir + "/pictures/" + dataName + "_pred.jpg", pred_Result )
                     
    def Detect(self, slidePath, DbatchSize = 800): 
        slideFileName = slidePath.split('/')[-1]
        dataName = slideFileName.split('.tif')[0]   
        
        self.TxtMake(slidePath)
        self.VGGDetect(slidePath, DbatchSize)
        self.ResDetect(slidePath, DbatchSize)

        
        
        
        
        
        
if __name__ == '__main__':        
        
    outputDir = "./Results"
    cacheDir =  "./cache"
    
    VGG_NET_FILE =           "./Deploys/VGG_16_deploy.prototxt"
    VGG_TRAINED_MODEL_FILE = "./Deploys/Camelyon_VGG_L1.caffemodel"   
    Res_NET_FILE =           "./Deploys/Res-152-deploy.prototxt"
    Res_TRAINED_MODEL_FILE = "./Deploys/Camelyon_Res_L0.caffemodel" 
    
    detecter = VGGResDetecter()
    detecter.SetOutputDir(outputDir)
    detecter.SetCacheDir(cacheDir)
    detecter.SetVGGWorkingLevel(1)
    detecter.SetVGGStepSize(112)
    detecter.SetVGGNetDeployFile(VGG_NET_FILE)
    detecter.SetVGGTrainedModelFile(VGG_TRAINED_MODEL_FILE)
    detecter.SetResNetDeployFile(Res_NET_FILE)
    detecter.SetResTrainedModelFile(Res_TRAINED_MODEL_FILE)
    detecter.InitializeVGGDetecter()
    detecter.InitializeResDetecter()
    
    Ex_list = []#[15, 18, 20, 29, 33, 44, 46, 51, 54, 55, 79, 92, 95]
    
    dataNum = 0
    timeSum = 0
    
    for i in xrange(1,131):
        if i not in Ex_list:
            slidePath = "../../../2016ISBI/CAMELYON16/Testset/Test_%03d.tif" %i
            slideName = slidePath.split("/")[-1]
            print "Start Training: %s" %slideName
            
            starttime = datetime.datetime.now()
            
            detecter.Detect(slidePath, DbatchSize = 800)
            
            endtime = datetime.datetime.now()
            timeUsed = (endtime-starttime).seconds
            timePath = outputDir + "/timeUsed.txt"
            file_txt = open(timePath,'a')
            if not file_txt:
                print "Cannot open the file_csv %s for writing" %timePath   
            file_txt.write("%s: %d sec"%(slideName, timeUsed) + "\n") 
            file_txt.close()
            dataNum = dataNum + 1
            timeSum = timeSum + timeUsed
            
            print "Finish Detection: %s"%slideName
            
    timePath = outputDir + "/timeUsed.txt"
    file_txt = open(timePath,'a')
    if not file_txt:
        print "Cannot open the file_csv %s for writing" %timePath   
    file_txt.write("Average Time Consumption: %.2f sec"%( float(timeSum)/dataNum ) + "\n") 
    file_txt.close()  
    print 'Finish All Task!'
    
    
    
    
    
    
    
