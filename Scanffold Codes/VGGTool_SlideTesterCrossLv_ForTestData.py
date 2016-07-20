'''
Created on 25 Apr 2016

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
        self._win_size = 224
        self._step_size = 224
        self._workingLevel = 1
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
        
    def SetOutputDir(self, outputDir):
        '''Set the size of window.
           Window size will be 1/winProportion of Img height
           Default winProportion is 24
        '''
        self._outputDir = outputDir    
        
    def TxtMake(self, slidePath):
        '''Train net
           TxtMake( slidePath, maskPath, DbatchSize, trainStep, trainTimes)
           DbatchSize is the batch size of dataset in every trainStep,
           which is diffent from the batch size of network 
            for i in trainTimes(this is overall training times):
                for i in trainStep(this is local training times over a sub-dataset):
                    train "trainStep" times over a extracted DbatchSize dataset
        '''
        
        slide = openslide.open_slide(slidePath)
        max_level = slide.level_count - 1
        if(self._workingLevel>max_level or self._workingLevel<0):
            print "the level to fetch data is out of the range of TIFF image"
            return 0;
        
        level_size = slide.level_dimensions[self._workingLevel]
        zero_level_size = slide.level_dimensions[0]
        
        OVwindow_W = OVwindow_H = self._OV_win_size
        OVwindowShape = (OVwindow_W, OVwindow_H)
        
        L0_win_size = self._win_size * 2**self._workingLevel
        OVstep = self._OV_step_size * 2**self._OV_level
        OV_L0_win_size = self._OV_win_size * 2**self._OV_level   
        
        slideFileName = slidePath.split('/')[-1]
        dataName = slideFileName.split('.tif')[0]
        
        if os.path.exists(self._outputDir + "/" +dataName+"_coor.txt"):
            #read coordinates from .txt file
            print self._outputDir + "/" +dataName+"_coor.txt Exist!"
            return 0
        else:
            #calculate the coordinate list and save to .txt
            print "Begin Overview Image Calculation..."
            m_iter = 0 
            t_iter = 0
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
                        w = zero_level_size[0] - OV_L0_win_size
                        stop_flag_1 = True
                    if ( h + OV_L0_win_size > zero_level_size[1] ):
                        h = zero_level_size[1] - OV_L0_win_size
                        stop_flag_2 = True 
                        
                    #tracing the procedure...
                    m_iter = m_iter + 1
                    t_iter = t_iter +1
                    if t_iter > interval:
                        t_iter = 0
                        print "    Overview complete: %.2f %%" %(m_iter*100.0/total)
                        
                    OVslideTile = self._GetPatch(slide, w, h, OVwindowShape, self._workingLevel)           
                    r,g,b,a = cv2.split(OVslideTile)
                    OVslideTile = cv2.merge([r,g,b])
                    OVslideTile = cv2.resize(OVslideTile,(224,224)).astype('float32')
                                   
                    
                    #Average of tissue is RGB=[135,85,164]
                    t_thres = 10
                    t_list = []
                    t_tissue = np.zeros(b.shape)
                    for i in xrange(len(self._avg_list)):
                        t1 = np.zeros(b.shape)
                        t2 = np.zeros(b.shape)
                        t3 = np.zeros(g.shape)
                        t4 = np.zeros(g.shape)
                        t5 = np.zeros(r.shape)
                        t6 = np.zeros(r.shape)
                        t1[r>self._avg_list[i][0]-t_thres] = 1
                        t2[r<self._avg_list[i][0]+t_thres] = 1
                        t3[g>self._avg_list[i][1]-t_thres] = 1
                        t4[g<self._avg_list[i][1]+t_thres] = 1
                        t5[b>self._avg_list[i][2]-t_thres] = 1
                        t6[b<self._avg_list[i][2]+t_thres] = 1
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
                    
                    #if(t_tissue.sum()>5 and t_white.sum()<thres ):
                    if(t_tissue.sum()>5 and white_flag == False ):
                        OV_node_list.append([w + OV_L0_win_size/2, h + OV_L0_win_size/2])
        
                    h = h + OVstep
                h = 0
                w = w + OVstep
            print "Finish Overview Image Calculation..."
            
####################### Show the Overview Image########################            
            width, height = slide.level_dimensions[ self._OV_level ]
            show_Result = np.zeros((height, width, 3))
            for i in xrange(len(OV_node_list)):
                OV_WCoor = OV_node_list[i][0]
                OV_HCoor = OV_node_list[i][1]
                show_Result[ int(OV_HCoor/2**self._OV_level+0.5), int(OV_WCoor/2**self._OV_level+0.5), :] = 255
           
            slideFileName = slidePath.split('/')[-1]
            dataName = slideFileName.split('.tif')[0]
            cv2.imwrite( self._outputDir + "/" + dataName + "_Overview.jpg", show_Result )  

#######################################################################            
            
            print "Generating Coordinate .txt file..."
            step = self._step_size * 2**self._workingLevel
            L0_win_size = self._win_size * 2**self._workingLevel
            window_H = window_W = self._win_size
            windowShape = (window_H, window_W)
            
            m_iter = 0 
            t_iter = 0
            total = len(OV_node_list)
            interval = total / 20   
            for i in xrange(len(OV_node_list)):
                #tracing the procedure...
                m_iter = m_iter + 1
                t_iter = t_iter +1
                if t_iter > interval:
                    t_iter = 0
                    print "  Coor complete: %.2f %%" %(m_iter*100.0/total)
                    
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
                        
                        slideTile = self._GetPatch(slide, w, h, windowShape, self._workingLevel)
                        r,g,b,a = cv2.split(slideTile)
                        slideTile = cv2.merge([r,g,b])
                        slideTile = cv2.resize(slideTile,(self._win_size,self._win_size)).astype('float32')
                                      
                        #Average of tissue is RGB=[135,85,164]
                        t_thres = 10
                        t_list = []
                        t_tissue = np.zeros(b.shape)
                        for i_t in xrange(len(self._avg_list)):
                            t1 = np.zeros(b.shape)
                            t2 = np.zeros(b.shape)
                            t3 = np.zeros(g.shape)
                            t4 = np.zeros(g.shape)
                            t5 = np.zeros(r.shape)
                            t6 = np.zeros(r.shape)
                            t1[r>self._avg_list[i_t][0]-t_thres] = 1
                            t2[r<self._avg_list[i_t][0]+t_thres] = 1
                            t3[g>self._avg_list[i_t][1]-t_thres] = 1
                            t4[g<self._avg_list[i_t][1]+t_thres] = 1
                            t5[b>self._avg_list[i_t][2]-t_thres] = 1
                            t6[b<self._avg_list[i_t][2]+t_thres] = 1
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
                        
                        #if(t_tissue.sum()>100 and t_white.sum()<thres ):
                        if( t_white.sum()<thres ):
                            coor_list.append([w + L0_win_size/2,h + L0_win_size/2])
                                
                        h = h + step
                    w = w + step
                
            coorPath = self._outputDir + "/" + dataName + "_coor.txt"
            file = open(coorPath ,'w')  
            if not file:
                print "Cannot open the file %s for writing" %coorPath
            for i in xrange(len(coor_list)):
                file.write( "-1,"+str(coor_list[i][0]) + ", " + str(coor_list[i][1]) + "\n" )                       
            
            del OV_node_list[:]   
            del coor_list[:]  
            print "Successfully Generate " + self._outputDir + "/" + dataName + "_coor.txt"
            return 1    
    
    def Detect(self, slidePath, DbatchSize = 60):
        slideFileName = slidePath.split('/')[-1]
        dataName = slideFileName.split('.tif')[0]
        
        slide = openslide.open_slide(slidePath)
        
        window_H = window_W = self._win_size
        windowShape = (window_H, window_W)
        
        step = self._step_size * 2**self._workingLevel
        L0_win_size = self._win_size * 2**self._workingLevel
        
        coordinate_list = []
        
        self.TxtMake(slidePath)
        if os.path.exists(self._outputDir + "/" +dataName+"_coor.txt"):
            #read coordinates from .txt file
            coorPath = self._outputDir + "/" + dataName + "_coor.txt"
            file = open( coorPath,'r')
            coor_lines = file.readlines()
            for i in xrange(len(coor_lines)):
                line = coor_lines[i]
                elems = line.rstrip().split(',')
                #labelCoor = int(elems[0])
                WCoor = int(elems[1])
                HCoor = int(elems[2]) 
                
                coordinate_list.append([WCoor, HCoor])

        else:
            print "Failure to find the coor file: " + self._outputDir + "/" +dataName+"_coor.txt"
            return False
        
        m_iter = 0
        t_iter = 0
        all_iter = len(coordinate_list)
        interval = all_iter/20.0
        
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
            coor = [WCoor, HCoor]
            
            slideTile = self._GetPatch(slide, WCoor, HCoor, windowShape, self._workingLevel)
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
                        Positive_coords.append([t_coor_list[j][0],t_coor_list[j][1],preds[j,1]])
                        
                del batch[:]
                del img_list[:]
                del t_coor_list[:]
                dz = 0
                
        if len(img_list)>0:
            batch = img_list
            preds = self._net.predict(batch,False)
            for j in xrange(preds.shape[0]):
                if preds[j,1] >= self._m_threshold:
                    Positive_coords.append([t_coor_list[j][0],t_coor_list[j][1],preds[j,1]])
                    
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
            P_X = float(Positive_coords[i][0]) + float(L0_win_size)/2
            P_Y = float(Positive_coords[i][1]) + float(L0_win_size)/2
        
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
    
        
        print "Finish Detecting %s " %slideFileName
        
if __name__ == '__main__':
    NET_FILE =           "../VGG_16_deploy.prototxt"
    TRAINED_MODEL_FILE = "../Camelyon_VGG_L1.caffemodel"
    outputDir =          "../Results"

    tester = CrossSlideDetecter()
    tester.SetNetDeployFile(NET_FILE)
    tester.SetTrainedModelFile(TRAINED_MODEL_FILE)
    tester.SetOutputDir(outputDir)
    tester.InitializeDetecter()
    
    Ex_list = [15, 18, 20, 29, 33, 44, 46, 51, 54, 55, 79, 92, 95]
    Test_list = []#[1, 10, 17, 24, 28, 30, 97]
    #T_list = [ 72,88,105,91,50,2,106]
    T_list = [2,5,8,10,12,17,19,27,28,30,35]#40,44,48,50,53,57,59,65,67,69,70,80,86]
    for i in xrange(1,111):
        if i not in Ex_list and i not in Test_list:
            slidePath = "../../../2016ISBI/CAMELYON16/Testset/Test_%03d.tif" %i
            slideName = slidePath.split("/")[-1]
            print "Start Predicting: %s" %slideName
            tester.Detect(slidePath, DbatchSize = 200)
            print "Finish Predicting: %s"%slideName
    print "Finish All Prediction"
    
        
        