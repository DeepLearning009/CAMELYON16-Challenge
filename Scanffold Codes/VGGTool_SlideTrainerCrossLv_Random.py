'''
Created on 12 May 2016

@author: hjlin
'''
import sys
import os
import random
caffe_root = '/home/hjlin/caffe-master/'
sys.path.insert(0,caffe_root+'python')
import caffe
import cv2
import numpy as np
import openslide

caffe.set_mode_gpu()
caffe.set_device(0)

class CrossSlideTrainer:
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
        
        self._OV_node_list = []
        self._pos_coor_list = []
        self._neg_coor_list = []
    
    def InitializeTrainer(self):
        self._solver = caffe.SGDSolver(self._solverPath)
        self._solver.net.copy_from(self._weightPath)  
    
    def InitializeTester(self):
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
        
    def SetSolverPath(self, solverPath):
        '''Set the size of window.
           Window size will be 1/winProportion of Img height
           Default winProportion is 24
        '''
        self._solverPath = solverPath
        
    def SetWeightPath(self, weightPath):
        '''Set the size of window.
           Window size will be 1/winProportion of Img height
           Default winProportion is 24
        '''
        self._weightPath = weightPath
        
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
        
    def SaveModel(self, modelName):
        self._solver.net.save( self._outputDir + "/" + modelName +".caffemodel" )
        
    def TxtMake(self, slidePath, maskPath ):
        '''Train net
           TxtMake( slidePath, maskPath, DbatchSize, trainStep, trainTimes)
           DbatchSize is the batch size of dataset in every trainStep,
           which is diffent from the batch size of network 
            for i in trainTimes(this is overall training times):
                for i in trainStep(this is local training times over a sub-dataset):
                    train "trainStep" times over a extracted DbatchSize dataset
        '''
        
        slide = openslide.open_slide(slidePath)
        mask = openslide.open_slide(maskPath)
        max_level = slide.level_count - 1
        if(self._workingLevel>max_level or self._workingLevel<0):
            print "the level to fetch data is out of the range of TIFF image"
            return 0;
        
        level_size = slide.level_dimensions[self._workingLevel]
        zero_level_size = slide.level_dimensions[0]
        
        OVwindow_W = OVwindow_H = self._OV_win_size
        OVwindowShape = (OVwindow_W, OVwindow_H)
        
        h = w = 0
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
            
            OV_node_list = []
            pos_coor_list = []
            neg_coor_list = []
            
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
            for i in xrange(len(self._OV_node_list)):
                OV_WCoor = self._OV_node_list[i][0]
                OV_HCoor = self._OV_node_list[i][1]
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
                        slideTile = cv2.resize(slideTile,(224,224)).astype('float32')
                        
                        maskTile = self._GetPatch(mask, w, h, windowShape, self._workingLevel)
                        r2,g2,b2,a2 = cv2.split(maskTile)
                        maskTile = cv2.merge([r2])  
                        
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
                        
                        if(t_tissue.sum()>100 and t_white.sum()<thres ):
                            if ( maskTile.max()>100 ):
                                mask_tmp = np.zeros(maskTile.shape)
                                mask_tmp[maskTile > 100] = 1
                                pixelNum = maskTile.shape[0] * maskTile.shape[1]
                                if maskTile[maskTile.shape[0]/2][maskTile.shape[1]/2] > 100 and mask_tmp.sum() > 0.9 * pixelNum: #only Tumor in centural is considered
                                    pos_coor_list.append([w + L0_win_size/2, h + L0_win_size/2]) 
                            else:
                                neg_coor_list.append([w + L0_win_size/2,h + L0_win_size/2])
                                
                        h = h + step
                    w = w + step
                
            coorPath = self._outputDir + "/" + dataName + "_coor.txt"
            file = open(coorPath ,'w')  
            if not file:
                print "Cannot open the file %s for writing" %coorPath
            for i in xrange(len(pos_coor_list)):
                file.write( "1,"+str(pos_coor_list[i][0]) + ", " + str(pos_coor_list[i][1]) + "\n" )
            for i in xrange(len(neg_coor_list)):
                file.write( "0,"+str(neg_coor_list[i][0]) + ", " + str(neg_coor_list[i][1]) + "\n" )   
            print "Successfully Generate " + self._outputDir + "/" + dataName + "_coor.txt"
            return 1
    
    def AddDataset(self, slidePath, maskPath):
        
        slideFileName = slidePath.split('/')[-1]
        dataName = slideFileName.split('.tif')[0]
        
        self.TxtMake(slidePath, maskPath)
        if os.path.exists(self._outputDir + "/" +dataName+"_coor.txt"):
            #read coordinates from .txt file
            coorPath = self._outputDir + "/" + dataName + "_coor.txt"
            file = open( coorPath,'r')
            coor_lines = file.readlines()
            for i in xrange(len(coor_lines)):
                line = coor_lines[i]
                elems = line.rstrip().split(',')
                labelCoor = int(elems[0])
                WCoor = int(elems[1])
                HCoor = int(elems[2]) 
                
                if labelCoor == 0:
                    self._neg_coor_list.append([WCoor, HCoor, 0, slidePath, maskPath])
                elif labelCoor == 1:
                    self._pos_coor_list.append([WCoor, HCoor, 1, slidePath, maskPath])
            return True
        else:
            print "Failure to find the coor file: " + self._outputDir + "/" +dataName+"_coor.txt"
            return False            
            
    def CleanDataset(self, slidePath, maskPath):     
        del self._neg_coor_list[:]    
        del self._pos_coor_list[:]           
    
    def Train(self, DbatchSize = 60,trainStep = 60, trainTimes = 10):
        #Begin Training
        window_H = window_W = self._win_size
        windowShape = (window_H, window_W)
        prob_P = 0.05
        LEN_POS = len(self._pos_coor_list)
        LEN_NEG = len(self._neg_coor_list)
        print "Positive Number = %d, Negtive Number = %d" %(LEN_POS, LEN_NEG)
        print "Positive Sampling Probability = %f"%(prob_P)
        for train_time in xrange(trainTimes):
            datasetT = []
            sldNmT = []
            mskNmT = []
            img_list = []
            label_list = []
            for dz in xrange(DbatchSize):
                if random.uniform(0.0, 1.0) < prob_P:
                    randN = random.randint(0, LEN_POS-1)
                    datasetT.append(self._pos_coor_list[randN])
                    if self._pos_coor_list[randN][3] not in sldNmT:
                        sldNmT.append(self._pos_coor_list[randN][3])
                        mskNmT.append(self._pos_coor_list[randN][4])
                else:
                    randN = random.randint(0, LEN_NEG-1)
                    datasetT.append(self._neg_coor_list[randN])
                    if self._neg_coor_list[randN][3] not in sldNmT:
                        sldNmT.append(self._neg_coor_list[randN][3])
                        mskNmT.append(self._neg_coor_list[randN][4])
            for i_sld in xrange(len(sldNmT)):
                slide = openslide.open_slide(sldNmT[i_sld])
                for i_dsT in xrange(len(datasetT)):
                    if datasetT[i_dsT][3] == sldNmT[i_sld]:
                        WCoor = datasetT[i_dsT][0] - windowShape[0]/2
                        HCoor = datasetT[i_dsT][1] - windowShape[1]/2
                        labelCoor = datasetT[i_dsT][2]
                        slideTile = self._GetPatch(slide, WCoor, HCoor, windowShape, self._workingLevel)
                        slideTile = slideTile.astype('float32')
                        r,g,b,a = cv2.split(slideTile)
                        slideTile_sw = np.array([r-185, g-50, b-185])
                        pos = random.randint(0,len(img_list))
                        img_list.insert(pos, slideTile_sw)
                        label_list.insert(pos, labelCoor)
                        
            data = np.array(img_list).astype('float32')
            labels = np.array(label_list).astype('float32')
            self._solver.net.set_input_arrays(data, labels)
            self._solver.step(trainStep)
            del data
            del labels
            del img_list[:]
            del label_list[:]
            del sldNmT[:]
            del mskNmT[:]
            
    def Test(self, DbatchSize = 100, testTimes = 2):
        #Begin Training
        TP = TN = FP = FN =0
        m_threshold = 0.5
        window_H = window_W = self._win_size
        windowShape = (window_H, window_W)
        LEN_POS = len(self._pos_coor_list)
        LEN_NEG = len(self._neg_coor_list)
        for test_time in xrange(testTimes):
            datasetT = []
            sldNmT = []
            mskNmT = []
            img_list = []
            label_list = []
            
            #Test Positive and Negtive
            for dz in xrange(DbatchSize):
                randN = random.randint(0, LEN_POS-1)
                datasetT.append(self._pos_coor_list[randN])
                if self._pos_coor_list[randN][3] not in sldNmT:
                    sldNmT.append(self._pos_coor_list[randN][3])
                    mskNmT.append(self._pos_coor_list[randN][4])
            for dz in xrange(DbatchSize):
                randN = random.randint(0, LEN_NEG-1)
                datasetT.append(self._neg_coor_list[randN])
                if self._neg_coor_list[randN][3] not in sldNmT:
                    sldNmT.append(self._neg_coor_list[randN][3])
                    mskNmT.append(self._neg_coor_list[randN][4])
            for i_sld in xrange(len(sldNmT)):
                slide = openslide.open_slide(sldNmT[i_sld])
                for i_dsT in xrange(len(datasetT)):
                    if datasetT[i_dsT][3] == sldNmT[i_sld]:
                        WCoor = datasetT[i_dsT][0] - windowShape[0]/2
                        HCoor = datasetT[i_dsT][1] - windowShape[1]/2
                        labelCoor = datasetT[i_dsT][2]
                        slideTile = self._GetPatch(slide, WCoor, HCoor, windowShape, self._workingLevel)
                        slideTile = slideTile.astype('float32')
                        r,g,b,a = cv2.split(slideTile)
                        slideTile_sw = cv2.merge([r-185, g-50, b-185])
                        pos = random.randint(0,len(img_list))
                        img_list.insert(pos, slideTile_sw)
                        label_list.insert(pos, labelCoor)
            batch = img_list
            preds = self._net.predict(batch,False)
            for j in xrange(preds.shape[0]):
                if preds[j,0] > 1 - m_threshold: # Predict as Negative
                    if label_list[j] == 0:
                        TN = TN + 1
                    else:
                        FN = FN + 1
                if preds[j,1] >= m_threshold:
                    if label_list[j] == 1:
                        TP = TP + 1
                    else:
                        FP = FP + 1 
            del batch[:]
            del img_list[:]
            del label_list[:]

        file = open(self._outputDir + "/_Accuracy.txt", 'a')
        if not file:
            filename = self._outputDir + "/_Accuracy.txt" 
            print "Cannot open the file %s for writing" %filename
        prt0 = "Positive Number = %d, Negtive Number = %d" %(LEN_POS, LEN_NEG)
        prt1 = "Accuracy: %f" %(float(TP + TN)/float(TP + TN + FP + FN))
        prt2 = "TFPN Accuracy: %f" %(float(TP)/float(TP + FP + FN))
        prt3 = "Tumour(Positive) Accuracy: %f" %(float(TP)/(TP + FN))
        prt4 = "Background(Negtive) Accuracy: %f" %(float(TN)/(TN + FP))
        print prt0
        print prt1
        print prt2
        print prt3
        print prt4
        file.write(prt0+'\n')
        file.write(prt1+'\n')
        file.write(prt2+'\n')
        file.write(prt3+'\n')
        file.write(prt4+'\n\n')
        file.close() 
            

if __name__ == '__main__':
    solverPath = "/media/hjlin/HJLin_Disk/zHJProgram/Camelyon_VGG_L1/solver.prototxt"
    weightPath = "/media/hjlin/HJLin_Disk/zHJProgram/Camelyon_VGG_L1/Camelyon_VGG_L1.caffemodel"
    outputDir =  "/media/hjlin/HJLin_Disk/zHJProgram/Camelyon_VGG_L1/Snapshots"
    
    NET_FILE =   "/media/hjlin/HJLin_Disk/zHJProgram/Camelyon_VGG_L1/VGG_16_deploy.prototxt"
    
    trainer = CrossSlideTrainer()
    trainer.SetSolverPath(solverPath)
    trainer.SetWeightPath(weightPath)
    trainer.SetOutputDir(outputDir)
    trainer.InitializeTrainer()
    
    trainer.SetNetDeployFile(NET_FILE)
    
    Ex_list = [15, 18, 20, 29, 33, 44, 46, 51, 54, 55, 79, 92, 95]
    Test_list = [1, 10, 17, 24, 28, 30, 97, 110]
    #train_list = [1,2,3,4,5,6,7,11,12,13,14,15,16,17,21,22,23,24,25,26,27,31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,51,52,53,54,55,56,57,58,61,62,63,64,65,66,67,68,71,72,73,74,75,76,77,78,79,81,82,83,84,85,86,87,88,89,91,92,93,94,95,96,97,98,101,102,103,104,105,106]
    #train_list = [72]

    for i in xrange(1,111):
    #for i in train_list:
        if i not in Ex_list :#and i not in Test_list:
            slidePath = "/media/hjlin/HJLin_Disk/2016ISBI/CAMELYON16/TrainingData/Train_Tumor/Tumor_%03d.tif" %i
            maskPath =  "/media/hjlin/HJLin_Disk/2016ISBI/CAMELYON16/TrainingData/Ground_Truth/Mask/Tumor_%03d_Mask.tif"%i
            trainer.AddDataset(slidePath, maskPath)
    for i in xrange(1000):
        trainer.Train(200, 200, 3)
        modelName = "Camelyon_VGG_L1"
        trainer.SaveModel( modelName )
        modelPath = outputDir + "/" + modelName +".caffemodel"
        trainer.SetTrainedModelFile(modelPath)
        trainer.InitializeTester()
        trainer.Test(200, 2)
            
    print 'Finish All Task!'
    
