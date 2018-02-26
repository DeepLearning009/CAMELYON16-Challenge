'''
Created on 18 Jun 2016

@author: hjlin
'''

import sys
import os
import random
caffe_root = '/Users/daniel/caffe/'
sys.path.insert(0, caffe_root + 'python')
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
        self._win_size = 227
        self._step_size = 227
        self._workingLevel = 0
        self._VGGcsvDir = ""
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

    def SaveModel(self, modelName):
        self._solver.net.save( self._outputDir + "/" + modelName +".caffemodel" )

    def AddDataset(self, slidePath, maskPath):

        slideFileName = slidePath.split('/')[-1]
        dataName = slideFileName.split('.tif')[0]

        mask  = openslide.open_slide(maskPath)

        window_H = window_W = self._win_size
        windowShape = (window_H, window_W)

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

                if maskTile.max()<100:
                    self._neg_coor_list.append([WCoor, HCoor, 0, slidePath, maskPath])
                elif maskTile[maskTile.shape[0]/2][maskTile.shape[1]/2] > 100:
                    self._pos_coor_list.append([WCoor, HCoor, 1, slidePath, maskPath])
            print "Add %s Successfully!!"%dataName
            return True
        else:
            print "Failure to find the VGGcsv file: " + self._VGGcsvDir + "/" +dataName+".csv"
            return False

    def CleanDataset(self, slidePath, maskPath):
        del self._neg_coor_list[:]
        del self._pos_coor_list[:]

    def Train(self, DbatchSize = 60,trainStep = 60, trainTimes = 10):
        #Begin Training
        window_H = window_W = self._win_size
        windowShape = (window_H, window_W)
        EXTwindow_H = EXTwindow_W = self._win_size +int(self._win_size*1.0)
        EXTwindowShape = (EXTwindow_H, EXTwindow_W)
        diffEXT = EXTwindow_H - window_H

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
                mask  = openslide.open_slide(mskNmT[i_sld])
                for i_dsT in xrange(len(datasetT)):
                    if datasetT[i_dsT][3] == sldNmT[i_sld]:
                        WCoor = datasetT[i_dsT][0]
                        HCoor = datasetT[i_dsT][1]
                        labelCoor = datasetT[i_dsT][2]
                        slideTile = self._GetPatch(slide, WCoor- EXTwindowShape[0]/2, HCoor- EXTwindowShape[1]/2, EXTwindowShape, self._workingLevel)
                        slideTile = slideTile.astype('float32')
                        maskTile  =  self._GetPatch(mask, WCoor- EXTwindowShape[0]/2, HCoor- EXTwindowShape[1]/2, EXTwindowShape, self._workingLevel)
                        r2,g2,b2,a2 = cv2.split(maskTile)
                        maskTile = cv2.merge([r2])

                        #Randomly Crop the Tile
                        if datasetT[i_dsT][2] == 1:
                            flagInValid = True
                            while(flagInValid):
                                pos_offset1 = random.randint(0, diffEXT)
                                pos_offset2 = random.randint(0, diffEXT)
                                t_maskTile = maskTile[pos_offset1:pos_offset1+windowShape[0], pos_offset2:pos_offset2+windowShape[1]]
                                if t_maskTile[t_maskTile.shape[0]/2][t_maskTile.shape[1]/2] > 100:
                                    flagInValid = False
                            slideTile = slideTile[pos_offset1:pos_offset1+windowShape[0], pos_offset2:pos_offset2+windowShape[1],:]
                        else:
                            flagInValid = True
                            while(flagInValid):
                                pos_offset1 = random.randint(0, diffEXT)
                                pos_offset2 = random.randint(0, diffEXT)
                                t_maskTile = maskTile[pos_offset1:pos_offset1+windowShape[0], pos_offset2:pos_offset2+windowShape[1]]
                                if t_maskTile.max() < 100:
                                    flagInValid = False
                            slideTile = slideTile[pos_offset1:pos_offset1+windowShape[0], pos_offset2:pos_offset2+windowShape[1],:]

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
                        WCoor = datasetT[i_dsT][0]
                        HCoor = datasetT[i_dsT][1]
                        labelCoor = datasetT[i_dsT][2]
                        slideTile = self._GetPatch(slide, WCoor- windowShape[0]/2, HCoor- windowShape[1]/2, windowShape, self._workingLevel)
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
            print "Cannot open the file %s for writing" %(self._outputDir + "/_Accuracy.txt")
        prt0 = "Positive Number = %d, Negtive Number = %d" %(LEN_POS, LEN_NEG)
        prt1 = "Accuracy: %f" %(float(TP + TN)/float(TP + TN + FP + FN))
        prt2 = "IU Accuracy: %f" %(float(TP)/float(TP + FP + FN))
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
    solverPath = "./Deploys/ResSolver.prototxt"
    weightPath = "./ResSnapshots_L0/Camelyon_Res_L0.caffemodel"
    VGGcsvDir =  "./ResSnapshots_L0"
    outputDir =  "./ResSnapshots_L0"

    NET_FILE =   "./Deploys/Res-152-deploy.prototxt"

    trainer = CrossSlideTrainer()
    trainer.SetSolverPath(solverPath)
    trainer.SetWeightPath(weightPath)
    trainer.SetVGGcsvDir(VGGcsvDir)
    trainer.SetOutputDir(outputDir)
    trainer.InitializeTrainer()

    trainer.SetNetDeployFile(NET_FILE)

    Ex_list = [15, 18, 20, 29, 33, 44, 46, 51, 54, 55, 79, 92, 95]
    Test_list = [1, 10, 17, 24, 28, 30, 97, 110]
    train_list = [1,2,3,4,5,6,7,8,10,11,12,13,17,19,22,23,24,25,27,28,30,32,35,36,37,38,40,41,43,45,48,49,50,52,53,57,59,60,61,62,63,65,66,67,69,70,71,73,75,77,80,81,82,83,84,86,87,91,93,96,97,98,99,100,103,106,107]

    for i in xrange(1,111):
    #for i in train_list:
        if i not in Ex_list :#and i not in Test_list:
            slidePath = "../../../../DataSet/TrainingData/Train_Tumor/Tumor_%03d.tif" %i
            maskPath =  "../../../../DataSet/TrainingData/Ground_Truth/Mask/Tumor_%03d_Mask.tif"%i
            trainer.AddDataset(slidePath, maskPath)
    for i in xrange(1000):
        trainer.Train(200, 200, 3)
        modelName = "Camelyon_Res_L0"
        trainer.SaveModel( modelName )
        modelPath = outputDir + "/" + modelName +".caffemodel"
        trainer.SetTrainedModelFile(modelPath)
        trainer.InitializeTester()
        trainer.Test(200, 4)

    print 'Finish All Task!'
