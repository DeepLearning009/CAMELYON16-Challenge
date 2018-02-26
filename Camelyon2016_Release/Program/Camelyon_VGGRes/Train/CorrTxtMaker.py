'''
Created on 12 May 2016

@author: hjlin
'''
import sys
import os
import random

import cv2
import numpy as np
import openslide

class CrossTxtMaker:
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
        avg_list = [[135,85,164],[155,100,135],[165,125,151],[187,155,180],[188,130,160],[145,88,122],[110,75,115],[67,28,120],[100,47,140],[100, 64, 110]]

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
            print self._outputDir + "/" +dataName+"_coor.txt" + "  Exists!!!"
            return False
        else:
            #calculate the coordinate list and save to .txt
            print "Begin Overview Image Calculation..."
            m_iter = 0
            t_iter = 0
            total = (float(zero_level_size[0])/OVstep) * (float(zero_level_size[1])/OVstep)
            interval = total / 20
            h = w = 0
            OV_node_list = []
            pos_coor_list = []
            neg_coor_list = []
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
                    m_iter = m_iter + 1
                    t_iter = t_iter +1
                    if t_iter > interval:
                        t_iter = 0
                        print "    Overview complete: %.2f %%" %(m_iter*100.0/total)

                    OVslideTile = self._GetPatch(slide, w, h, OVwindowShape, self._OV_level)
                    r,g,b,a = cv2.split(OVslideTile)
                    OVslideTile = cv2.merge([r,g,b])
                    OVslideTile = cv2.resize(OVslideTile,(224,224)).astype('float32')


                    #Average of tissue is RGB=[135,85,164]
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

                        #Remove the duplicated elements
                        if [w + L0_win_size/2,h + L0_win_size/2] in pos_coor_list or [w + L0_win_size/2,h + L0_win_size/2] in neg_coor_list:
                            h = h + step
                            continue

                        slideTile = self._GetPatch(slide, w, h, windowShape, self._workingLevel)
                        r,g,b,a = cv2.split(slideTile)
                        slideTile = cv2.merge([r,g,b])
                        slideTile = cv2.resize(slideTile,(224,224)).astype('float32')

                        maskTile = self._GetPatch(mask, w, h, windowShape, self._workingLevel)
                        r2,g2,b2,a2 = cv2.split(maskTile)
                        maskTile = cv2.merge([r2])

                        #Average of tissue is RGB=[135,85,164]
                        avg_list
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

                        #if(t_tissue.sum()>100 and t_white.sum()<thres ):
                        if( t_white.sum()<thres ):
                            if ( maskTile.max()>100 ):
                                mask_tmp = np.zeros(maskTile.shape)
                                mask_tmp[maskTile > 100] = 1
                                pixelNum = maskTile.shape[0] * maskTile.shape[1]
                                if maskTile[maskTile.shape[0]/2][maskTile.shape[1]/2] > 100: #and mask_tmp.sum() > 0.1 * pixelNum: #only Tumor in centural is considered
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
            del OV_node_list[:]
            del neg_coor_list[:]
            del pos_coor_list[:]

if __name__ == '__main__':
    outputDir = "./VGGSnapshots_L1"

    txtMaker = CrossTxtMaker()
    txtMaker.SetWorkingLevel(1)
    txtMaker.SetStepSize(112)
    txtMaker.SetOutputDir(outputDir)

    Ex_list = []
    for i in xrange(1,21):
        if i not in Ex_list:
            slidePath = "../../../../DataSet/TrainingData/Train_Tumor/Tumor_%03d.tif" %i
            maskPath =  "../../../../DataSet/TrainingData/Ground_Truth/Mask/Tumor_%03d_Mask.tif"%i
            slideName = slidePath.split("/")[-1]
            print "Start Training: %s" %slideName
            txtMaker.TxtMake(slidePath, maskPath )
            print "Finish Training: %s"%slideName

    print 'Finish All Task!'
