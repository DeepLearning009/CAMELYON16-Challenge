'''
Created on 25 May 2016

@author: hjlin
'''
from matplotlib import pyplot as plt
from scipy import ndimage
import scipy
import cv2
import numpy as np
import random

def postProcessing(csvPath, outputDir, step_size = 224, drop_size = 4):
    file = open( csvPath,'r')
    coor_lines = file.readlines()
    imgM = np.zeros([1100,1100])
    t_tmp = np.zeros([1100,1100])
    probM = np.zeros([1100,1100])
    coor_list = []
    prob_list = []
    for i in xrange(len(coor_lines)):
        line = coor_lines[i]
        elems = line.rstrip().split(',')
        probCoor = float(elems[0])
        WCoor = float(elems[1])
        HCoor = float(elems[2]) 
        wcoor_t = int(float(WCoor)/step_size)
        hcoor_t = int(float(HCoor)/step_size)
        if probCoor>0.5:
            t_tmp[hcoor_t,wcoor_t] = 255
        coor_list.append([wcoor_t, hcoor_t])
        prob_list.append(probCoor)
        
    for i in xrange(len(coor_list)):   
        wcoor_t = coor_list[i][0]
        hcoor_t = coor_list[i][1]
        iso_flag = False
        for w_t in [-1,0,1]:
            for h_t in [-1,0,1]:
                if w_t == 0 and h_t == 0:
                    continue
                if t_tmp[hcoor_t + h_t, wcoor_t + w_t] > 100:
                    iso_flag = False
        
        if prob_list[i] > 0.5 and iso_flag == False:
            imgM[hcoor_t,wcoor_t] = 1
            probM[hcoor_t,wcoor_t] = prob_list[i]
            
    csvFileName = csvPath.split('/')[-1]
    newcsvPath = outputDir + "/" + csvFileName
    file_csv = open(newcsvPath,'w')
    if not file_csv:
        print "Cannot open the file_csv %s for writing" %newcsvPath
    file_csv.write(str(0.0000)+ ","+ str(int(0)) + "," + str(int(0))+ "\n")
    
    labeled, nr_objects = ndimage.label(imgM)
    sizes = ndimage.sum(imgM, labeled, range(1, nr_objects+1))
    t_Color = np.zeros([1100,1100,3])
    for i in xrange(1,nr_objects+1):
        if sizes[i-1] < drop_size:
            continue
        
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        t_Color[:,:,0][labeled==i] = r
        t_Color[:,:,1][labeled==i] = g
        t_Color[:,:,2][labeled==i] = b
        
        t_prob = np.array(probM)
        t_prob[labeled!=i] = 0 
        prob = t_prob.max()
        
        t_labeled = np.array(labeled)
        t_labeled[labeled!=i] = 0 
        #centroid = ndimage.center_of_mass(t_labeled)
        #P_X = centroid[1] * step_size 
        #P_Y = centroid[0] * step_size 
        posMax = np.where(t_prob == t_prob.max())
        randI = random.randint(0,posMax[0].shape[0]-1)
        P_X = posMax[1][randI] * step_size 
        P_Y = posMax[0][randI] * step_size 
        file_csv.write(str(prob)+ ","+ str(int(P_X+0.5)) + "," + str(int(P_Y+0.5))+ "\n")
    
    file_csv.close()
    dataName = csvFileName.split('.')[0]
    cv2.imwrite( outputDir + "/"+dataName+"_ColorLabel.jpg", t_Color )


if __name__ == '__main__':
    csvPath =   "/media/hjlin/HJLin_Disk/zHJProgram/Camelyon_VGG_L1/Results/Tumor_089.csv"
    outputDir = "/media/hjlin/HJLin_Disk/zHJProgram/Camelyon_VGG_L1/RefinedResults"
    Ex_list = [15, 18, 20, 29, 33, 44, 46, 51, 54, 55, 79, 92, 95]
    T_list = [2,5,8,10,12,17,19,27,28,30,35,40,44,48,50,53,57,59,65,67,69,70,80,86]
    for i in xrange(1,111):
        if i not in Ex_list :
            print "Preparing...  Tumor_%03d.csv" %i
            csvPath = "/media/hjlin/HJLin_Disk/zHJProgram/Camelyon_VGG_L1/Results/Tumor_%03d.csv" %i
            postProcessing(csvPath, outputDir, 224, 0)
    
    
    

    
    print "Finish!"
