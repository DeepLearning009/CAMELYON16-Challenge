'''
Created on 5 Jan 2017

@author: hjlin
'''
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import os
import sys


def plotROC( GTArray, PredArray, figPath):
    fpr, tpr, thresholds = metrics.roc_curve(GTArray, PredArray, pos_label=2)
    roc_auc = metrics.auc(fpr, tpr)
    
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='blue', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    ax = plt.subplot(111)
    xmajorLocator = MultipleLocator(0.2)
    xminorLocator = MultipleLocator(0.05)
    ymajorLocator = MultipleLocator(0.2)
    yminorLocator = MultipleLocator(0.05)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='minor')
    
    plt.savefig(figPath)    
    plt.show()  

if __name__ == '__main__':
    
    GTCsv_FilePath = "/media/CUDisk1/hjlin/Backup/Camelyon2016/2016ISBI/CAMELYON16/Testset/Ground_Truth/GT.csv"
    PredCsv_FilePath = "/media/CUDisk1/hjlin/Backup/Camelyon2016/Program/Camelyon_FCN/TestTrainData/FCNTestRunning/Submit3/FinalResults/Classifications_Median/ClassifyResults.csv"
    result_folder = "/media/CUDisk1/hjlin/Backup/Camelyon2016/Program/Camelyon_FCN/TestTrainData/FCNTestRunning/Submit3/FinalResults/Classifications_Median"
    figName = "Figure_ROC_AUC.png"
    
    figPath = "%s/%s" %(result_folder, figName)
    
    GT_Dict = dict()
    if os.path.exists(GTCsv_FilePath):
        #read coordinates from .txt file
        file1 = open( GTCsv_FilePath,'r')
        GT_lines = file1.readlines()
        for i in xrange(len(GT_lines)):
            line = GT_lines[i]
            elems = line.rstrip().split(',')
            dataName = elems[0]
            Label = elems[1]
            if Label == "Tumor":
                GT_Dict[dataName] = 2
            else:
                GT_Dict[dataName] = 1
    
    GT_List = []
    Pred_List = []
    if os.path.exists(PredCsv_FilePath):
        #read coordinates from .txt file
        file2 = open( PredCsv_FilePath,'r')
        Pred_lines = file2.readlines()
        for i in xrange(len(Pred_lines)):
            line = Pred_lines[i]
            elems = line.rstrip().split(',')
            dataName = elems[0]
            Prob = float(elems[1])
            GTValue = GT_Dict[dataName]
            GT_List.append(GTValue)
            Pred_List.append(Prob)
    
    GTArray = np.array(GT_List)
    PredArray = np.array(Pred_List)
    plotROC(GTArray, PredArray, figPath)
    
    print "Finish Evalusation!"
    
    
    
    
    
    
    