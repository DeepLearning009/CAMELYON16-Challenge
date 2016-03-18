'''
Created on 17 Mar 2016

@author: hjlin
'''
import openslide
import numpy
import os
import cv2
import sys

class DataMakerBase:
    '''
    This is the base class of Every DataMaker
    '''
    def __init__(self):
        '''
        Constructor
        '''
        self._win_propotion = 24
        self._fetchingLevel = 1;
        
    def _GetPatch(self, TIFFImg, start_h, start_w, windowShape, fetchingLevel):
        tile = numpy.array(TIFFImg.read_region((start_h, start_w), fetchingLevel, windowShape)) 
        return tile;
        
    def SetWinProportion(self, winProportion):
        '''Set the size of window.
           Window size will be 1/winProportion of Img height
           Default winProportion is 24
        '''
        self._win_propotion = winProportion

    def SetFetchingLevel(self, fetchingLevel):
        '''Set the level of slide to fetch the data.
           Default winProportion is 1
        '''
        self._fetchingLevel = fetchingLevel
        
class CamelyonDataMaker(DataMakerBase):
    def __init__(self):
        '''
        Constructor
        '''
        DataMakerBase.__init__(self)
        self._patchID = 0
    
    def Apply(self, slidePath, outputDir, folderName):
        '''Greate Dataset by dividing slide into patches.
           The result will be stored into outputPath/folderName
        '''
        slide = openslide.open_slide(slidePath)
        max_level = slide.level_count - 1
        if(self._fetchingLevel>max_level or self._fetchingLevel<0):
            print "the level to fetch data is out of the range of TIFF image"
            return 0;
        
        splits = slidePath.split("/")
        tiffImgName = splits[-1]
        dataName = tiffImgName.split('.tif')[0]
        
        PathDir = outputDir + '/' +folderName
        if os.path.exists(PathDir) is False:
            os.system('mkdir '+PathDir)
        
        level_size = slide.level_dimensions[self._fetchingLevel]
        zero_level_size = slide.level_dimensions[0]
        
        window_H = window_W = int(level_size[0]/self._win_propotion)
        windowShape = (window_H, window_W)
        
        h = w = 0
        step = int(zero_level_size[0]/self._win_propotion)
        while(h<zero_level_size[0]):
            while(w<zero_level_size[1]):
                if ( h + step > zero_level_size[0] ):
                    h = zero_level_size[0] - step
                if ( w + step > zero_level_size[1] ):
                    w = zero_level_size[1] - step
                tile = self._GetPatch(slide, h, w, windowShape, self._fetchingLevel)
                PathFile = PathDir + '/' + dataName + '_' + str(self._patchID) + '.tif'
                cv2.imwrite( PathFile, tile )
                self._patchID = self._patchID + 1
                w = w + step
            w = 0
            h = h + step

class CamelyonDataMakerWithMask(DataMakerBase):
    def __init__(self):
        '''
        Constructor
        '''
        DataMakerBase.__init__(self)
        self._patchID = 0
    
    def Apply(self, slidePath, maskPath, outputDir, tumorFolderName, maskFolderName):
        '''Greate Dataset by dividing slide into patches.
           The result will be stored into outputPath/folderName
        '''
        slide = openslide.open_slide(slidePath)
        mask = openslide.open_slide(maskPath)
        max_level = mask.level_count - 1  if mask.level_count < slide.level_count else slide.level_count - 1
        if(self._fetchingLevel>max_level or self._fetchingLevel<0):
            print "the level to fetch data is out of the range of TIFF image"
            return 0;
        
        splits = slidePath.split("/")
        tiffImgName = splits[-1]
        dataName = tiffImgName.split('.tif')[0]
        
        slidePathDir = outputDir + '/' +tumorFolderName
        if os.path.exists(slidePathDir) is False:
            os.system('mkdir '+slidePathDir)
            
        maskPathDir = outputDir + '/' +maskFolderName
        if os.path.exists(maskPathDir) is False:
            os.system('mkdir '+maskPathDir)
        
        level_size = slide.level_dimensions[self._fetchingLevel]
        zero_level_size = slide.level_dimensions[0]
        
        window_H = window_W = int(level_size[0]/self._win_propotion)
        windowShape = (window_H, window_W)
        
        h = w = 0
        step = int(zero_level_size[0]/self._win_propotion)
        while(h<zero_level_size[0]):
            while(w<zero_level_size[1]):
                if ( h + step > zero_level_size[0] ):
                    h = zero_level_size[0] - step
                if ( w + step > zero_level_size[1] ):
                    w = zero_level_size[1] - step
                    
                slideTile = self._GetPatch(slide, h, w, windowShape, self._fetchingLevel)
                maskTile = self._GetPatch(mask, h, w, windowShape, self._fetchingLevel)
                
                b,g,r,a = cv2.split(slideTile)
                slideTile = cv2.merge([b,g,r])
                
                b,g,r,a = cv2.split(maskTile)
                maskTile = cv2.merge([b,g,r])
                
                if ( maskTile.max()>100 ):
                    slidePathFile = slidePathDir + '/' + dataName + '_' + str(self._patchID) + '.tif'  
                    maskPathFile = maskPathDir + '/' + dataName + '_Mask_' + str(self._patchID) + '.tif'
                    
                    cv2.imwrite( slidePathFile, slideTile )
                    cv2.imwrite( maskPathFile, maskTile )
                    self._patchID = self._patchID + 1
                
                w = w + step
            w = 0
            h = h + step


if __name__=='__main__':
    slidePath = "/media/Data2/hjlin/CAMELYON16/TrainingData/Train_Tumor/Tumor_001.tif"
    maskPath = "/media/Data2/hjlin/CAMELYON16/TrainingData/Ground_Truth/Mask/Tumor_001_Mask.tif"
    outputDir ="/media/Data2/hjlin/CAMELYON16/ForTraining"
    tumorFolderName = "Tumor_2"
    maskFolderName = "Mask_2"
    
    slidePath = sys.argv[1]
    maskPath = sys.argv[2]
    outputDir = sys.argv[3]
    tumorFolderName = sys.argv[4]
    maskFolderName = sys.argv[5]
    
    dataMaker = CamelyonDataMakerWithMask()
    dataMaker.SetFetchingLevel(2)
    dataMaker.Apply(slidePath, maskPath, outputDir, tumorFolderName, maskFolderName)
    print 'finish!'
    