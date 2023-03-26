import pandas as pd
from sklearn.model_selection import train_test_split
import cv2 as cv
from glob import glob
import numpy as np
class utils():
    def loadData(self,dir):
        data = []
        mismatchedData = []
        for f in glob(dir + '\\*'):
            matchedData = []
            if len(mismatchedData) == 3:
                mismatchedData = []
            isMismatchAppended = False   
            for j, im in enumerate(glob(f + '\\*')):
                if isMismatchAppended == False and (len(mismatchedData) == 0 or len(mismatchedData) == 1):
                    mismatchedData.append(im)
                    isMismatchAppended = True
                    continue
                if (j == 2 and isMismatchAppended == True) or (j ==1 and isMismatchAppended == False):
                    matchedData.append(im)
                    break
                else:
                    matchedData.append(im)
            matchedData.append(1)
            data.append(matchedData)
            if len(mismatchedData) == 2:
                mismatchedData.append(0)
                data.append(mismatchedData)
        data = pd.DataFrame(data,columns = ['image_A','image_B','match'])  
        print(data['image_A'][0])
        imagesA = data['image_A']
        imagesB = data['image_B']
        label = data['match']
        imagesA = imagesA.apply(lambda addr: self.loadAndPrerocess(addr))
        imagesB = imagesB.apply(lambda addr: self.loadAndPrerocess(addr))
        return [imagesA,imagesB,label]
        
    @staticmethod
    def loadAndPrerocess(path):
        image = cv.imread(path)
        image = cv.resize(image,(128,128))
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image = image/255.0
        image = np.expand_dims(image,axis = 0)
        return image
    @staticmethod
    def Split(images,label):
        return train_test_split(images,label,test_size=0.2)
    def visualizeData(self):
        pass
