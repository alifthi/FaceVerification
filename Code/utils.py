import pandas as pd
from sklearn.model_selection import train_test_split
import cv2 as cv
from glob import glob
import numpy as np
class utils():
    def loadData(self,dir,loadFor = None):
        if loadFor == None:
            data = []
            dataInAFolder = []
            for f in glob(dir + '\\*'):
                for im in glob(f + '\\*'):
                    dataInAFolder.append(im)
                data.append(dataInAFolder)
                dataInAFolder = []
            data = np.array(data)
            matchedData = [list(data[i,[j,j+1]])+[1] for j in range(0,data.shape[1],2) for i in range(data.shape[0])]     
            mismatchedData = [list(data[[i,i+1],j])+[0] for i in range(0, data.shape[0],2) for j in range(data.shape[1])] 
            data = matchedData + mismatchedData
            data = pd.DataFrame(data,columns = ['image_A','image_B','match'])  
            imagesA = data['image_A']
            imagesB = data['image_B']
            label = data['match']
            imagesA = imagesA.apply(lambda addr: self.loadAndPrerocess(addr))
            imagesB = imagesB.apply(lambda addr: self.loadAndPrerocess(addr))
            return [imagesA,imagesB,label]
        elif loadFor == 'triplet':
            data = []
            for i,f in enumerate(glob(dir + '\\*')):
                for im in glob(f + '\\*'):
                    data.append([im,i])
            data = np.array(data)
            data = pd.DataFrame(data,columns = ['image','label'])  
            images = data['image']
            label = data['label']
            images = images.apply(lambda addr: self.loadAndPrerocess(addr))
            return [images,label]
    @staticmethod
    def loadAndPrerocess(path):
        image = cv.imread(path)
        image = cv.resize(image,(128,128))
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image = image/255.0
        image = np.expand_dims(image,axis = 0)
        return image
    @staticmethod
    def Split(imagesA,label,imagesB = None):
        if imagesB is not None:
            return train_test_split(imagesA,imagesB,label,test_size=0.2)
        else:
            return train_test_split(imagesA,label,test_size=0.2)
    def visualizeData(self):
        pass
