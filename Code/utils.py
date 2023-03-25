import pandas as pd
from sklearn.model_selection import train_test_split
import cv2 as cv
from glob import glob
class utils():
    def loadData(self,dir):
        Data = pd.DataFrame(columns = ['image_A','image_B','match'])
        data = []
        mismatchedData = []
        for f,i in enumerate(glob(dir)):
            matchedData = []
            isMismatchAppended = False   
            for im,j in enumerate(glob(f)):
                if len(mismatchedData) == 0 or len(mismatchedData) == 1:
                    mismatchedData.append(im)
                    isMismatchAppended = True
                if not isMismatchAppended:
                    matchedData.append(im)
                if j == 2:
                    break
            matchedData.append(1)
            data.append(matchedData)
            if len(mismatchedData) == 2:
                mismatchedData.append(0)
                data.append(mismatchedData)
        Data[i] = mismatchedData
                
            
        imagesA = Data['image_A']
        imagesB = Data['image_B']
        label = Data['match']
        imagesA = imagesA.apply(lambda addr: self.loadAndPrerocess('../Data/'+addr))
        imagesB = imagesB.apply(lambda addr: self.loadAndPrerocess('../Data/'+addr))
        return [imagesA,imagesB,label]
        
    @staticmethod
    def loadAndPrerocess(path):
        image = cv.imread(path)
        image = cv.resize(image,(64,64))
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image = image/255.0
        return image
    @staticmethod
    def Split(images,label):
        return train_test_split(images,label,test_size=0.2)
    def visualizeData(self):
        pass
