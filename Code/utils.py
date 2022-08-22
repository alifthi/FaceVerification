import pandas as pd
from sklearn.model_selection import train_test_split
import cv2 as cv
class utils():
    def loadData(self,manAddr):
        man = pd.read_csv(manAddr)
        imagesA = man['image_A']
        imagesB = man['image_B']
        label = man['match']
        imagesA = imagesA.apply(lambda addr: self.loadAndPrerocess('../Data/'+addr))
        imagesB.apply(lambda addr: self.loadAndPrerocess('../Data/'+addr))
        return [imagesA,imagesB,label]
        
    @staticmethod
    def loadAndPrerocess(path):
        image = cv.imread(path)
        try:
            image = cv.resize(image,(128,128))
        except:
            return None
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image = image/255.0
        return image

    def Split(slef):
        return train_test_split(pd.DataFrame,test_size=0.2)
        
    def visualizeData(self):
        pass

u = utils()
a = u.loadData('/home/alifathi/Documents/AI/Git/siameseNetwork/Data/verification_dev.csv')