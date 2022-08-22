import pandas as pd
from sklearn.model_selection import train_test_split
import cv2 as cv
class utils():
    def loadData(self,imagesAddr,manAddr):
        man = pd.read_csv(manAddr)
        man.apply(lambda addr: self.loadAndPreprocess('../Data'+addr))
        
    @staticmethod
    def loadAndPrerocess(path):
        image = cv.imread(path)
        image = cv.resize(image,(128,128))
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image = image/255.0
        return image

    def Split(slef):
        return train_test_split(pd.DataFrame,test_size=0.2)
        
    def visualizeData(self):
        pass