from siamese import Model
from utils import utils
import numpy as np
model = Model()
util = utils()
imageA,imageB,label = util.loadData(r'C:\Users\alifa\Documents\AI\DATA\Robtech\HW4\HW4-dataset\dataset\classification\train1000')
imageA = np.concatenate(imageA,axis = 0)
imageB = np.concatenate(imageB,axis = 0)
imageATrain, imageBTrain, labelTrain, imageATest, imageBTest, labelTest = util.Split(imagesA=imageA,imagesB=imageB,label=label)
model.compileModel()
model.train([imageA,imageB],np.array(label),valData=[[imageATest,imageBTest],labelTest])
