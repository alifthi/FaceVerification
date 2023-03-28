from siamese import Model
from utils import utils
import numpy as np
model = Model()
model.compileModel()
util = utils()
mode = 'triplet'
if mode == None:
    imageA,imageB,label = util.loadData(r'C:\Users\alifa\Documents\AI\DATA\Robtech\HW4\HW4-dataset\dataset\classification\train1000')
    imageA = np.concatenate(imageA,axis = 0)
    imageB = np.concatenate(imageB,axis = 0)
    imageATrain, imageATest, imageBTrain, imageBTest, labelTrain, labelTest = util.Split(imagesA=imageA,imagesB=imageB,label=label)
    model.train([imageATrain,imageBTrain],np.array(labelTrain),valData=[[imageATest,imageBTest],labelTest])
elif mode == 'triplet':
    images , label = util.loadData(r'C:\Users\alifa\Documents\AI\DATA\Robtech\HW4\HW4-dataset\dataset\classification\train1000',loadFor=mode)
    imagesTrain, imagesTest, labelTrain, labelTest = util.Split(images,label)

