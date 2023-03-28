from siamese import Model
from utils import utils
import numpy as np

util = utils()
mode = 'triplet'
if mode == None:
    imageA,imageB,label = util.loadData(r'C:\Users\alifa\Documents\AI\DATA\Robtech\HW4\HW4-dataset\dataset\classification\train1000')
    imageA = np.concatenate(imageA,axis = 0)
    imageB = np.concatenate(imageB,axis = 0)
    imageATrain, imageATest, imageBTrain, imageBTest, labelTrain, labelTest = util.Split(imagesA=imageA,imagesB=imageB,label=label)
    model = Model(mode='simple')
    model.compileModel()
    model.net = mode.buildModel(mode = 'simple')
    model.compileModel()
    model.train([imageATrain,imageBTrain],np.array(labelTrain),valData=[[imageATest,imageBTest],labelTest])
elif mode == 'triplet':
    import tensorflow_addons
    images , label = util.loadData(r'C:\Users\alifa\Documents\AI\DATA\Robtech\HW4\HW4-dataset\dataset\classification\train1000',loadFor=mode)
    imagesTrain, imagesTest, labelTrain, labelTest = util.Split(images,label)
    model = Model(mode=mode)
    model.net = mode.buildModel(mode = mode)
    model.compileModel()
    model.train(imagesTrain,np.array(labelTrain),valData=[imagesTest,labelTest])

