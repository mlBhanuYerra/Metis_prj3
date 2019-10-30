import numpy as np
import pandas as pd

import os, glob, re, scipy.io
import seaborn as sns
from joblib import dump, load
import matplotlib.pyplot as plt
    

###############################################################################
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import precision_score, recall_score, precision_recall_curve, f1_score, fbeta_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.utils.multiclass import unique_labels

###############################################################################
from keras.models import Model
from keras.preprocessing import image
# Pretrained
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_ppi
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_ppi
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet_v2 import preprocess_input as Res50V2_ppi
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inceptv3_ppi
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input as moblv2_ppi

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

###############################################################################
from StanfordCars import *

PRETRAINED_MODELS = {
    'VGG16': {
        'model': VGG16,
        'preprocess': vgg16_ppi,
        'shape': (224, 224)
    },
    'VGG19': {
        'model': VGG19,
        'preprocess': vgg19_ppi,
        'shape': (224, 224)
    },
    'ResNet50V2': {
        'model': ResNet50V2,
        'preprocess': Res50V2_ppi,
        'shape': (224, 224)
    },
    'InceptionV3': {
        'model': InceptionV3,
        'preprocess': inceptv3_ppi,
        'shape': (299, 299)
    },
    'MobileNetV2': {
        'model': MobileNetV2,
        'preprocess': moblv2_ppi,
        'shape': (224, 224)
    }
}


class retrain_Keras():
    """
    """ 
    
    def __init__(self, cnn="MobileNetV2", database = GetStandfordCars(), path="../Data/", no_of_classes = 5, \
                 add_layers = [600, 300, 100], input_size = (224, 224) ):
        """
        
        properties:
        - cnn 
        - no_of_classes
        - classes
        - features_file
        - target_labels_file
        - train_test (a list to identify a train/test datapoint)
        
        """
        
        if cnn in PRETRAINED_MODELS.keys():
            self.cnn = cnn

        self.database = database
        self.path = path
        self.no_of_classes = no_of_classes
        self.target_labels_file = path + "cars_labels_train_test.csv"
        self.add_layers = add_layers
        self.input_size = input_size
        
        self.get_data()
        self.set_model()

    def set_model(self):

        base_model = PRETRAINED_MODELS[self.cnn]['model'](weights='imagenet', include_top=True)
        
        model_layer = base_model.layers[-2].output
        for fc_layer in self.add_layers:    
            model_layer = Dense(fc_layer, activation='relu')(model_layer)

        model_layer = Dense(self.no_of_classes, activation='softmax')(model_layer)

        self.model = Model(inputs = base_model.input, outputs = model_layer)
        #model_layer = Model(inputs=model.input, outputs=model.layers[self.layer_level].output)
        #preprocess = PRETRAINED_MODELS[self.cnn]['preprocess']
        
        #input_shape = PRETRAINED_MODELS[self.cnn]['shape']


    #def get_layers(self):
        #for layer in self.model.layers:


    def org_test_train_csv(self):
        """
        """
        train_file = self.path + "train_classes"+str(self.no_of_classes)+".csv"
        test_file = self.path + "test_classes"+str(self.no_of_classes)+".csv"

        trainStream = open(train_file, "w+")
        for pic, carClass in zip(self.Xtrain_fileName, self.Ytrain):
            fileName = self.database.getPath(pic)[-10:]
            trainStream.write(fileName+",'"+str(carClass)+"'\n")
        trainStream.close()

        testStream = open(test_file, "w+")
        for pic, carClass in zip(self.xtest_fileName, self.ytest):
            fileName = self.database.getPath(pic)[-10:]
            testStream.write(fileName+",'"+str(carClass)+"'\n")
        testStream.close()



    
    def get_data(self):
        """
        """
        #features_df = pd.read_csv(self.features_file)
        targets_df = pd.read_csv(self.target_labels_file)
        
        # --- Get the class names and labels columns
        if self.no_of_classes == 3:
            target_column = "Label1_No"
            target_labels = "Label1"
        elif self.no_of_classes == 5:
            target_column = "Label2_No"
            target_labels = "Label2"
        else:
            target_column = "class_no"
            target_labels = "class_label"
            
        # --- Get train-test split as binary var: train as 0 & test as 1
        self.train_test = np.array(targets_df.loc[:,'trn0tst1'])

        self.Xtrain_fileName = targets_df[ self.train_test == 0]["fileNumber"]
        self.xtest_fileName = targets_df[ self.train_test == 1]["fileNumber"]
        
        self.Ytrain = targets_df[ self.train_test == 0 ][target_column]
        self.ytest = targets_df[ self.train_test == 1 ][target_column]

        # --- memory intense. AVOID
        #Xtrain = []
        #xtest = []
        #
        #for pic in self.Xtrain_fileName:
        #    Xtrain.append(self.database.getImgArray(pic, target_size=self.input_size ))
        #
        #for pic in self.xtest_fileName:
        #    xtest.append(self.database.getImgArray(pic, target_size=self.input_size ))
        #
        #reShape = tuple([-1] + list(Xtrain[0].shape[1:]))
        #self.Xtrain = np.array(Xtrain).reshape(reShape)
        #self.xtest = np.array(xtest).reshape(reShape)


        
    def make_model():
        pass
