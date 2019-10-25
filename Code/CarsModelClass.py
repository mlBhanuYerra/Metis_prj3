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


class cars_model_class():
    """
    """
    
    def __init__(self, cnn="MobileNetV2", no_of_classes = 3, path="../Data/"):
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
        self.no_of_classes = no_of_classes
        self.features_file = path + "features/" + self.cnn + ".csv"
        self.target_labels_file = path + "cars_labels_train_test.csv"
        
        self.get_data()
    
    def get_data(self):
        """
        """
        features_df = pd.read_csv(self.features_file)
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
        
        self.Xtrain = features_df[ self.train_test == 0 ].iloc[:,2:]
        self.xtest = features_df[ self.train_test == 1 ].iloc[:,2:]
        self.Xtrain_fileName = features_df[ self.train_test == 0 ].iloc[:,1]
        self.xtest_fileName = features_df[ self.train_test == 1 ].iloc[:,1]
        
        self.Ytrain = targets_df[ self.train_test == 0 ][target_column]
        self.ytest = targets_df[ self.train_test == 1 ][target_column]
        
    def make_model():
        pass
    
    def build_misclass_df(self, ypred_lbls, ypred_proba):
        """
        """
        
        # --- Build up numpy ndarray adding one column at a time.
        # ---   yTrue <== yPred Labels first, then <== add yPred Probability , then <== fileNames
        temp_ytrue_preds = np.append(np.array(self.ytest).reshape(-1, 1), ypred_lbls.reshape(-1, 1), axis=1)
        temp_y_wpred_proba = np.append(temp_ytrue_preds, ypred_proba, axis=1)
        temp_fileNames = np.array(self.xtest_fileName).reshape(-1, 1)
        temp_array = np.append(temp_fileNames, temp_y_wpred_proba, axis=1)
        
        cols = ["fileName", "ytrue_lbl", "ypred_lbl"] + [str(i) for i in range(1, self.no_of_classes + 1)]
        
        test_wPreds_df = pd.DataFrame(temp_array, columns=cols)
        
        #temp_df.sort_values('2', ascending=False, inplace=True)
        #np.array(temp_df[(temp_df['ytrue_lbl']==1.0) & ((temp_df["ypred_lbl"])==2.0)].head(20)['fileName'])
        return test_wPreds_df
    
    def get_misclass_topFileNames(self, ypred_lbls, ypred_proba, ytrue, yfalse, number_of_imgs=10):
        """
        """
        # --- Get a combined dataframe with filenames, true test labels, predicted test labels and predict probas
        df_wpreds = self.build_misclass_df(ypred_lbls, ypred_proba)
        
        # --- Build up dataframe to an array of top fileNames
        fileNames_df = df_wpreds[(df_wpreds['ytrue_lbl']==float(ytrue)) & ((df_wpreds["ypred_lbl"])==float(yfalse))]
        fileNames_df.sort_values(str(int(yfalse)), ascending=False, inplace=True)

        no_of_misclass = fileNames_df.shape[0]
        
        # --- Create a list of image file names from the misclassified indices "list_of_indices"
        to_return = number_of_imgs
        if no_of_misclass == 0:
            to_return = 0
        elif no_of_misclass < number_of_imgs:
            to_return = no_of_misclass
        
        fileNames_to_return = np.array(fileNames_df['fileName'].head(to_return))
        
        return fileNames_to_return
         