import numpy as np
import pandas as pd

import os, glob, re, scipy.io
import seaborn as sns
from joblib import dump, load
import matplotlib.pyplot as plt
    

###############################################################################
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import precision_score, recall_score, precision_recall_curve, f1_score, fbeta_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.utils.multiclass import unique_labels

from xgboost import XGBClassifier
from mlxtend.classifier import StackingClassifier
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
        
        if no_of_classes == 3:
            self.labels = {
                1: "Coupe/Convertibles",
                2: "Sedans",
                3: "SUV/Trucks/Vans"
            }
        elif no_of_classes == 5:
            self.labels = {
                1: "Coupe/Convertibles",
                2: "Sedans",
                3: "SUV",
                4: "Trucks",
                5: "Vans"
            }
        
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
        self.Xtrain_fileName = targets_df[ self.train_test == 0]["fileNumber"]
        self.xtest_fileName = targets_df[ self.train_test == 1]["fileNumber"]
        
        self.Ytrain = targets_df[ self.train_test == 0 ][target_column]
        self.ytest = targets_df[ self.train_test == 1 ][target_column]
        
    def make_model(self, model, stdScaler = False, PCA_M=0, oversample=""):
        self.Xtrain_fit = self.Xtrain
        self.xtest_fit = self.xtest
        self.Ytrain_fit = self.Ytrain #To account for oversampling techniques
        #self.ytest_fit = self.ytest
        
        if stdScaler:
            scaler = StandardScaler()
            self.Xtrain_fit = scaler.fit_transform(self.Xtrain_fit)
            self.xtest_fit = scaler.transform(self.xtest_fit)
            
        if PCA_M:
            cars_PCA = PCA(n_components=PCA_M)
            self.Xtrain_fit = cars_PCA.fit_transform(self.Xtrain_fit)
            self.xtest_fit = cars_PCA.transform(self.xtest_fit)
            print(np.sum(cars_PCA.explained_variance_ratio_))
        
        if oversample == "random":
            ros = RandomOverSampler(random_state=3)
            self.Xtrain_fit, self.Ytrain_fit = ros.fit_sample(self.Xtrain_fit, self.Ytrain_fit)
        elif oversample == "SMOTE":
            smote_sample = SMOTE(random_state=3)
            self.Xtrain_fit, self.Ytrain_fit = smote_sample.fit_sample(self.Xtrain_fit, self.Ytrain_fit)
        #elif oversample == "ADASYN":
        #    adasyn_sample = ADASYN(random_state=3)
        #    self.Xtrain_fit, self.Ytrain_fit = adasyn_sample.fit_sample(self.Xtrain_fit, self.Ytrain_fit)
            
        
        self.model = model
        self.model.fit(self.Xtrain_fit, self.Ytrain_fit)
        
        # --- Calc model performance measures/metrics
        self.ytst_lbls = self.model.predict(self.xtest_fit)
        self.ytst_proba = self.model.predict_proba(self.xtest_fit)
        
        self.train_acc = self.model.score(self.Xtrain_fit, self.Ytrain_fit)
        self.test_acc = self.model.score(self.xtest_fit, self.ytest)
        self.test_f1_score = f1_score(self.ytest, self.ytst_lbls, average="macro")
        
        self.confusion_matrix = confusion_matrix(self.ytest, self.ytst_lbls)
        
    def model_results(self):
        if self.model != None:
            print("Train Score for the classifier: {:.3f}".format(self.train_acc))
            print("Test Score for the classifier: {:.3f}".format(self.test_acc))
            print("F1 score for the classifier: {}".format(self.test_f1_score))
            print("Confusion matrix: \n\n", self.confusion_matrix)
            
            plt.rcParams['figure.figsize'] = [10, 10]
            self.create_ROC_Curves()
    
    
    def create_ROC_Curves(self):
        fpr = dict()
        tpr = dict()
        auc_value = dict()
        
        classes = [i+1 for i in range(self.no_of_classes)]

        # --- Binarize the true and preds
        y_true_bin = label_binarize(self.ytest, classes)
        #y_pred_bin = label_binarize(ypred, classes)
        #print(y_true_bin.shape)

        # --- Get FPR, TPR & AUC for each class
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:,i], self.ytst_proba[:,i])
            #print(len(fpr[i]))
            auc_value[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i],
                     label="ROC Curve for " + self.labels[i+1]+" (area = {:.2f})".format(auc_value[i]),
                     linestyle = ":", lw=3)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize = 15)
        plt.ylabel('True Positive Rate', fontsize = 15)
        plt.legend(loc="lower right", fontsize=15)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.title("Multi-class ROC for Classifying Car Images: One class vs Rest", fontsize = 18)
        plt.show()
        
        
    
    def build_misclass_df(self):
        """
        """
        
        # --- Build up numpy ndarray adding one column at a time.
        # ---   yTrue <== yPred Labels first, then <== add yPred Probability , then <== fileNames
        temp_labels = self.ytst_lbls
        temp_ytrue_preds = np.append(np.array(self.ytest).reshape(-1, 1), temp_labels.reshape(-1, 1), axis=1)
        temp_y_wpred_proba = np.append(temp_ytrue_preds, self.ytst_proba, axis=1)
        temp_fileNames = np.array(self.xtest_fileName).reshape(-1, 1)
        temp_array = np.append(temp_fileNames, temp_y_wpred_proba, axis=1)
        
        cols = ["fileName", "ytrue_lbl", "ypred_lbl"] + [str(i) for i in range(1, self.no_of_classes + 1)]
        
        test_wPreds_df = pd.DataFrame(temp_array, columns=cols)
        
        #temp_df.sort_values('2', ascending=False, inplace=True)
        #np.array(temp_df[(temp_df['ytrue_lbl']==1.0) & ((temp_df["ypred_lbl"])==2.0)].head(20)['fileName'])
        return test_wPreds_df
    
    def get_misclass_topFileNames(self, ytrue, yfalse, number_of_imgs=10, top=True):
        """
        """
        # --- Get a combined dataframe with filenames, true test labels, predicted test labels and predict probas
        df_wpreds = self.build_misclass_df()
        
        # --- Build up dataframe to an array of top fileNames
        fileNames_df = df_wpreds[(df_wpreds['ytrue_lbl']==float(ytrue)) & ((df_wpreds["ypred_lbl"])==float(yfalse))]
        if top:
            fileNames_df.sort_values(str(int(yfalse)), ascending=False, inplace=True)
        else:
            fileNames_df.sort_values(str(int(yfalse)), ascending=True, inplace=True)

        no_of_misclass = fileNames_df.shape[0]
        
        # --- Create a list of image file names from the misclassified indices "list_of_indices"
        to_return = number_of_imgs
        if no_of_misclass == 0:
            to_return = 0
        elif no_of_misclass < number_of_imgs:
            to_return = no_of_misclass
        
        fileNames_to_return = np.array(fileNames_df['fileName'].head(to_return))
        
        return fileNames_to_return
         