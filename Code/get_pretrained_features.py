import numpy as np
import pandas as pd

from stanford_cars import *

from keras.models import Model

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

class GetPretrainedFeatures:
    """
    
    
    """
    
    def __init__(self, cnn="VGG16", database = GetStandfordCars(), layer = -1):
        
        self.cnn = cnn
        self.db = database
        self.layer_level = layer
        
    
    def get_pretrained_features(self, limit=20, folder = "../Data/features/"):
        
        feature_df = pd.DataFrame()
        
        model = PRETRAINED_MODELS[self.cnn]['model'](weights='imagenet', include_top=True)
        model_layer = Model(inputs=model.input, outputs=model.layers[self.layer_level].output)
        preprocess = PRETRAINED_MODELS[self.cnn]['preprocess']
        
        input_shape = PRETRAINED_MODELS[self.cnn]['shape']
        #print(input_shape)
        
        count = 0
        for img in self.db.getNextImgArray(target_size=input_shape):
            count += 1
            x = preprocess(img[1])
            features = model_layer.predict(x)
            features = [[img[0]] + list(np.ndarray.flatten(features))]
            feature_df = feature_df.append(pd.DataFrame(features), ignore_index=True)
            
            if count % 1000 == 0:
                feature_df.to_csv(folder + self.cnn + "_" + str(count//1000) + ".csv")
                print("Completed {} pics for {}".format(count, self.cnn))
                feature_df = pd.DataFrame()
            
            if count == limit:
                    break
        
        feature_df.to_csv(folder + self.cnn + "_rest" + ".csv")
    
    
    
    def stitch_feature_files(self, features_folder = "../Data/features/", subscript=(1,16), rest=True,
                            output_filename = False):
        """
        Stitches the features extracted from the CNNs using get_pretrained_features() method
        into a single csv file.
        
        Args:
            features_folder = location of extracted files, assumes "<cnn>_<x>.csv" format
            scubscript = a tuple with file subscipts from low to high (inclusive)
                         as the <x> of the csv files
            rest = Boolean to include "rest" as the <x> of the csv files
            output_filename = output_filename to use. Uses <cnn>.csv as default. Specify a string.
        
        Returns:
            a new csv file 
        """
        
        #feature_df = pd.DataFrame()
        
        
        if not output_filename:
            output_filename = self.cnn
        
        if "." not in output_filename:
            output_filename = features_folder + output_filename + ".csv"
        else:
            output_filename = features_folder + output_filename
        
        input_filename = features_folder + self.cnn + "_" + str(subscript[0]) + ".csv"
        features = np.genfromtxt(input_filename, delimiter=",", skip_header=1)
        
        for suffix in range(subscript[0]+1, subscript[1]+1):
            input_filename = features_folder + self.cnn + "_" + str(suffix) + ".csv"
            #df = pd.read_csv(input_filename)
            #feature_df = feature_df.append(df, ignore_index=True)
            data = np.genfromtxt(input_filename, delimiter=",", skip_header=1)
            features = np.concatenate((features, data), axis=0)
        
        if rest:
            input_filename = features_folder + self.cnn + "_rest" + ".csv"
            data = np.genfromtxt(input_filename, delimiter=",", skip_header=1)
            features = np.concatenate((features, data), axis=0)
        
        
        feature_df = pd.DataFrame(features[:,1:])
        feature_df.to_csv(output_filename)
        #np.savetxt(output_filename, features[:,1:], delimiter=',')
            
    