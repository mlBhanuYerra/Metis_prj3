import numpy as np
import pandas as pd

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
                feature_df = pd.DataFrame()
            
            if count == limit:
                    break
        
        if count != limit:
            feature_df.to_csv(folder + self.cnn + "_rest" + ".csv")
            
    