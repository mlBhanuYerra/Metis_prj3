import numpy as np
import pandas as pd

import os, glob, re, scipy.io
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import Model

class GetStandfordCars:
    """
    
    
    """
    
    def __init__(self, loc="../Data/car_ims/", labels_matlab_file ="../Data/cars_annos.mat"):
        
        self.car_images_location = loc
        self.labels_mat = scipy.io.loadmat(labels_matlab_file)
        
        self.class_names = self.labels_mat['class_names'][0]
        
        self.max_pics = 16185
        
        self.annot_df = self.create_annot_df()
        # --- for future
        #  Code to check if the 'loc' is a valid dataset
        #  if not, download data and unzip
        #  same for the labels matlab file
        
    
    def bringAnnot(self, fileName):
        """
        """
        
        if type(fileName) == str:
            numb = int(re.sub(r'[^0-9]', '', fileName))
        else:
            numb = int(fileName)

        return ((self.labels_mat['annotations'][0][numb-1]))  
    
    
    def bringup_ClassLabel(self, fileName):
        if type(fileName) == str:
            numb = int(re.sub(r'[^0-9]', '', fileName))
        else:
            numb = int(fileName)

        #print("Class of image {}: {}".format(fileName, class_labels[int(mat['annotations'][0][numb-1][5])-1][0]))
        return self.class_names[int(self.labels_mat['annotations'][0][numb-1][5])-1][0]

        
    def bringAnnotList(self, fileName):
        """
        """
        
        if type(fileName) == str:
            numb = int(re.sub(r'[^0-9]', '', fileName))
        else:
            numb = int(fileName)
        
        annot_array = bringAnnot(numb)
        xmin = annot_array[1][0][0]
        ymin = annot_array[2][0][0]
        xmax = annot_array[3][0][0]
        ymax = annot_array[4][0][0]
        class_no = annot_array[5][0][0]
        class_label = bringup_ClassLabel(numd)
        train_test = annot_array[6][0][0]
        
        # --- file number, xmin, ymin, xmax, ymax, class no, train(0)/test(1)
        annot = [numb, xmin, ymin, xmax, ymax, class_no, class_label, train_test]

        return annot  
    
    def create_annot_df(self):
        
        list_df = []
        
        for fileName in range(1, self.max_pics + 1):
            list_df.append(bringAnnotList(fileNamwe))
            
        cols = ["fileNumber", "xmin", "ymin", "xmax", "ymax", "class_no", "class_label", "train_test"]
        
        return pd.DataFrame(list_df, columns = cols)
    
    def show_annots(fileName):
        """
        """

        if type(fileName) == str:
            numb = int(re.sub(r'[^0-9]', '', fileName))
        else:
            numb = int(fileName)

        print("Annot:     "+str(cars.labels_mat['annotations'][0][numb-1]))
        print("File Name: "+str(cars.labels_mat['annotations'][0][numb-1][0][0]))
        print("x min:     "+str(cars.labels_mat['annotations'][0][numb-1][1][0][0]))
        print("y min:     "+str(cars.labels_mat['annotations'][0][numb-1][2][0][0]))
        print("x max:     "+str(cars.labels_mat['annotations'][0][numb-1][3][0][0]))
        print("y max:     "+str(cars.labels_mat['annotations'][0][numb-1][4][0][0]))
        print("class no:  "+str(cars.labels_mat['annotations'][0][numb-1][5][0][0]))
        print("Train(0)/Test(1): "+str(cars.labels_mat['annotations'][0][numb-1][6][0][0]))

    
    def getPath(self, number):
        fileName  = str('0') * int(6 - len(str(number)))
        fileName += str(number)+".jpg"
        return (self.car_images_location + fileName)

    
    def getImgArray(self, number, target_size=(224, 224) ):
        img_path =  self.getPath(number)
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        return x
    

    def getNextImgArray(self, target_size=(224, 224) ):
        count = 0
        
        while count<self.max_pics:
            count += 1        
        
            yield (count, self.getImgArray(count, target_size))
    

    def showRandomNxN(self, N):
        '''
        Function to plot the MNIST data on a grid of NxN
        '''
        plt.rcParams['figure.figsize'] = [30, 30]

        listofRand = np.random.randint(1, 16185, size=int(N*N))

        image_size = (128, 128)

        fig = plt.figure()

        for i in range(0, N*N):
            img = image.load_img(self.getPath(listofRand[i]), target_size=image_size)
            ax = fig.add_subplot(N, N, i+1)
            imgplot = ax.imshow(img)
            ax.set_title(self.bringup_ClassLabel(listofRand[i]))
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show();