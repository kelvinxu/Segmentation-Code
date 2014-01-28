"""
This script is designed to create filter responses for the for the Pascal
VOC2012 ImageSet.

It works by computing filter responses for images, then pickling the entries. The 
format required is 'data', 'labels', 'K', where is K is the number of classes (i.e
2 for binary classification. 

Kelvin Xu 01/20/2014
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imgtools.filter as filter
from scipy import misc
import cPickle as pickle
import os.path

# For running on the cluster, the VOC image set is here.
#VOCpath = '/ais/gobi3/u/yujiali/pascal_2012/VOC2012/'
VOCpath = '/home/kelvin/Documents/Thesis/VOCdevkit/VOC2012/'
jpegfolder = 'JPEGImages/'
segfolder = 'SegmentationClass/'
classfile = 'ImageSets/Main/aeroplane_train.txt'
outputfile = '/home/kelvin/Documents/Thesis/Kelvin.pdata'

def fetch_labeled(file):
    '''
    Fetches the labelled image, which is in PNG form. 1 signifies
    an object

    file - location of the labelled image

    '''
    if os.path.isfile(VOCpath + segfolder + file + '.png'):
        labeling = misc.imread(VOCpath + segfolder + file + '.png')
        labeling[labeling != 1] = 0
        return labeling        

def gen_training_val_index(n_datapoints, split):
    """
    n_datapoints: size of original set
    split: number between 0 & 1 representing the proportion of points
    to be in the training set

    Generates a split for training and validation data
    Takes in input_size, split, returns a randomized split of
    the data

    returns training_indx, validation_inx
    """

    assert (split < 1) and (split >0)
    n_train = int(n_datapoints * split)
    r_idx = np.random.permutation(n_datapoints)
    return r_idx[:n_train], r_idx[n_train:] 

def train_nn_seg():
    """
    Takes pickled dict of data, then seperates it into test and 
    validation and trains a nn for the unary potentials

    """
    pass
    
def filters(VOCpath, file):
    '''
    Retrieves all the images, returns the filter responses,
    Stacks arrays and returns them for pickling. 
    
    Downsamples the produced responses
    Return: (Data NXF matrix of features: N - sampled number
            F - length of feature vector)
            (label: N-d vector with class labels)
    '''
    f = open(VOCpath + '/' + file, 'r')
    input_images = []
    l = []
    images = 0
    for lines in f:
        x = lines.split()
        # if the second entry is a 1, the image contains an object
        if int(x[1]) == 1:
            labeling = fetch_labeled(x[0])
            if labeling is None:
                print x[0]+ ' does not exit'
            else:
                jpeglocation = VOCpath + jpegfolder + x[0] + '.jpg'
                labeling = labeling.flatten()
                im = Image.open(jpeglocation)
                input_images.append(np.array(im))
                l.append(labeling)
                images += 1
        if images == 5:
            resp = filter.filter_response(input_images)
            data = resp[0]
            for i in range(1,len(resp)):
                data = np.vstack((data,resp[i]))
            label = np.concatenate(np.array(l),1) 
            label = label.flatten()
            #down sample the data
            print label.mean()
            rows,cols = data.shape
            rand_rows = np.random.randint(rows, size = int(rows*.25))
            return (label[rand_rows],data[rand_rows])
            break

def main():
    label,data = filters(VOCpath, classfile)
    # see readme.txt in nn code for details
    print label.mean()
    storedict = {'data': data, 'labels':label, 'K': 2}   
    f = open(outputfile, mode='w')
    pickle.dump(storedict,f)
    f.close()

if __name__ == '__main__':
    main()
