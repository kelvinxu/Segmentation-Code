"""

Script created to put pascal images in binary format for usage in nn's
Yujia's NN code requires a specific format for data file 

Kelvin Xu 1/20/2014

"""
import sys
from scipy import misc
import numpy as np

def relabel_image():
    im = misc.imread('2007_000032.png')
    (sy,sx) = im.shape
    im[im != 1] = 0 
    im[im == 1] = 255
    misc.imsave('kelvin.png',im)
    '''
    for i in range(sx):
        for j in range(sy):
            if im[j][i] == 200:
                im[j][i] = 0
    misc.imsave('test.png',im)
    '''
def main():
    sys.path.append("~/Documents/Thesis/VOCdevkit/VOC2012/SegmentationClass")
    relabel_image()

if __name__ == "__main__":
    main()
