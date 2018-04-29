import numpy as np
import cv2
import sklearn
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog


def bin_spatial(img, size = (32, 32)):
    img = img.copy()
    features = cv2.resize(img, size).ravel()
    return features

def color_hist(img, nbins = 32, bins_range = (0, 255)):
    channel1_hist = np.histogram(img[:,:,0], bins = nbins, range = bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins = nbins, range = bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins = nbins, range = bins_range)
    features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return features

def extract_color_features(img, cspace = 'RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    for img in images:
        img = mpimg.imread(path)
        if cspace != 'RGB':
            exec("%s = %s" % ('img','cv2.cvtColor(img, cv2.COLOR_RGB2' + cspace + ')'))
        spatial_features = bin_spatial(img, spatial_size)
        hist_features = color_hist(img, hist_bins, hist_range)
        features.append(np.concatenate((spatial_features, hist_features)))
    return features

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features =  hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features

def extract_color_features(images, cspace = 'RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    features = []
    for path in images:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if cspace != 'RGB':
            exec("%s = %s" % ('img','cv2.cvtColor(img, cv2.COLOR_RGB2' + cspace + ')'))
        spatial_features = bin_spatial(img, spatial_size)
        hist_features = color_hist(img, hist_bins, hist_range)
        features.append(np.concatenate((spatial_features, hist_features)))
    return features


def extract_hog_features(imgs, cspace='RGB', orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for path in imgs:
        # Read in each one by one
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            exec("%s = %s" % ('img','cv2.cvtColor(img, cv2.COLOR_RGB2' + cspace + ')'))

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.append(get_hog_features(img[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(img[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features
