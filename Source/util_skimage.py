import numpy as np
import os
import math

import cv2
import skimage
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imread, imsave


from sklearn.neighbors import NearestNeighbors


def create_data(data_path):
    x = []
    y = []
    for folder in os.listdir(data_path):
        if folder != '.DS_Store':
            img_path = data_path + '/' + folder + '/images/'
            for img in os.listdir(img_path):
                if img != '.DS_Store':
                    image = imread(img_path + img)
                    if (image.shape != (64, 64)):
                        image_lab = rgb2lab(image)
                        x.append(image_lab[:, :, 0].reshape(64, 64, 1).astype(np.float32))
                        # Convert the original image to Lab, taking only the latter dimensions:
                        y.append(image_lab[:, :, 1:3])
    return np.array(x), np.array(y)

def create_data_test(data_path):
    x = []
    y = []
    img_path = data_path + '/images/'
    for img in os.listdir(img_path):
        if img != '.DS_Store':
            image = imread(img_path + img)
            if (image.shape != (64, 64)):
                image_lab = rgb2lab(image)
                x.append(image_lab[:, :, 0].reshape(64, 64, 1).astype(np.float32))
                # Convert the original image to Lab, taking only the latter dimensions:
                y.append(image_lab[:, :, 1:3])
    return np.array(x), np.array(y)


def deflatten(images, n_channels = 2):
    n_image = images.shape[0]
    image_dim = int(math.sqrt(images[0].shape[0]/n_channels))
    return images.reshape(n_image, image_dim, image_dim, n_channels)

def export_predictions(path, predictions):
    for i in range(len(predictions)):
        imsave(path + str(i) + ".JPEG", predictions[i])

def compare(path, predictions, y_test, X_test):
    # X_test = X_test
    # predictions = predictions * 128 / 100
    # y_test = y_test * 128 / 100

    for i in range(len(predictions)):
        # prediction = cv2.cvtColor(np.uint8(np.concatenate((X_test[i], predictions[i]), axis = 2)), cv2.COLOR_Lab2RGB)
        prediction = lab2rgb(np.concatenate((X_test[i], predictions[i]), axis = 2))
        # original = cv2.cvtColor(np.uint8(np.concatenate((X_test[i], y_test[i]), axis = 2)), cv2.COLOR_Lab2RGB)
        original = lab2rgb(np.concatenate((X_test[i], y_test[i]), axis = 2))
        imsave(path + str(i) + "_predict.JPEG", prediction)
        imsave(path + str(i) + "_original.JPEG", original)
        grey_scale = X_test[i].reshape(X_test[i].shape[0], X_test[i].shape[1]) / 100
        imsave(path + str(i) + "_greyscale.JPEG", grey_scale)




def create_bins(a_min = -87, a_max = 95, b_min = -108, b_max = 99, bin_size = 10):
    # a_min = -87, a_max = 95, b_min = -108, b_max = 99, bin_size = 10
    bins = []
    num_a = math.ceil((a_max - a_min) / bin_size)
    num_b = math.ceil((b_max - b_min) / bin_size)
    n_classes = num_a * num_b
    for i in range(num_a):
        for j in range(num_b):
            bins.append([a_min + i * bin_size, b_min + j * bin_size])
    return np.array(bins)



# def bin_distribution(y_train, y_test, bin, bin_size = 10):
#     a_min = bin[0][0]
#     b_min = bin[0][1]
#     n_classes = bin.shape[0]
#
#     for img in y_train:
#         for h in img:
#             for w in h:
#                 a_inc = math.ceil(w[0] - a_min)




# Encode points using NN search and Gaussian kernel:

class NNEncode():
    def __init__(self, bins, NN = 5, sigma = 5):
        '''

        :param bins: [n_classes, a, b]:
        :param NN: int: number of neareast neighbors considered
        :param sigma: standard deviation of Gaussian kernel
        '''
        self._NN = NN
        self._sigma = sigma
        self._bins = bins
        self._K = self._bins.shape[0]
        self._nbrs = NearestNeighbors(n_neighbors = NN, algorithm = 'ball_tree').fit(bins)
        self._already_used = False


    def encode_pts(self, pts_nd):
        pts_flt = flatten_pts(pts_nd)
        num_pix = pts_flt.shape[0]
        (dists, inds) = self._nbrs.kneighbors(pts_flt)
        self._pts_enc_flt = np.zeros(shape = (num_pix, self._K))
        self._p_inds = np.arange(0, num_pix, dtype = 'int')[:, na()]


        wts = np.exp(-dists ** 2 / (2 * self._sigma ** 2))
        wts = wts / np.sum(wts, axis=1)[:, na()]

        self._pts_enc_flt[self._p_inds, inds] = wts

        return deflatten_pts(self._pts_enc_flt, pts_nd)

    def decode_pts(self, pts_enc_nd):
        pts_enc_flt = flatten_pts(pts_enc_nd)
        pts_dec_flt = np.dot(pts_enc_flt, self._bins)
        pts_dec_nd = deflatten_pts(pts_dec_flt, pts_enc_nd)
        return pts_dec_nd


    # def encode_points_mtx_nd(self, pts_nd, axis = 1, same_block = True):
    #     pts_flt = flatten_nd_array(pts_nd, axis = axis)
    #     num_pts = pts_flt.shape[0]
    #     self.already_used = True
    #
    #
    #     self.pts_enc_flt = np.zeros((num_pts, self.K))
    #     self.p_inds = np.arange(0, num_pts, dtype = 'int')[:, na()]
    #
    #     (dists, inds) = self.nbrs.kneighbors(pts_flt)
    #     wts = np.exp(-dists**2 / (2*self.sigma**2))
    #     wts = wts/np.sum(wts, axis = 1)[:, na()]
    #
    #     self.pts_enc_flt[self.p_inds, inds] = wts
    #     pts_enc_nd = unflatten_2d_array(self.pts_enc_flt, pts_nd, axis = axis)
    #
    #     return pts_enc_nd
    #
    # def decode_points_mtx_nd(self, pts_enc_nd, axis = 1):
    #     pts_enc_flt = flatten_nd_array(pts_enc_nd, axis = axis)
    #     pts_dec_flt = np.dot(pts_enc_flt, self.bins)
    #     pts_dec_nd = unflatten_2d_array(pts_dec_flt, pts_enc_nd, axis = axis)
    #     return pts_dec_nd







# Some utility functions:
def flatten_pts(pts_nd):
    return pts_nd.reshape(-1, pts_nd.shape[-1])

def deflatten_pts(pts_flt, pts_nd):
    dimensions = pts_nd.shape[:-1]
    dimensions = dimensions + (-1,)
    pts_dflt = pts_flt.reshape(dimensions)
    return pts_dflt


def flatten_nd_array(pts_nd,axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array([axis])) # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax,np.array([axis]).flatten()),axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS,SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array([axis])) # non axis indices
    NPTS = np.prod(SHP[nax])

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax,np.array([axis]).flatten()),axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out

def na():
    return np.newaxis





