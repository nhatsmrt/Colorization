import numpy as np
import os
import math

import cv2
import skimage
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imread, imsave
from sklearn.preprocessing import OneHotEncoder


from sklearn.neighbors import NearestNeighbors

def create_data(data_path, N_IMG_MAX):
    x = []
    y = []

    n_img_processed = 0
    for file in os.listdir(data_path):
        if file.endswith('.png') or file.endswith('.jpg'):
            img_path = data_path + file
            image = cv2.imread(img_path)
            image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            x.append(image_lab[:, :, 0].reshape(128, 128, 1).astype(np.float32))

            # Convert the original image to Lab, taking only the latter dimensions:
            y.append(image_lab[:, :, 1:3])
            n_img_processed += 1

            if n_img_processed == N_IMG_MAX:
                break

    return np.array(x), np.array(y)

def create_data_places(data_path, N_IMG_MAX_PER_CLASS = 500, N_CLASS_MAX = 256):

    x = []
    ab = []
    y = []

    n_classes = 0

    for letter in os.listdir(data_path):
        if letter != ".DS_Store":
            for class_name in os.listdir(data_path + letter + "/"):
                n_classes += 1
                n_img_processed = 0

                if class_name != ".DS_Store":
                    for file in os.listdir(data_path + letter + "/" + class_name + "/"):
                        if file.endswith('.png') or file.endswith('.jpg'):
                            y.append(n_classes - 1)

                            img_path = data_path + letter + "/" + class_name + "/" + file
                            image = cv2.imread(img_path)
                            image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                            x.append(image_lab[:, :, 0].reshape(128, 128, 1).astype(np.float32))

                            # Convert the original image to Lab, taking only the latter dimensions:
                            ab.append(image_lab[:, :, 1:3])
                            n_img_processed += 1

                            if n_img_processed == N_IMG_MAX_PER_CLASS:
                                break

                    if n_classes == N_CLASS_MAX:
                        return np.array(x), np.array(ab), np.array(y), n_classes

    return np.array(x), np.array(ab), np.array(y), n_classes



def oh_encode(ab):
    batch_size = ab.shape[0]
    img_h = ab.shape[1]
    img_w = ab.shape[2]

    ab = ab.reshape(batch_size * img_h * img_h * 2, -1)
    encoder = OneHotEncoder(n_values = 256)
    ab = encoder.fit_transform(ab).toarray()


    return ab.reshape(batch_size, img_h, img_w, 2, 256)

def oh_decode(ab_oh):
    return np.argmax(ab_oh, axis = -1)




def create_data_test(data_path):
    x = []
    y = []
    img_path = data_path + '/images/'
    for img in os.listdir(img_path):
        if img != '.DS_Store':
            image = cv2.imread(img_path + img)
            image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
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
        prediction = cv2.cvtColor(np.uint8(np.concatenate((X_test[i], predictions[i]), axis = 2)), cv2.COLOR_Lab2RGB)
        original = cv2.cvtColor(np.uint8(np.concatenate((X_test[i], y_test[i]), axis = 2)), cv2.COLOR_Lab2RGB)
        cv2.imwrite(path + str(i) + "_predict.JPEG", prediction)
        cv2.imwrite(path + str(i) + "_original.JPEG", original)
        grey_scale = X_test[i].reshape(X_test[i].shape[0], X_test[i].shape[1])
        cv2.imwrite(path + str(i) + "_greyscale.JPEG", grey_scale)




def create_bins(a_min = 42, a_max = 226, b_min = 20, b_max = 223, bin_size = 10):
    # a_min = -87, a_max = 95, b_min = -108, b_max = 99, bin_size = 10
    bins = []
    num_a = math.floor((a_max - a_min) / bin_size) + 1
    num_b = math.floor((b_max - b_min) / bin_size) + 1
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


    def encode_pts(self, pts_nd, save_path = None, load_path = None):


        if load_path is not None:
            ret = np.load(load_path)
            return ret

        pts_flt = flatten_pts(pts_nd)
        num_pix = pts_flt.shape[0]
        (dists, inds) = self._nbrs.kneighbors(pts_flt)
        print(dists)
        print(inds)
        self._pts_enc_flt = np.zeros(shape = (num_pix, self._K))
        self._p_inds = np.arange(0, num_pix, dtype = 'int')[:, na()]


        wts = np.exp(-dists ** 2 / (2 * self._sigma ** 2))
        wts = wts / np.sum(wts, axis=1)[:, na()]

        self._pts_enc_flt[self._p_inds, inds] = wts

        ret = deflatten_pts(self._pts_enc_flt, pts_nd)
        if save_path is not None:
            np.save(save_path, ret)

        return ret

    def decode_pts(self, pts_enc_nd):
        pts_enc_flt = flatten_pts(pts_enc_nd)
        pts_dec_flt = np.dot(pts_enc_flt, self._bins)
        pts_dec_nd = deflatten_pts(pts_dec_flt, pts_enc_nd)
        return pts_dec_nd

class FourNNEncoders():

    def __init__(self, bins, sigma = 5, a_min = 42, a_max = 226, b_min = 20, b_max = 223, bin_size = 10):
        self._bins = bins
        self._sigma = sigma
        self._K = self._bins.shape[0]
        self._amin = a_min
        self._bmin = b_min
        self._num_a = math.ceil((a_max - a_min) / bin_size)
        self._num_b = math.ceil((b_max - b_min) / bin_size)
        self._bin_size = bin_size


    def encode_pts(self, pts_nd, save_path = None, load_path = None):


        if load_path is not None:
            ret = np.load(load_path)
            return ret

        pts_flt = flatten_pts(pts_nd)
        num_pix = pts_flt.shape[0]
        (dists, inds) = self.find_nnbrs(pts_flt)
        self._pts_enc_flt = np.zeros(shape = (num_pix, self._K))
        self._p_inds = np.arange(0, num_pix, dtype = 'int')[:, na()]


        wts = np.exp(-dists ** 2 / (2 * self._sigma ** 2))
        wts = wts / np.sum(wts, axis=1)[:, na()]

        self._pts_enc_flt[self._p_inds, inds] = wts

        ret = deflatten_pts(self._pts_enc_flt, pts_nd)
        if save_path is not None:
            np.save(save_path, ret)

        return ret


    def find_nnbrs(self, pts_fltn):
        pts_fltn_rounded = (np.floor(pts_fltn) - np.tile(np.array([self._amin % self._bin_size, self._bmin % self._bin_size]), (pts_fltn.shape[0], 1))) // self._bin_size * self._bin_size + np.tile(np.array([self._amin % self._bin_size, self._bmin % self._bin_size]), (pts_fltn.shape[0], 1))
        pts_fltn_rounded_all = add_vertices(pts_fltn_rounded, inc = self._bin_size)
        pts_fltn_ind = self.find_ind(pts_fltn_rounded_all)
        pts_fltn_dist = dist_mat(pts_fltn, pts_fltn_rounded_all)

        return pts_fltn_dist, pts_fltn_ind

    def find_ind(self, pts_fltn_rounded_all):
        tmp = (pts_fltn_rounded_all - np.tile(np.array([self._amin, self._bmin]), (pts_fltn_rounded_all.shape[0], 4, 1))) / self._bin_size
        tmp[:, :, 0] *= self._num_b
        return np.sum(tmp, axis = -1, dtype = np.int32)


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

def dist_mat(a, b):
    a_broadcast = np.zeros(shape = (b.shape[0], 4, 2))
    for i in range(a_broadcast.shape[0]):
        a_broadcast[i, :] = np.tile(a[i, :], (4, 1))
    return np.sqrt(np.sum((b - a_broadcast) ** 2, axis=-1))

def add_vertices(a, inc):
    ret = np.zeros(shape = (a.shape[0], 4, 2))
    for i in range(ret.shape[1]):
        ret[:, i, :] = a

    ret[:, 1, 0] += inc
    ret[:, 2, 1] += inc
    ret[:, 3, :] += inc


    return ret





