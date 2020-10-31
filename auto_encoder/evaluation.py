#!/usr/bin/env python
#
# evaluation.py
#
# Created by Scott Lee on 10/30/20.
#

'''
Description: evaluation metrics. Original code written by Nikesh Mishra in Colab notebook.

'''

import tensorflow as tf
import keras.backend as K
from sklearn.utils.extmath import cartesian

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# taken from https://jeune-research.tistory.com/entry/Loss-Functions-for-Image-Segmentation-Distance-Based-Losses

class HausdorffLoss:
    def __init__(self, W, H, alpha=2):
        self.W = W
        self.H = H
        self.alpha = alpha
        self.all_img_locations = tf.convert_to_tensor(cartesian([np.arange(W), np.arange(H)]), dtype=tf.float32)
        self.max_dist = math.sqrt(W ** 2 + H ** 2)

    def cdist(self, A, B):
        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])
        D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
        return D

    def weighted_hausdorff_distance(self, y_true, y_pred):
        all_img_locations = self.all_img_locations
        W = self.W
        H = self.H
        alpha = self.alpha
        max_dist = self.max_dist
        eps = 1e-6

        y_true = K.reshape(y_true, [W, H])
        gt_points = K.cast(tf.where(y_true > 0.5), dtype=tf.float32)
        num_gt_points = tf.shape(gt_points)[0]

        y_pred = K.flatten(y_pred)
        p = y_pred
        p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p, axis=-1), num_gt_points))

        d_matrix = self.cdist(all_img_locations, gt_points)
        num_est_pts = tf.reduce_sum(p)
        term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

        d_div_p = K.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)
        d_div_p = K.clip(d_div_p, 0, max_dist)
        term_2 = K.mean(d_div_p, axis=0)

        return term_1 + term_2

    def hausdorff_loss(self, y_true, y_pred):
        batched_losses = tf.map_fn(lambda x:
                                   self.weighted_hausdorff_distance(x[0], x[1]),
                                   (y_true, y_pred),
                                   dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))

if __name__ == '__main__':
    # Sample method for running code. Need to add actual models, labels, and data here. Most likely will import the
    # functions in other code blocks to run
    print("Evaluate on test data:")
    results = model.evaluate(data, [labels, data], batch_size=1)
    print("     test loss, test acc:", results)
    print("Dice coefficient: " + str(dice_coef(data_paths[4], predictions)))
    print("Dice loss: " + str(dice_coef_loss(data_paths[4], predictions)))
    print("Hausdorff Loss: " + str(HausdorffLoss.weighted_hausdorff_distance(data_paths[4], predictions)))