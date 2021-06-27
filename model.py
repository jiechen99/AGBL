import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim
from tensorflow.python.ops import array_ops
tf.app.flags.DEFINE_integer('text_scale', 512, '')

from nets import resnet_v1
from nets import vgg

FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])

def unpool2(inputs, ratio=2):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*ratio,  tf.shape(inputs)[2]*ratio])

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def cbam_block(input_feature, name, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    with tf.variable_scope(name):
        attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        attention_feature = spatial_attention(attention_feature, 'sp_at')
        print ("CBAM Hello")
    return attention_feature


def channel_attention(input_feature, name, ratio=8):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keep_dims=True)

        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_0',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_1',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keep_dims=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   name='mlp_0',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel,
                                   name='mlp_1',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

    return input_feature * scale


def spatial_attention(input_feature, name):
    kernel_size = 7
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keep_dims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keep_dims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = tf.layers.conv2d(concat,
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False,
                                  name='conv')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat

def model(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = vgg.vgg_16(images, is_training=is_training, scope='vgg_16')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2'], end_points['pool1']]
            for i in range(5):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None, None]
            h = [None, None, None, None, None]
            num_outputs = [None, 256, 128, 64, 32]
            for i in range(5):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 3:
                    h[i] = cbam_block(h[i], 'feature_fusion_{}'.format(i), 8)
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(unpool(h[i]), num_outputs[i], 3)
                    #h[1] = unpool2(h[1], 8)
                    #h[2] = unpool2(h[2], 4)
                    #h[3] = unpool2(h[3], 2)
                    #g[i] = tf.concat([h[1], h[2], h[3], h[4]], axis=-1)

                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            F_score = slim.conv2d(g[4], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            #F_score = unpool(F_score)
            #F_score = unpool(F_score)

            B_score = slim.conv2d(g[4], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            #F_score, B_score = tf.split(value=FB_score, num_or_size_splits=2, axis=3)
            #B_score = unpool(B_score)
            #B_score = unpool(B_score)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            #geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            #angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
            #F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, B_score


def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask, type):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    if type == 'classification':
        tf.summary.scalar('classification_dice_loss', loss)
    if type == 'border':
        tf.summary.scalar('border_dice_loss', loss)
    return loss

def OHNM_single_image(scores, n_pos, neg_mask):
    """Online Hard Negative Mining.
        scores: the scores of being predicted as negative cls
        n_pos: the number of positive samples
        neg_mask: mask of negative samples
        Return:
            the mask of selected negative samples.
            if n_pos == 0, top 10000 negative samples will be selected.
    """

    def has_pos():
        return n_pos * 3

    def no_pos():
        return tf.constant(10000, dtype=tf.int32)

    n_neg = tf.cond(n_pos > 0, has_pos, no_pos)
    max_neg_entries = tf.reduce_sum(tf.cast(neg_mask, tf.int32))

    n_neg = tf.minimum(n_neg, max_neg_entries)
    n_neg = tf.cast(n_neg, tf.int32)

    def has_neg():
        neg_conf = tf.boolean_mask(scores, neg_mask)
        vals, _ = tf.nn.top_k(neg_conf, k=n_neg)
        threshold = vals[-1]  # a negtive value
        selected_neg_mask = tf.logical_and(neg_mask, scores >= threshold)
        return selected_neg_mask

    def no_neg():
        selected_neg_mask = tf.zeros_like(neg_mask)
        return selected_neg_mask

    selected_neg_mask = tf.cond(n_neg > 0, has_neg, no_neg)
    return tf.cast(selected_neg_mask, tf.int32)


def OHNM_batch(neg_conf, pos_mask, neg_mask):
    batch_size = FLAGS.batch_size_per_gpu
    selected_neg_mask = []
    for image_idx in xrange(batch_size):
        image_neg_conf = neg_conf[image_idx, :]
        image_neg_mask = neg_mask[image_idx, :]
        image_pos_mask = pos_mask[image_idx, :]
        n_pos = tf.reduce_sum(tf.cast(image_pos_mask, tf.int32))
        selected_neg_mask.append(OHNM_single_image(image_neg_conf, n_pos, image_neg_mask))

    selected_neg_mask = tf.stack(selected_neg_mask)
    return selected_neg_mask

def dice_coefficient_OHNM(y_true_cls, y_pred_cls,
                     training_mask, type):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    pos_mask = tf.equal(y_true_cls, 1)
    neg_mask = tf.equal(y_true_cls, 0)

    selected_neg_mask = OHNM_batch(y_pred_cls, pos_mask, neg_mask)
    selected_pixel_mask = tf.cast(tf.cast(pos_mask, tf.int32) + selected_neg_mask, tf.float32)
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * selected_pixel_mask  *  training_mask)
    union = tf.reduce_sum(y_true_cls * selected_pixel_mask  * training_mask) + tf.reduce_sum(y_pred_cls * selected_pixel_mask * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    if type == 'classification':
        tf.summary.scalar('classification_dice_loss', loss)
    if type == 'border':
        tf.summary.scalar('border_dice_loss', loss)
    return loss

def focal_loss(y_true_cls, y_pred_cls, training_mask, type):
    #pos_mask = y_true_cls == 1
    #neg_mask = y_true_cls == 0

    #y_pred_cls = tf.nn.sigmoid(y_pred_cls)
    pos_mask = tf.cast(tf.equal(y_true_cls, 1), dtype=tf.float32)
    neg_mask = tf.cast(tf.equal(y_true_cls, 0), dtype=tf.float32)
    alpha = 0.25
    gama = 2
    eps = 1e-8
    
    pos_loss = tf.reduce_sum(-alpha*tf.pow((1-y_pred_cls), gama)*tf.log(y_pred_cls+eps) * pos_mask * training_mask)
    neg_loss = tf.reduce_sum(-(1-alpha)*tf.pow(y_pred_cls, gama)*tf.log(1-y_pred_cls+eps) * neg_mask * training_mask)

    print 'pos_loss: {}, neg_loss: {}'.format(pos_loss, neg_loss)
    loss = pos_loss + neg_loss
    '''
    alpha = 0.25
    gamma = 2
    eps = 1e-8
    sigmoid_p = tf.nn.sigmoid(y_pred_cls)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
     
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(y_true_cls > zeros, y_true_cls - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(y_true_cls > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, eps, 1.0))*training_mask - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, eps, 1.0))*training_mask
    loss = tf.reduce_sum(per_entry_cross_ent)
    '''
    if type == 'classification':
        tf.summary.scalar('classification_focal_loss', loss)
    if type == 'border':
        tf.summary.scalar('border_focal_loss', loss)

    return loss


def loss(y_true_cls, y_pred_cls,
         training_mask, y_true_brd, y_pred_brd):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    #classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask, 'classification')
    # scale classification loss to match the iou loss part
    #border_loss = dice_coefficient(y_true_brd, y_pred_brd, training_mask, 'border')

    classification_loss = dice_coefficient_OHNM(y_true_cls, y_pred_cls, training_mask, 'classification')
    # scale classification loss to match the iou loss part
    border_loss = dice_coefficient_OHNM(y_true_brd, y_pred_brd, training_mask, 'border')

    return classification_loss + border_loss * 2
