import copy
import numpy as np
import tensorflow as tf
from train.operations import Operations
from util.util import Util
from train.trainConfig import cfg

class YoloTrainModel:

    def __init__(self, input_layer, amount_classes, FLAGS):
        self.input_layer = input_layer
        self.amount_classes = amount_classes

        strides, anchors, NUM_CLASS, XYSCALE = Util.load_config(FLAGS)
        self.train_settings = {
            'strides': strides,
            'anchors': anchors,
            'NUM_CLASS': NUM_CLASS,
            'XYSCALE': XYSCALE,
            'IOU_LOSS_THRESH': cfg.YOLO.IOU_LOSS_THRESH,
        }

    def create_full_model(self):
        route_1, route_2, conv = self.__darknet53()

        conv = Operations.convolutional(conv, (1, 1, 1024, 512))
        conv = Operations.convolutional(conv, (3, 3, 512, 1024))
        conv = Operations.convolutional(conv, (1, 1, 1024, 512))
        conv = Operations.convolutional(conv, (3, 3, 512, 1024))
        conv = Operations.convolutional(conv, (1, 1, 1024, 512))

        conv_lobj_branch = Operations.convolutional(conv, (3, 3, 512, 1024))
        conv_lbbox = Operations.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (self.amount_classes + 5)), activate=False, bn=False)

        conv = Operations.convolutional(conv, (1, 1, 512, 256))
        conv = Operations.upsample(conv)

        conv = tf.concat([conv, route_2], axis=-1)

        conv = Operations.convolutional(conv, (1, 1, 768, 256))
        conv = Operations.convolutional(conv, (3, 3, 256, 512))
        conv = Operations.convolutional(conv, (1, 1, 512, 256))
        conv = Operations.convolutional(conv, (3, 3, 256, 512))
        conv = Operations.convolutional(conv, (1, 1, 512, 256))

        conv_mobj_branch = Operations.convolutional(conv, (3, 3, 256, 512))
        conv_mbbox = Operations.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (self.amount_classes + 5)), activate=False, bn=False)

        conv = Operations.convolutional(conv, (1, 1, 256, 128))
        conv = Operations.upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)

        conv = Operations.convolutional(conv, (1, 1, 384, 128))
        conv = Operations.convolutional(conv, (3, 3, 128, 256))
        conv = Operations.convolutional(conv, (1, 1, 256, 128))
        conv = Operations.convolutional(conv, (3, 3, 128, 256))
        conv = Operations.convolutional(conv, (1, 1, 256, 128))

        conv_sobj_branch = Operations.convolutional(conv, (3, 3, 128, 256))
        conv_sbbox = Operations.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (self.amount_classes + 5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]

    def create_tiny_model(self):
        route_1, conv = self.__darknet53_tiny()

        conv = Operations.convolutional(conv, (1, 1, 1024, 256))

        conv_lobj_branch = Operations.convolutional(conv, (3, 3, 256, 512))
        conv_lbbox = Operations.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (self.amount_classes + 5)), activate=False, bn=False)

        conv = Operations.convolutional(conv, (1, 1, 256, 128))
        conv = Operations.upsample(conv)
        conv = tf.concat([conv, route_1], axis=-1)

        conv_mobj_branch = Operations.convolutional(conv, (3, 3, 128, 256))
        conv_mbbox = Operations.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (self.amount_classes + 5)), activate=False, bn=False)

        return [conv_mbbox, conv_lbbox]


    def decode_train(self, conv_output, output_size, i=0):
        conv_output = tf.reshape(conv_output,
                                 (tf.shape(conv_output)[0], output_size, output_size, 3, 5 + self.amount_classes))

        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, self.amount_classes),
                                                                              axis=-1)

        xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = ((tf.sigmoid(conv_raw_dxdy) * self.train_settings['XYSCALE'][i]) - 0.5 * (self.train_settings['XYSCALE'][i] - 1) + xy_grid) * \
                  self.train_settings['strides'][i]
        pred_wh = (tf.exp(conv_raw_dwdh) * self.train_settings['anchors'][i])
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def decode_tf(self, conv_output, output_size, i=0):
        batch_size = tf.shape(conv_output)[0]
        conv_output = tf.reshape(conv_output,
                                 (batch_size, output_size, output_size, 3, 5 + self.amount_classes))

        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, self.amount_classes),
                                                                              axis=-1)

        xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = ((tf.sigmoid(conv_raw_dxdy) * self.train_settings['XYSCALE'][i]) - 0.5 * (self.train_settings['XYSCALE'][i] - 1) + xy_grid) * \
                  self.train_settings['strides'][i]
        pred_wh = (tf.exp(conv_raw_dwdh) * self.train_settings['anchors'][i])
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        pred_prob = pred_conf * pred_prob
        pred_prob = tf.reshape(pred_prob, (batch_size, -1, self.amount_classes))
        pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

        return pred_xywh, pred_prob
        # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def filter_boxes(self, box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
        scores_max = tf.math.reduce_max(scores, axis=-1)

        mask = scores_max >= score_threshold
        class_boxes = tf.boolean_mask(box_xywh, mask)
        pred_conf = tf.boolean_mask(scores, mask)
        class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
        pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

        box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

        input_shape = tf.cast(input_shape, dtype=tf.float32)

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        box_mins = (box_yx - (box_hw / 2.)) / input_shape
        box_maxes = (box_yx + (box_hw / 2.)) / input_shape
        boxes = tf.concat([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)
        # return tf.concat([boxes, pred_conf], axis=-1)
        return (boxes, pred_conf)


    def compute_loss(self, pred, conv, label, bboxes, i=0):
        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = self.train_settings['strides'][i] * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + self.amount_classes))

        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]

        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]
        label_prob    = label[:, :, :, :, 5:]

        giou = tf.expand_dims(Util.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

        iou = Util.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < self.train_settings['IOU_LOSS_THRESH'], tf.float32)

        conf_focal = tf.pow(respond_bbox - pred_conf, 2)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

        return giou_loss, conf_loss, prob_loss

    def __darknet53(self):
        input_data = self.input_layer

        input_data = Operations.convolutional(input_data, (3, 3, 3, 32))
        input_data = Operations.convolutional(input_data, (3, 3, 32, 64), downsample=True)

        for i in range(1):
            input_data = Operations.residual_block(input_data, 64, 32, 64)

        input_data = Operations.convolutional(input_data, (3, 3, 64, 128), downsample=True)

        for i in range(2):
            input_data = Operations.residual_block(input_data, 128, 64, 128)

        input_data = Operations.convolutional(input_data, (3, 3, 128, 256), downsample=True)

        for i in range(8):
            input_data = Operations.residual_block(input_data, 256, 128, 256)

        route_1 = input_data
        input_data = Operations.convolutional(input_data, (3, 3, 256, 512), downsample=True)

        for i in range(8):
            input_data = Operations.residual_block(input_data, 512, 256, 512)

        route_2 = input_data
        input_data = Operations.convolutional(input_data, (3, 3, 512, 1024), downsample=True)

        for i in range(4):
            input_data = Operations.residual_block(input_data, 1024, 512, 1024)

        return route_1, route_2, input_data

    def __darknet53_tiny(self):
        input_data = self.input_layer

        input_data = Operations.convolutional(input_data, (3, 3, 3, 16))
        input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
        input_data = Operations.convolutional(input_data, (3, 3, 16, 32))
        input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
        input_data = Operations.convolutional(input_data, (3, 3, 32, 64))
        input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
        input_data = Operations.convolutional(input_data, (3, 3, 64, 128))
        input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
        input_data = Operations.convolutional(input_data, (3, 3, 128, 256))
        route_1 = input_data
        input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
        input_data = Operations.convolutional(input_data, (3, 3, 256, 512))
        input_data = tf.keras.layers.MaxPool2D(2, 1, 'same')(input_data)
        input_data = Operations.convolutional(input_data, (3, 3, 512, 1024))

        return route_1, input_data