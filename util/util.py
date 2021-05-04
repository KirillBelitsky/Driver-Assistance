import numpy as np
import cv2
import pickle
import tensorflow as tf
from train.trainConfig import cfg

class Util:

    @staticmethod
    def RGB2HSL(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    @staticmethod
    def get_l_s_channels_from_HSL(image):
        return image[:, :, 1], image[:, :, 2]

    @staticmethod
    def get_random_RGB_colors(colorsAmount=100):
        np.random.seed(42)
        return np.random.randint(0, 255, size=(colorsAmount, 3), dtype="uint8")

    @staticmethod
    def read_classes(inputPath):
        with open(inputPath) as file:
            classNames = file.readlines()
        return [c.strip() for c in classNames]

    @staticmethod
    def get_calibration_properties(inputPath):
        calibration_data = pickle.load(open(inputPath, "rb"))
        matrix = calibration_data['camera_matrix']
        distortion_coef = calibration_data['distortion_coefficient']
        return matrix, distortion_coef

    @staticmethod
    def scale_abs(x, m=255):
        x = np.absolute(x)
        x = np.uint8(m * x / np.max(x))
        return x

    @staticmethod
    def roi(gray, mn=125, mx=1200):
        m = np.copy(gray) + 1
        m[:, :mn] = 0
        m[:, mx:] = 0
        return m

    @staticmethod
    def freeze_all(model, frozen=True):
        model.trainable = not frozen
        if isinstance(model, tf.keras.Model):
            for l in model.layers:
                Util.freeze_all(l, frozen)

    @staticmethod
    def unfreeze_all(model, frozen=False):
        model.trainable = not frozen
        if isinstance(model, tf.keras.Model):
            for l in model.layers:
                Util.unfreeze_all(l, frozen)

    @staticmethod
    def load_config(FLAGS):
        if FLAGS.tiny:
            STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
            ANCHORS = Util.get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
            XYSCALE = [1, 1]
        else:
            STRIDES = np.array(cfg.YOLO.STRIDES)
            ANCHORS = Util.get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
            XYSCALE = [1, 1, 1]
        NUM_CLASS = len(Util.read_class_names(cfg.YOLO.CLASSES))

        return STRIDES, ANCHORS, NUM_CLASS, XYSCALE

    @staticmethod
    def read_class_names(class_file_name):
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    @staticmethod
    def get_anchors(anchors_path, tiny=False):
        anchors = np.array(anchors_path)
        if tiny:
            return anchors.reshape(2, 3, 2)
        else:
            return anchors.reshape(3, 3, 2)

    @staticmethod
    def load_freeze_layer(tiny=False):
        if tiny:
            freeze_layouts = ['conv2d_9', 'conv2d_12']
        else:
            freeze_layouts = ['conv2d_58', 'conv2d_66', 'conv2d_74']

        return freeze_layouts

    @staticmethod
    def bbox_iou(bboxes1, bboxes2):
        """
        @param bboxes1: (a, b, ..., 4)
        @param bboxes2: (A, B, ..., 4)
            x:X is 1:n or n:n or n:1
        @return (max(a,A), max(b,B), ...)
        ex) (4,):(3,4) -> (3,)
            (2,1,4):(2,3,4) -> (2,3)
        """
        bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
        bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

        bboxes1_coor = tf.concat(
            [
                bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
                bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
            ],
            axis=-1,
        )
        bboxes2_coor = tf.concat(
            [
                bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
                bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
            ],
            axis=-1,
        )

        left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
        right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        union_area = bboxes1_area + bboxes2_area - inter_area

        iou = tf.math.divide_no_nan(inter_area, union_area)

        return iou

    @staticmethod
    def bbox_giou(bboxes1, bboxes2):
        """
        Generalized IoU
        @param bboxes1: (a, b, ..., 4)
        @param bboxes2: (A, B, ..., 4)
            x:X is 1:n or n:n or n:1
        @return (max(a,A), max(b,B), ...)
        ex) (4,):(3,4) -> (3,)
            (2,1,4):(2,3,4) -> (2,3)
        """
        bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
        bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

        bboxes1_coor = tf.concat(
            [
                bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
                bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
            ],
            axis=-1,
        )
        bboxes2_coor = tf.concat(
            [
                bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
                bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
            ],
            axis=-1,
        )

        left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
        right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        union_area = bboxes1_area + bboxes2_area - inter_area

        iou = tf.math.divide_no_nan(inter_area, union_area)

        enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
        enclose_right_down = tf.maximum(
            bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
        )

        enclose_section = enclose_right_down - enclose_left_up
        enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

        giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)

        return giou

    @staticmethod
    def bbox_ciou(bboxes1, bboxes2):
        """
        Complete IoU
        @param bboxes1: (a, b, ..., 4)
        @param bboxes2: (A, B, ..., 4)
            x:X is 1:n or n:n or n:1
        @return (max(a,A), max(b,B), ...)
        ex) (4,):(3,4) -> (3,)
            (2,1,4):(2,3,4) -> (2,3)
        """
        bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
        bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

        bboxes1_coor = tf.concat(
            [
                bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
                bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
            ],
            axis=-1,
        )
        bboxes2_coor = tf.concat(
            [
                bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
                bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
            ],
            axis=-1,
        )

        left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
        right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        union_area = bboxes1_area + bboxes2_area - inter_area

        iou = tf.math.divide_no_nan(inter_area, union_area)

        enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
        enclose_right_down = tf.maximum(
            bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
        )

        enclose_section = enclose_right_down - enclose_left_up

        c_2 = enclose_section[..., 0] ** 2 + enclose_section[..., 1] ** 2

        center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]

        rho_2 = center_diagonal[..., 0] ** 2 + center_diagonal[..., 1] ** 2

        diou = iou - tf.math.divide_no_nan(rho_2, c_2)

        v = (
                    (
                            tf.math.atan(
                                tf.math.divide_no_nan(bboxes1[..., 2], bboxes1[..., 3])
                            )
                            - tf.math.atan(
                        tf.math.divide_no_nan(bboxes2[..., 2], bboxes2[..., 3])
                    )
                    )
                    * 2
                    / np.pi
            ) ** 2

        alpha = tf.math.divide_no_nan(v, 1 - iou + v)

        ciou = diou - alpha * v

        return ciou

    @staticmethod
    def image_preprocess(image, target_size, gt_boxes=None):

        ih, iw = target_size
        h, w, _ = image.shape

        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
        image_paded = image_paded / 255.

        if gt_boxes is None:
            return image_paded

        else:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return image_paded, gt_boxes

    @staticmethod
    def load_weights(model, weights_file, is_tiny=False):
        if is_tiny:
            layer_size = 13
            output_pos = [9, 12]
        else:
            layer_size = 75
            output_pos = [58, 66, 74]

        wf = open(weights_file, 'rb')
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        j = 0
        for i in range(layer_size):
            conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
            bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if i not in output_pos:
                # darknet weights: [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                # tf weights: [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if i not in output_pos:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        # assert len(wf.read()) == 0, 'failed to read all data'
        wf.close()