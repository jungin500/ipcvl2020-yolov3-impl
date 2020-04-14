from tensorflow.keras.initializers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import tensorflow as tf

GRID_CELL_SIZE = 7
CLASSES = 20

DEBUG_MODE = True


# TODO Refactoring class
class YoloLayer(Layer):
    def __init__(self, anchors, max_grid, batch_size, warmup_batches, ignore_thresh,
                 grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale,
                 **kwargs):
        # make the model settings persistent
        self.ignore_thresh = ignore_thresh
        self.warmup_batches = warmup_batches
        self.anchors = tf.constant(anchors, dtype='float', shape=[1, 1, 1, 3, 2])
        self.grid_scale = grid_scale
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.xywh_scale = xywh_scale
        self.class_scale = class_scale

        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.cast(dtype=tf.float32,
                         x=tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
        self.cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, 3, 1])

        super(YoloLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(YoloLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        input_image, y_pred, y_true, true_boxes = x

        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))

        # initialize the masks
        object_mask = tf.expand_dims(y_true[..., 4], 4)

        # the variable to keep track of number of batches processed
        batch_seen = tf.Variable(0.)

        # compute grid factor and net factor
        grid_h = tf.shape(y_true)[1]
        grid_w = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])

        net_h = tf.shape(input_image)[1]
        net_w = tf.shape(input_image)[2]
        net_factor = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1, 1, 1, 1, 2])

        """
        Adjust prediction
        """
        pred_box_xy = (self.cell_grid[:, :grid_h, :grid_w, :, :] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
        pred_box_wh = y_pred[..., 2:4]  # t_wh
        pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)  # adjust confidence
        pred_box_class = y_pred[..., 5:]  # adjust class probabilities

        """
        Adjust ground truth
        """
        true_box_xy = y_true[..., 0:2]  # (sigma(t_xy) + c_xy)
        true_box_wh = y_true[..., 2:4]  # t_wh
        true_box_conf = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Compare each predicted box to all true boxes
        """
        # initially, drag all objectness of all boxes to 0
        conf_delta = pred_box_conf - 0

        # then, ignore the boxes which have good overlap with some true box
        true_xy = true_boxes[..., 0:2] / grid_factor
        true_wh = true_boxes[..., 2:4] / net_factor

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)

        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_delta *= tf.expand_dims(tf.cast(dtype=tf.float32, x=best_ious < self.ignore_thresh), 4)

        """
        Compute some online statistics
        """
        true_xy = true_box_xy / grid_factor
        true_wh = tf.exp(true_box_wh) * self.anchors / net_factor

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = pred_box_xy / grid_factor
        pred_wh = tf.exp(pred_box_wh) * self.anchors / net_factor

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)
        iou_scores = object_mask * tf.expand_dims(iou_scores, 4)

        count = tf.reduce_sum(object_mask)
        count_noobj = tf.reduce_sum(1 - object_mask)
        detect_mask = tf.cast(dtype=tf.float32, x=(pred_box_conf * object_mask) >= 0.5)
        class_mask = tf.expand_dims(
            tf.cast(dtype=tf.float32, x=tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
        recall50 = tf.reduce_sum(tf.cast(dtype=tf.float32, x=iou_scores >= 0.5) * detect_mask * class_mask) / (
                    count + 1e-3)
        recall75 = tf.reduce_sum(tf.cast(dtype=tf.float32, x=iou_scores >= 0.75) * detect_mask * class_mask) / (
                    count + 1e-3)
        avg_iou = tf.reduce_sum(iou_scores) / (count + 1e-3)
        avg_obj = tf.reduce_sum(pred_box_conf * object_mask) / (count + 1e-3)
        avg_noobj = tf.reduce_sum(pred_box_conf * (1 - object_mask)) / (count_noobj + 1e-3)
        avg_cat = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3)

        """
        Warm-up training
        """
        batch_seen = batch_seen.assign_add(1.)

        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches + 1),
                                                      lambda: [true_box_xy + (
                                                              0.5 + self.cell_grid[:, :grid_h, :grid_w, :, :]) * (
                                                                       1 - object_mask),
                                                               true_box_wh + tf.zeros_like(true_box_wh) * (
                                                                       1 - object_mask),
                                                               tf.ones_like(object_mask)],
                                                      lambda: [true_box_xy,
                                                               true_box_wh,
                                                               object_mask])

        """
        Compare each true box to all anchor boxes
        """
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1],
                                  axis=4)  # the smaller the box, the bigger the scale

        xy_delta = xywh_mask * (pred_box_xy - true_box_xy) * wh_scale * self.xywh_scale
        wh_delta = xywh_mask * (pred_box_wh - true_box_wh) * wh_scale * self.xywh_scale
        conf_delta = object_mask * (pred_box_conf - true_box_conf) * self.obj_scale + (
                1 - object_mask) * conf_delta * self.noobj_scale
        class_delta = object_mask * \
                      tf.expand_dims(
                          tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class),
                          4) * \
                      self.class_scale

        loss_xy = tf.reduce_sum(tf.square(xy_delta), list(range(1, 5)))
        loss_wh = tf.reduce_sum(tf.square(wh_delta), list(range(1, 5)))
        loss_conf = tf.reduce_sum(tf.square(conf_delta), list(range(1, 5)))
        loss_class = tf.reduce_sum(class_delta, list(range(1, 5)))

        loss = loss_xy + loss_wh + loss_conf + loss_class

        # if DEBUG_MODE:
        #     tf.print('avg_obj \t\t{} {}'.format(grid_h, avg_obj))
        #     tf.print('avg_noobj \t\t{} {}'.format(grid_h, avg_noobj))
        #     tf.print('avg_iou \t\t{} {}'.format(grid_h, avg_iou))
        #     tf.print('avg_cat \t\t{} {}'.format(grid_h, avg_cat))
        #     tf.print('recall50 \t{} {}'.format(grid_h, recall50))
        #     tf.print('recall75 \t{} {}'.format(grid_h, recall75))
        #     tf.print('count \t{} {}'.format(grid_h, count))
        #     tf.print('loss xy, wh, conf, class: \t{} {} {} {} {}'.format(grid_h, tf.reduce_sum(loss_xy),
        #                                                                  tf.reduce_sum(loss_wh),
        #                                                                  tf.reduce_sum(loss_conf),
        #                                                                  tf.reduce_sum(loss_class)))

        return loss * self.grid_scale

    def compute_output_shape(self, input_shape):
        return [(None, 1)]


def Conv2DLRU(x, filters, kernel_size, strides=(1, 1), name=None):
    x = Conv2D(filters, kernel_size, strides, use_bias=False, padding='same',
               name=None if name is None else 'conv_' + name)(x)
    x = BatchNormalization(epsilon=0.001)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def ResidualIDBlock(X, filters, stage, block='a'):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2 = filters
    X_shortcut = X

    # first component path
    X = Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        name=conv_name_base + '2a',
        use_bias=None,
        kernel_initializer=glorot_uniform(seed=0)  # Xavier Uniform Initializer
    )(X)
    X = BatchNormalization(epsilon=0.001, axis=3, name=bn_name_base + '2a')(X)
    X = LeakyReLU(alpha=0.1)(X)

    # last component path
    X = Conv2D(
        filters=F2,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        name=conv_name_base + '2b',
        use_bias=None,
        kernel_initializer=glorot_uniform(seed=0)
    )(X)
    X = BatchNormalization(epsilon=0.001, axis=3, name=bn_name_base + '2b')(X)  # axis=3 (channel)?
    X = LeakyReLU(alpha=0.1)(X)

    # final step: shortcut to main path
    X = Add()([X, X_shortcut])
    return X


def YoloModel(
        nb_class,
        anchors,
        max_box_per_image,
        max_grid,
        batch_size,
        warmup_batches,
        ignore_thresh,
        grid_scales,
        obj_scale,
        noobj_scale,
        xywh_scale,
        class_scale
):
    # input = Input(shape=(256, 256, 3))
    input_image = Input(shape=(None, None, 3))  # net_h, net_w, 3
    true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))
    true_yolo_1 = Input(
        shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class))  # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_2 = Input(
        shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class))  # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_3 = Input(
        shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class))  # grid_h, grid_w, nb_anchor, 5+nb_class

    # L1 Outer
    x = Conv2DLRU(input_image, filters=32, kernel_size=3)
    x = Conv2DLRU(x, filters=64, kernel_size=3, strides=2)

    # L2 Residual (x1)
    x = ResidualIDBlock(x, filters=[32, 64], block='a', stage=2)

    # L3 Outer
    x = Conv2DLRU(x, filters=128, kernel_size=3, strides=2)

    # L4 Residual (x2)
    x = ResidualIDBlock(x, filters=[64, 128], block='a', stage=4)
    x = ResidualIDBlock(x, filters=[64, 128], block='b', stage=4)

    # L5 Outer
    x = Conv2DLRU(x, filters=256, kernel_size=3, strides=2)

    # L6 Residual (x8)
    x = ResidualIDBlock(x, filters=[128, 256], block='a', stage=6)
    x = ResidualIDBlock(x, filters=[128, 256], block='b', stage=6)
    x = ResidualIDBlock(x, filters=[128, 256], block='c', stage=6)
    x = ResidualIDBlock(x, filters=[128, 256], block='d', stage=6)
    x = ResidualIDBlock(x, filters=[128, 256], block='e', stage=6)
    x = ResidualIDBlock(x, filters=[128, 256], block='f', stage=6)
    x = ResidualIDBlock(x, filters=[128, 256], block='g', stage=6)
    x = ResidualIDBlock(x, filters=[128, 256], block='h', stage=6)

    skip_l6 = x

    # L7 Outer
    x = Conv2DLRU(x, filters=512, kernel_size=3, strides=2)

    # L8 Residual (x8)
    x = ResidualIDBlock(x, filters=[256, 512], block='a', stage=8)
    x = ResidualIDBlock(x, filters=[256, 512], block='b', stage=8)
    x = ResidualIDBlock(x, filters=[256, 512], block='c', stage=8)
    x = ResidualIDBlock(x, filters=[256, 512], block='d', stage=8)
    x = ResidualIDBlock(x, filters=[256, 512], block='e', stage=8)
    x = ResidualIDBlock(x, filters=[256, 512], block='f', stage=8)
    x = ResidualIDBlock(x, filters=[256, 512], block='g', stage=8)
    x = ResidualIDBlock(x, filters=[256, 512], block='h', stage=8)

    skip_l8 = x

    # L9 Outer
    x = Conv2DLRU(x, filters=1024, kernel_size=3, strides=2)

    # L10 Residual (x4), End of Feature Extractor (YOLOv3, Table 1, Darknet-53)
    x = ResidualIDBlock(x, filters=[512, 1024], block='a', stage=10)
    x = ResidualIDBlock(x, filters=[512, 1024], block='b', stage=10)
    x = ResidualIDBlock(x, filters=[512, 1024], block='c', stage=10)
    x = ResidualIDBlock(x, filters=[512, 1024], block='d', stage=10)

    # L11 Classifier #1 (Layer 75~)
    x = Conv2DLRU(x, filters=512, kernel_size=1)
    x = Conv2DLRU(x, filters=1024, kernel_size=3)
    x = Conv2DLRU(x, filters=512, kernel_size=1)
    x = Conv2DLRU(x, filters=1024, kernel_size=3)
    x = Conv2DLRU(x, filters=512, kernel_size=1)

    # L12 Sub-Branch Classifier #1 (Layer 80~)
    pred_yolo_1 = Conv2DLRU(x, filters=1024, kernel_size=3)
    pred_yolo_1 = Conv2D(filters=3 * (5 + CLASSES), kernel_size=1)(x)  # Conv2DLRU without BN and LRU
    loss_yolo_1 = YoloLayer(anchors[12:],
                            [1 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            grid_scales[0],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_1, true_yolo_1, true_boxes])

    # L13 Classifier #2 (Layer 83~)
    x = Conv2DLRU(x, filters=256, kernel_size=1)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_l8])

    # L14 Classifier #3 (Layer 87~)
    x = Conv2DLRU(x, filters=256, kernel_size=1)
    x = Conv2DLRU(x, filters=512, kernel_size=3)
    x = Conv2DLRU(x, filters=256, kernel_size=1)
    x = Conv2DLRU(x, filters=512, kernel_size=3)
    x = Conv2DLRU(x, filters=256, kernel_size=1)

    # L15 Sub-Branch Classifier #2 (Layer 92~)
    pred_yolo_2 = Conv2DLRU(x, filters=512, kernel_size=3)
    pred_yolo_2 = Conv2D(filters=3 * (5 + CLASSES), kernel_size=1)(pred_yolo_2)  # Conv2DLRU without BN and LRU
    loss_yolo_2 = YoloLayer(anchors[6:12],
                            [2 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            grid_scales[1],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_2, true_yolo_2, true_boxes])

    # L16 Classifier #4 (Layer 95~)
    x = Conv2DLRU(x, filters=128, kernel_size=1)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_l6])

    # L17 Sub-Branch Classifier #3 (Layer 99~)
    pred_yolo_3 = Conv2DLRU(x, filters=128, kernel_size=1)
    pred_yolo_3 = Conv2DLRU(pred_yolo_3, filters=256, kernel_size=1)
    pred_yolo_3 = Conv2DLRU(pred_yolo_3, filters=128, kernel_size=1)
    pred_yolo_3 = Conv2DLRU(pred_yolo_3, filters=256, kernel_size=1)
    pred_yolo_3 = Conv2DLRU(pred_yolo_3, filters=128, kernel_size=1)
    pred_yolo_3 = Conv2DLRU(pred_yolo_3, filters=256, kernel_size=1)
    pred_yolo_3 = Conv2D(filters=3 * (5 + CLASSES), kernel_size=1)(pred_yolo_3)  # Conv2DLRU without BN and LRU
    loss_yolo_3 = YoloLayer(anchors[:6],
                            [4 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            grid_scales[2],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_3, true_yolo_3, true_boxes])

    # L18 Output of Model
    # TODO OUTPUT 3개중 하나만 가지고 Test 해보기!
    train_model = Model(
        [input_image, true_boxes, true_yolo_1, true_yolo_2, true_yolo_3],
        [loss_yolo_1, loss_yolo_2, loss_yolo_3]
    )
    infer_model = Model(
        input_image,
        [pred_yolo_1, pred_yolo_2, pred_yolo_3]
    )

    return [train_model, infer_model]
