import tensorflow.keras.backend as K
import tensorflow as tf

import numpy as np
import sys

def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max

'''
    pred_mins, pred_maxes: shape(? * S * S * 3 * 2)
    true_mins, true_maxes: shape(? * S * S * 3 * 2)
    
    returns [? * S * S * 3]
'''
def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    # pred와 true 각각 최소값 중 큰값
    intersect_mins = K.maximum(pred_mins, true_mins)  # ? * S * S * 3 * 2
    
    # pred와 true 각각 최대값 중 작은값
    intersect_maxes = K.minimum(pred_maxes, true_maxes)  # ? * S * S * 3 * 2

    # wh는 max에서 min을 뺀 값(intersact_xmax - intersact_xmin = intersact_w, intersact_ymax - intersact_ymin = intersact_h)
    # wh -> intersact_wh
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)  # ? * S * S * 3 * 2

    # 그 공간에서 w와 h를 곱하면 그 공간의 면적
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]  # ? * S * S * 3
    
    # pred와 true 각각 wh -> area 순으로 구함
    pred_wh = pred_maxes - pred_mins  # ? * S * S * 3 * 2
    true_wh = true_maxes - true_mins  # ? * S * S * 3 * 2
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]  # ? * S * S * 3
    true_areas = true_wh[..., 0] * true_wh[..., 1]  # ? * S * S * 3

    # A union B = A + B - (A intersect B)
    union_areas = pred_areas + true_areas - intersect_areas  # ? * S * S * 3

    # IOU = Intersact of Union = intersact_area / union_area
    # 교집합이 합집합의 어느정도를 차지하는지를 의미
    # intersact가 전체인경우 -> 1
    # intersact가 전체보다 작은 경우 -> ( < 1 )
    # intersact가 없는 경우 -> 0
    iou_scores = intersect_areas / union_areas  # ? * S * S * 3

    return iou_scores  # ? * S * S * 3


'''
    box_xy, box_wh -> [B * S * S * 3 * 1 * 2]
'''
ANCHOR_BOXES = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
ANCHOR_BOXES_TF = tf.reshape(
    tf.constant(ANCHOR_BOXES, dtype=tf.float32),
    shape=(1, 1, 1, 3, 3, 2)
)
SCALES = [13, 26, 52]
# SCALES = [26]

# Returns 1 * S  * S * 1 * 1 * 2
def cell_offset_table(scale_size):
    # Dynamic implementation of conv dims for fully convolutional model.
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=scale_size)
    conv_width_index = K.arange(0, stop=scale_size)
    conv_height_index = K.tile(conv_height_index, [scale_size]) # 늘어놓는 함수  tile -> 같은걸 N번 반복함
    # 결과 -> 0~12, 0~12, ...., 0~12

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [scale_size, 1]) # tile을 [n, m] 쓰면 dims 2로 만들어줌
    # 결과 -> [0~12], [0~12], [0~12], ...

    conv_width_index = K.flatten(K.transpose(conv_width_index))
    # 결과 -> 0, 0, 0, 0, 0, 0, 0 (13개), 1, 1, 1, 1, 1, 1, 1 (13개), ...

    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    # 결과 -> [0, 0], [1, 0], [2, 0], ..., [11, 12], [12, 12]

    conv_index = K.reshape(conv_index, [1, scale_size, scale_size, 1, 2])
    # 결과 -> 1 * 13 * 13 에 있는 [1 * 2]의 conv index item이 만들어짐
    # 각각 [1 * 2]의 값은 [0, 0], [1, 0], [2, 0], ..., [11, 12], [12, 12]
    # 이런 식으로 이루어져 있음 -> Mask를 만들기 위한 과정
    # 결과 shape -> 1, 13, 13, 1, 2

    conv_index = K.cast(conv_index, tf.float32)

    diff = (1 / scale_size * 416)
    conv_index = conv_index * diff

    return conv_index


def CreateYolov3Loss(scale_size, scale_index):

    '''
    scale_size = 13 -> 32
    scale_size = 26 -> 16
    scale_size = 52 -> 8
    '''
    # cell_size = 32
    cell_size = 416 / scale_size

    # loss multiply
    scale_index_divider = scale_index + 1
    volume_loss_multiply = 1. / (scale_index_divider * scale_index_divider)  # /1, /4, /9

    def __loss__(y_true, y_pred):
        label_class = y_true[..., :20]  # ? * S * S * 3 * 20
        label_box = y_true[..., 20:24]  # ? * S * S * 3 * 4
        responsible_mask = y_true[..., 24]  # ? * S * S * 3

        predict_class = y_pred[..., :20]  # ? * S * S * 3 * 20
        predict_box = y_pred[..., 20:24]  # ? * S * S * 3 * 4
        predict_bbox_confidences = y_pred[..., 24]  # ? * S * S * 3

        label_bxby, label_bwbh = label_box[..., :2], label_box[..., 2:4]  # ? * S * S * 3 * 2, ? * S * S * 3 * 2
        label_bxby_ext = tf.expand_dims(label_bxby, 4)  # ? * S * S * 3 * 1 * 2
        label_bwbh_ext = tf.expand_dims(label_bwbh, 4)  # ? * S * S * 3 * 1 * 2
        label_bxby_min, label_bxby_max = xywh2minmax(label_bxby_ext,
                                                     label_bwbh_ext)  # ? * S * S * 3 * 1 * 2, ? * S * S * 3 * 1 * 2

        predict_txty, predict_twth = predict_box[..., :2], predict_box[...,
                                                           2:4]  # ? * S * S * 3 * 2, ? * S * S * 3 * 2
        predict_bxby = cell_offset_table(scale_size) + (tf.sigmoid(predict_txty) * cell_size)  # ? * S * S * 3 * 2
        predict_bwbh = tf.math.exp(predict_twth) * ANCHOR_BOXES_TF[:, :, :, scale_index, :, :]  # ? * S * S * 3 * 2
        predict_bxby_ext = tf.expand_dims(predict_bxby, 4)  # ? * S * S * 3 * 1 * 2
        predict_bwbh_ext = tf.expand_dims(predict_bwbh, 4)  # ? * S * S * 3 * 1 * 2
        predict_bxby_min, predict_bxby_max = xywh2minmax(predict_bxby_ext,
                                                         predict_bwbh_ext)  # ? * S * S * 3 * 1 * 2, ? * S * S * 3 * 1 * 2

        iou_scores = iou(predict_bxby_min, predict_bxby_max, label_bxby_min, label_bxby_max)  # ? * S * S * 3 * 1
        iou_scores = iou_scores[..., 0]  # iou_scores의 마지막에 * 1은 의미가 없다... 떼어주자, ? * S * S * 3

        # 가장 큰 IOU를 가지는 Anchor 1개만 1이고,
        # 나머지 Anchor 2개는 0이다.

        # 문제점은, 만일 IOU가 모든 앵커에 대해 같은 수라면, Anchor가 2개 이상 골라질 수도 있다는 점이다.
        # Initializer가 알아서 잘 해주겠지 뭐....
        anchor_mask = K.cast(iou_scores >= K.max(iou_scores, axis=3, keepdims=True), K.dtype(iou_scores)) # ? * S * S * 3

        # label_xy, label_wh = label_box[..., 2:], label_box[..., 2:4]   # 각 ? * S * S * 3 * 2
        # predict_xy, predict_wh = predict_box[..., 2:], predict_box[..., 2:4]  # 각 ? * S * S * 3 * 2

        anchor_mask = K.expand_dims(anchor_mask)  # ? * S * S * 3 * 1, 나중에 곱하기 위해
        responsible_mask = K.expand_dims(responsible_mask)  # ? * S * S * 3 * 1, 나중에 곱하기 위해
        # prior_mask_ext = K.expand_dims(prior_mask)  # ? * S * S * 3 * 1 * 1
        # responsible_mask_ext = K.expand_dims(responsible_mask)  # ? * S * S * 3 * 1 * 1

        # Loss 함수 1번
        # xy_loss = 5 * anchor_mask * responsible_mask * K.square((label_bxby / 416) - (predict_bxby / 416))
        xy_loss = 5 * anchor_mask * responsible_mask * K.square(label_bxby / cell_size - predict_bxby / cell_size)
        box_loss = xy_loss

        # tf.print("\n- xy loss:", tf.reduce_sum(xy_loss), output_stream=sys.stdout)

        # Loss 함수 2번
        wh_loss = 5 * anchor_mask * responsible_mask * K.square(K.sqrt(label_bwbh / cell_size) - K.sqrt(predict_bwbh / cell_size))
        box_loss += wh_loss

        # tf.print("- wh loss:", tf.reduce_sum(wh_loss), output_stream=sys.stdout)

        # 1번+2번 총합
        box_loss = K.sum(box_loss) * volume_loss_multiply

        predict_bbox_confidences = K.expand_dims(predict_bbox_confidences)

        # Loss 함수 3번 (without lambda_noobj)
        object_loss = anchor_mask * responsible_mask * K.square(1 - predict_bbox_confidences)
        # Loss 함수 4번 (with lambda_noobj 0.5)
        no_object_loss = 0.5 * (1 - anchor_mask * responsible_mask) * K.square(0 - predict_bbox_confidences)

        tf.print("\n- [%d] no_object_loss:" % scale_index, tf.reduce_sum(no_object_loss), output_stream=sys.stdout)
        tf.print("- [%d] object_loss:" % scale_index, tf.reduce_sum(object_loss), output_stream=sys.stdout)

        confidence_loss = no_object_loss + object_loss
        # confidence_loss = K.sum(confidence_loss) * volume_loss_multiply
        confidence_loss = K.sum(confidence_loss) * volume_loss_multiply

        # Loss 함수 5번
        # class_loss = responsible_mask * K.square(label_class - predict_class)
        class_loss = responsible_mask * K.square(label_class - (predict_bbox_confidences * predict_class))
        # class_loss = responsible_mask * K.square(iou_scores)

        # Loss 함수 5번 총합
        class_loss = K.sum(class_loss) * volume_loss_multiply

        # loss = box_loss + confidence_loss + class_loss
        loss = box_loss + confidence_loss + class_loss

        tf.print("- [%d] confidence_loss:" % scale_index, confidence_loss, output_stream=sys.stdout)
        tf.print("- [%d] class_loss:" % scale_index, class_loss, output_stream=sys.stdout)
        tf.print("- [%d] box_loss:" % scale_index, box_loss, output_stream=sys.stdout)

        return loss

    def __loss_dev__(y_true, y_pred):
        label_class = y_true[..., :20]  # ? * S * S * 3 * 20
        label_box = y_true[..., 20:24]  # ? * S * S * 3 * 4
        responsible_mask = y_true[..., 24]  # ? * S * S * 3

        predict_class = y_pred[..., :20]  # ? * S * S * 3 * 20
        predict_box = y_pred[..., 20:24]  # ? * S * S * 3 * 4
        predict_bbox_confidences = y_pred[..., 24]  # ? * S * S * 3

        label_bxby, label_bwbh = label_box[..., :2], label_box[..., 2:4]  # ? * S * S * 3 * 2, ? * S * S * 3 * 2
        label_bxby_ext = tf.expand_dims(label_bxby, 4)  # ? * S * S * 3 * 1 * 2
        label_bwbh_ext = tf.expand_dims(label_bwbh, 4)  # ? * S * S * 3 * 1 * 2
        label_bxby_min, label_bxby_max = xywh2minmax(label_bxby_ext,
                                                     label_bwbh_ext)  # ? * S * S * 3 * 1 * 2, ? * S * S * 3 * 1 * 2

        predict_txty, predict_twth = predict_box[..., :2], predict_box[...,
                                                           2:4]  # ? * S * S * 3 * 2, ? * S * S * 3 * 2
        predict_bxby = cell_offset_table(scale_size) + (tf.sigmoid(predict_txty) * cell_size)  # ? * S * S * 3 * 2
        predict_bwbh = tf.math.exp(predict_twth) * ANCHOR_BOXES_TF[:, :, :, scale_index, :, :]  # ? * S * S * 3 * 2
        predict_bxby_ext = tf.expand_dims(predict_bxby, 4)  # ? * S * S * 3 * 1 * 2
        predict_bwbh_ext = tf.expand_dims(predict_bwbh, 4)  # ? * S * S * 3 * 1 * 2
        predict_bxby_min, predict_bxby_max = xywh2minmax(predict_bxby_ext,
                                                         predict_bwbh_ext)  # ? * S * S * 3 * 1 * 2, ? * S * S * 3 * 1 * 2

        iou_scores = iou(predict_bxby_min, predict_bxby_max, label_bxby_min, label_bxby_max)  # ? * S * S * 3 * 1
        iou_scores = iou_scores[..., 0]  # iou_scores의 마지막에 * 1은 의미가 없다... 떼어주자, ? * S * S * 3

        # 가장 큰 IOU를 가지는 Anchor 1개만 1이고,
        # 나머지 Anchor 2개는 0이다.

        # 문제점은, 만일 IOU가 모든 앵커에 대해 같은 수라면, Anchor가 2개 이상 골라질 수도 있다는 점이다.
        # Initializer가 알아서 잘 해주겠지 뭐....
        anchor_mask = K.cast(iou_scores >= K.max(iou_scores, axis=3, keepdims=True), K.dtype(iou_scores)) # ? * S * S * 3

        # label_xy, label_wh = label_box[..., 2:], label_box[..., 2:4]   # 각 ? * S * S * 3 * 2
        # predict_xy, predict_wh = predict_box[..., 2:], predict_box[..., 2:4]  # 각 ? * S * S * 3 * 2

        anchor_mask = K.expand_dims(anchor_mask)  # ? * S * S * 3 * 1, 나중에 곱하기 위해
        responsible_mask = K.expand_dims(responsible_mask)  # ? * S * S * 3 * 1, 나중에 곱하기 위해
        # prior_mask_ext = K.expand_dims(prior_mask)  # ? * S * S * 3 * 1 * 1
        # responsible_mask_ext = K.expand_dims(responsible_mask)  # ? * S * S * 3 * 1 * 1

        # Loss 함수 1번
        # xy_loss = 5 * anchor_mask * responsible_mask * K.square((label_bxby / 416) - (predict_bxby / 416))
        xy_loss = 5 * anchor_mask * responsible_mask * K.square(label_bxby / cell_size - predict_bxby / cell_size)
        box_loss = xy_loss

        # tf.print("\n- xy loss:", tf.reduce_sum(xy_loss), output_stream=sys.stdout)

        # Loss 함수 2번
        wh_loss = 5 * anchor_mask * responsible_mask * K.square(K.sqrt(label_bwbh / cell_size) - K.sqrt(predict_bwbh / cell_size))
        box_loss += wh_loss

        # tf.print("- wh loss:", tf.reduce_sum(wh_loss), output_stream=sys.stdout)

        # 1번+2번 총합
        box_loss = K.sum(box_loss) * volume_loss_multiply

        predict_bbox_confidences = K.expand_dims(predict_bbox_confidences)

        # Loss 함수 3번 (without lambda_noobj)
        object_loss = anchor_mask * responsible_mask * K.square(1 - predict_bbox_confidences)
        # Loss 함수 4번 (with lambda_noobj 0.5)
        no_object_loss = 0.5 * (1 - anchor_mask * responsible_mask) * K.square(0 - predict_bbox_confidences)

        # tf.print("- no_object_loss:", tf.reduce_sum(no_object_loss), output_stream=sys.stdout)
        # tf.print("- object_loss:", tf.reduce_sum(object_loss), output_stream=sys.stdout)

        confidence_loss = no_object_loss + object_loss
        # confidence_loss = K.sum(confidence_loss) * volume_loss_multiply
        confidence_loss = K.sum(confidence_loss) * volume_loss_multiply

        # Loss 함수 5번
        # class_loss = responsible_mask * K.square(label_class - predict_class)
        class_loss = responsible_mask * K.square(label_class - (predict_bbox_confidences * predict_class))
        # class_loss = responsible_mask * K.square(iou_scores)

        # Loss 함수 5번 총합
        class_loss = K.sum(class_loss) * volume_loss_multiply

        loss = box_loss + confidence_loss + class_loss
        # loss = box_loss + confidence_loss

        tf.print("\n- [%d] confidence_loss:" % scale_index, confidence_loss, output_stream=sys.stdout)
        tf.print("- [%d] class_loss:" % scale_index, class_loss, output_stream=sys.stdout)
        tf.print("- [%d] box_loss:" % scale_index, box_loss, output_stream=sys.stdout)

        return loss

    def __dummy__(y_true, y_pred):
        tf.print("y_true shape: ", tf.shape(y_true))
        tf.print("y_pred shape: ", tf.shape(y_pred))

        return tf.reduce_mean(y_pred)

    return __loss_dev__
    # return __dummy__

