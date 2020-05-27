import tensorflow as tf
from tensorflow.keras.metrics import Metric
from YoloLoss import generate_index_matrix, get_anchor_box

class YoloMetric(Metric):
    def __init__(self, name='yolo_metric', **kwargs):
        super(YoloMetric, self).__init__(name=name)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')

        self.precision = self.add_weight(name='precision', initializer='zeros')
        self.recall = self.add_weight(name='recall', initializer='zeros')
        self.accuracy = self.add_weight(name='accuracy', initializer='zeros')

        self.thresh_confidance = 0.5  # Cell existance confidance threshold
        self.thresh_iou = 0.5  # IOU threshold
        self.epsilon = 1e-15
        self.verbose = kwargs.get('verbose') if kwargs.get('verbose') is not None else False

    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
        간단한 Precision Metric
        Scale마다 한번씩 돌며 해당 Scale의 Precision을 계산한다.
        y_true, y_pred 각각 [B * S * S * 3 * 25]의 shape를 갖는다.

        :param y_true: 정답값
        :param y_pred: 예측값
        :return: 예측된 Precision 값
        '''

        pred_class = y_pred[..., :20]
        pred_bbox = y_pred[..., 20:24]
        pred_conf = y_pred[..., 24:25]
        pred_conf_nonexpand = y_pred[..., 24]

        label_class = y_true[..., :20]
        label_bbox = y_true[..., 20:24]
        label_conf = y_true[..., 24:25]
        label_conf_nonexpand = y_true[..., 24]

        # 인덱스 매트릭스를 만든다
        # (cell-based가 아닌 global-based bbox coordinate를 구하기 위해)
        scale_size = tf.cast(tf.shape(y_true)[1], tf.float32)  # B * S * S * 3 * 25에서 S!
        scale_index = tf.cast(scale_size / 13 / 2, tf.int32)  # 13, 26, 52 -> 0, 1, 2
        anchor_box = get_anchor_box(scale_index)
        image_size = 416

        index_matrix = generate_index_matrix(scale_size)

        pred_bbox_xy_glr = (tf.sigmoid(pred_bbox[..., :2]) + index_matrix) / scale_size  # normalized from 0.0 to 1.0
        pred_bbox_wh_glr = (anchor_box * tf.exp(pred_bbox[..., 2:])) / image_size

        label_bbox_xy_glr = (label_bbox[..., :2] + index_matrix) / scale_size
        label_bbox_wh = label_bbox[..., 2:]

        pred_bbox_xy_min = pred_bbox_xy_glr - (pred_bbox_wh_glr / 2)
        pred_bbox_xy_max = pred_bbox_xy_glr + (pred_bbox_wh_glr / 2)
        pred_volume = pred_bbox_wh_glr[..., 0] * pred_bbox_wh_glr[..., 1]

        label_bbox_xy_min = label_bbox_xy_glr - (label_bbox_wh / 2)
        label_bbox_xy_max = label_bbox_xy_glr + (label_bbox_wh / 2)
        label_volume = label_bbox_wh[..., 0] * label_bbox_wh[..., 1]

        intersact_min = tf.maximum(pred_bbox_xy_min, label_bbox_xy_min)
        intersact_max = tf.minimum(pred_bbox_xy_max, label_bbox_xy_max)
        intersact_wh = tf.maximum(intersact_max - intersact_min, 0)  # clip to zero
        intersact_volume = intersact_wh[..., 0] * intersact_wh[..., 1]

        union_volume = pred_volume + label_volume - intersact_volume

        # 검출된 모든 셀의 모든 IoU를 계산한 값이다.
        # 이 값들 중, TP, FP를 구해서 "Precision Metric"을 구한다.
        iou_volume = intersact_volume / union_volume

        # Metric 계산

        # IoU (BBox)
        pred_gt_iou_match_mask = iou_volume >= self.thresh_iou
        # pred_gt_iou_volume = iou_volume

        # Class
        pred_gt_class_match_mask = tf.argmax(pred_class, axis=4) == tf.argmax(label_class, axis=4)
        # pred_class_value = tf.argmax(pred_class, axis=4)
        # label_class_value = tf.argmax(label_class, axis=4)

        # Confidance
        label_conf_obj_mask = label_conf_nonexpand == 1.
        pred_conf_obj_mask = pred_conf_nonexpand >= self.thresh_confidance

        '''
        TP (True Positive) 계산
        True Positive는, Network의 결과가 Positive(참)인데, 이는 정답(True)인 경우이다. 맞는 경우이다.
        '''
        true_positive = pred_conf_obj_mask & label_conf_obj_mask & pred_gt_iou_match_mask & pred_gt_class_match_mask  # logical and
        true_positive = tf.reduce_sum(tf.cast(true_positive, tf.float32))

        '''
        FP (False Positive) 계산
        False Positive는, Network의 결과가 Positive(참)인데, 이는 거짓(False)인 경우이다. 틀린 경우이다.
        '''
        # false_positive = pred_conf_obj_mask * (1. - label_conf_obj_mask * pred_gt_iou_match_mask * pred_gt_class_match_mask)
        false_positive = pred_conf_obj_mask & ~(label_conf_obj_mask & pred_gt_iou_match_mask & pred_gt_class_match_mask)
        false_positive = tf.reduce_sum(tf.cast(false_positive, tf.float32))

        '''
        FN (False Negative) 계산
        False Negative는, Network의 결과가 Negative(거짓)인데, 이는 거짓(False)인 경우이다. 틀린 경우이다.
        '''
        # false_negative = ~(pred_conf_obj_mask & pred_gt_iou_match_mask & pred_gt_class_match_mask) & label_conf_obj_mask
        false_negative = ~pred_conf_obj_mask & label_conf_obj_mask
        false_negative = tf.reduce_sum(tf.cast(false_negative, tf.float32))

        '''
        TN (True Negative) 계산
        True Negative는, Network의 결과가 Negative(거짓)인데, 이가 참(True)인 경우이다. 맞는 경우이다.
        '''
        # true_negative = (1. - pred_conf_obj_mask) * (1. - label_conf_obj_mask) * (1. - pred_gt_iou_match_mask * pred_gt_class_match_mask)
        true_negative = ~pred_conf_obj_mask & ~label_conf_obj_mask
        true_negative = tf.reduce_sum(tf.cast(true_negative, tf.float32))

        if self.verbose:
            tf.print("\ntrue_positive:", true_positive)
            tf.print("true_negative:", true_negative)
            tf.print("false_positive:", false_positive)
            tf.print("false_negative:", false_negative)

        self.true_positives.assign_add(true_positive)
        self.false_positives.assign_add(false_positive)
        self.false_negatives.assign_add(false_negative)
        self.true_negatives.assign_add(true_negative)

    def result(self):
        # Precision
        self.precision.assign((self.true_positives + self.epsilon) / (self.true_positives + self.false_positives + self.epsilon))

        # Recall
        self.recall.assign((self.true_positives + self.epsilon) / (self.true_positives + self.false_negatives + self.epsilon))

        # Accuracy
        self.accuracy.assign((self.true_positives + self.true_negatives + self.epsilon) / (
            self.true_positives + self.true_negatives + self.false_positives + self.false_negatives + self.epsilon
        ))

        return self.accuracy

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)
        self.true_negatives.assign(0)

        self.precision.assign(0)
        self.recall.assign(0)
        self.accuracy.assign(0)