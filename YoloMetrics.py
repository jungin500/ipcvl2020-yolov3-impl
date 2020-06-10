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
        self.verbose = False

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
        batch_size = tf.shape(y_pred)[0]
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
        iou_volume = intersact_volume / union_volume  # B * S * S * 3
        max_iou_volume = tf.reduce_max(iou_volume, axis=3, keepdims=True)  # B * S * S

        # Metric 계산
        # NMS (Non Maximum Suppression)
        cell_count = scale_size * scale_size
        pred_conf_class_prob = pred_class * tf.cast(pred_conf > 0.5, tf.float32)  # B * S * S * 3 * 20

        nms_object_classprob_mask = tf.zeros([tf.cast(batch_size, tf.float32), scale_size, scale_size, 3], tf.float32)
        nms_object_exist_mask = tf.zeros([tf.cast(batch_size, tf.float32), scale_size, scale_size, 3], tf.bool)

        for batch_id in range(batch_size):
            bbox_yxminmax = tf.concat([
                pred_bbox_xy_min[batch_id, ..., 1],
                pred_bbox_xy_min[batch_id, ..., 0],
                pred_bbox_xy_max[batch_id, ..., 1],
                pred_bbox_xy_max[batch_id, ..., 0]
            ], -1)  # S * S * 3 * 4
            bbox_flatten_yxminmax = tf.reshape(bbox_yxminmax, [-1, 4])  # (S * S * 3) * 4
            # 507 * 4, 2028 * 4, 8112 * 4

            class_score_per_batch = tf.zeros([scale_size, scale_size, 3, 0], tf.float32)

            for class_id in range(20):
                score_per_class = pred_conf_class_prob[batch_id, ..., class_id]  # S * S * 3
                score_flatten_indexes = tf.reshape(score_per_class, [-1])  # (S * S * 3)
                # 507, 2028, 8112

                selected_box_mask = tf.zeros([scale_size, scale_size, 3], tf.bool)
                selected_box_mask_flatten = tf.reshape(selected_box_mask, [-1])

                selected_box_indicies = tf.image.non_max_suppression(
                    boxes=bbox_flatten_yxminmax,
                    scores=score_flatten_indexes,
                    max_output_size=10,
                    # iou_threshold=.5,
                    score_threshold=.5,
                    # max_output_size=16  # tf.cast(cell_count, tf.int32)
                )

                # 선택된 Cell들을 Assign하고 다시 덮어씌운다.
                selected_box_indicies_count = tf.shape(selected_box_indicies)[0]
                selected_box_mask_flatten = tf.tensor_scatter_nd_update(
                    tensor=selected_box_mask_flatten,
                    indices=tf.expand_dims(selected_box_indicies, 1),
                    updates=tf.tile([True], [selected_box_indicies_count])
                )
                # tf.gather(selected_box_mask_flatten, selected_box_indicies).assign(True) # alternatvies upper
                selected_box_mask = tf.reshape(selected_box_mask_flatten, [scale_size, scale_size, 3])
                # selected_box_mask_expand = tf.expand_dims(selected_box_mask, 3)

                score_per_class = score_per_class * tf.cast(selected_box_mask, tf.float32)  # S * S * 3
                score_per_class_expand = tf.expand_dims(score_per_class, axis=3)  # S * S * 3 * 1

                # tf.print("class %d: selected bbox  " % class_id, selected_box_indicies)
                class_score_per_batch = tf.concat([class_score_per_batch, score_per_class_expand], -1)

            # tf.print("tf.shape(class_score_per_batch):", tf.shape(class_score_per_batch))
            # tf.print("class_score_per_batch length over 0.5's", tf.reduce_sum(tf.cast(class_score_per_batch > .5, tf.float32)))

            pred_nms_class_prob_per_batch = tf.reduce_max(class_score_per_batch, 3)  # S * S * 3, tf.bool
            pred_nms_obj_mask_per_batch = pred_nms_class_prob_per_batch > .5  # S * S * 3, tf.bool

            pred_nms_class_prob_pre_batch = tf.zeros([tf.cast(batch_id, tf.float32), scale_size, scale_size, 3], tf.float32)
            pred_nms_class_prob_expand = tf.expand_dims(pred_nms_class_prob_per_batch, 0)  # 1 * S * S * 3
            pred_nms_class_prob_post_batch = tf.zeros([tf.cast(batch_size - 1 - batch_id, tf.float32), scale_size, scale_size, 3], tf.float32)

            pred_nms_obj_mask_pre_batch = tf.zeros([tf.cast(batch_id, tf.float32), scale_size, scale_size, 3], tf.bool)
            pred_nms_obj_mask_expand = tf.expand_dims(pred_nms_obj_mask_per_batch, 0)  # 1 * S * S * 3
            pred_nms_obj_mask_post_batch = tf.zeros([tf.cast(batch_size - 1 - batch_id, tf.float32), scale_size, scale_size, 3], tf.bool)

            pred_nms_class_prob_mask = tf.concat([pred_nms_class_prob_pre_batch, pred_nms_class_prob_expand, pred_nms_class_prob_post_batch], 0)  # tf.float32, 0~1
            pred_nms_total_mask = tf.concat([pred_nms_obj_mask_pre_batch, pred_nms_obj_mask_expand, pred_nms_obj_mask_post_batch], 0)  # tf.bool

            # tf.print("pred_nms_total_mask", tf.shape(pred_nms_total_mask))
            # tf.print("pred_nms_class_id_mask", tf.shape(pred_nms_class_id_mask))

            nms_object_classprob_mask = nms_object_classprob_mask + pred_nms_class_prob_mask
            nms_object_exist_mask = nms_object_exist_mask | pred_nms_total_mask

        # IoU (BBox)
        pred_gt_iou_match_mask = max_iou_volume >= self.thresh_iou
        # pred_gt_iou_volume = iou_volume

        # Class
        pred_gt_class_match_mask = tf.argmax(pred_class, axis=4) == tf.argmax(label_class, axis=4)
        # pred_class_value = tf.argmax(pred_class, axis=4)
        # label_class_value = tf.argmax(label_class, axis=4)

        # Confidance
        label_conf_obj_mask = label_conf_nonexpand == 1.
        pred_conf_obj_mask = pred_conf_nonexpand >= self.thresh_confidance

        pred_object_exist_mask = nms_object_exist_mask

        # if pred_conf_class_prob.shape[1] == 26:
        #     tf.print("Anchor 2's class prob (left bottom):", pred_conf_class_prob[0, 21, 1, 0])
        #     tf.print("argmax:", tf.argmax(pred_conf_class_prob[0, 21, 1, 0], 0))
        #     tf.print("Anchor 2's class prob (left bottom):", tf.reduce_max(pred_conf_class_prob[0, 21, 1, 0], 0) > self.thresh_confidance)

        # if self.verbose:
        #     tf.print("\npred_conf_obj_mask:", tf.reduce_sum(tf.cast(pred_conf_obj_mask, tf.float32)))
        #     tf.print("nms_object_exist_mask:", tf.reduce_sum(tf.cast(nms_object_exist_mask, tf.float32)))

        '''
        TP (True Positive) 계산
        True Positive는, Network의 결과가 Positive(참)인데, 이는 정답(True)인 경우이다. 맞는 경우이다.
        '''
        # true_positive = pred_conf_obj_mask & label_conf_obj_mask & pred_gt_iou_match_mask & pred_gt_class_match_mask  # logical and
        true_positive = pred_object_exist_mask & label_conf_obj_mask & pred_gt_iou_match_mask & pred_gt_class_match_mask
        true_positive = tf.reduce_sum(tf.cast(true_positive, tf.float32))

        # tf.print("\nobject_exist_mask:", tf.reduce_sum(tf.cast(pred_object_exist_mask, tf.float32)))

        '''
        FP (False Positive) 계산
        False Positive는, Network의 결과가 Positive(참)인데, 이는 거짓(False)인 경우이다. 틀린 경우이다.
        '''
        # false_positive = pred_conf_obj_mask * (1. - label_conf_obj_mask * pred_gt_iou_match_mask * pred_gt_class_match_mask)
        false_positive = pred_object_exist_mask & ~(label_conf_obj_mask & pred_gt_iou_match_mask & pred_gt_class_match_mask)
        false_positive = tf.reduce_sum(tf.cast(false_positive, tf.float32))

        '''
        FN (False Negative) 계산
        False Negative는, Network의 결과가 Negative(거짓)인데, 이는 거짓(False)인 경우이다. 틀린 경우이다.
        '''
        # false_negative = ~(pred_conf_obj_mask & pred_gt_iou_match_mask & pred_gt_class_match_mask) & label_conf_obj_mask
        false_negative = ~pred_object_exist_mask & label_conf_obj_mask
        false_negative = tf.reduce_sum(tf.cast(false_negative, tf.float32))

        '''
        TN (True Negative) 계산
        True Negative는, Network의 결과가 Negative(거짓)인데, 이가 참(True)인 경우이다. 맞는 경우이다.
        '''
        # true_negative = (1. - pred_conf_obj_mask) * (1. - label_conf_obj_mask) * (1. - pred_gt_iou_match_mask * pred_gt_class_match_mask)
        true_negative = ~pred_object_exist_mask & ~label_conf_obj_mask
        true_negative = tf.reduce_sum(tf.cast(true_negative, tf.float32))

        if self.verbose:
            tf.print("\ntrue_positive:", true_positive)
            tf.print("false_positive:", false_positive)
            tf.print("false_negative:", false_negative)
            tf.print("true_negative:", true_negative)

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

        return self.recall

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)
        self.true_negatives.assign(0)

        self.precision.assign(0)
        self.recall.assign(0)
        self.accuracy.assign(0)