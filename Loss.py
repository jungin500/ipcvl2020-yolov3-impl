import tensorflow as tf

LAMBDA_COORD = 5.
LAMBDA_NOOBJ = .5
SCORE_THRESH = 0.2

CELL_SIZE = 7

def Yolov3Loss(y_true, y_pred):
    '''
    YOLOv3 Loss Function (equals YOLOv1 Loss function)

    :param y_true: 정답값 (7*7*6)
    :param y_pred: 예측값 (7*7*30)
    :return: Loss의 결과
    '''

    # y_pred: [batch, 7, 7, 30]
    batch_size, cell_shape_x, cell_shape_y, cell_shape_infos = y_true.shape
    if cell_shape_x != cell_shape_y:
        raise RuntimeError('Cell shape is not same: {} != {}'.format(cell_shape_x, cell_shape_y))

    # dtype of y_true: [batch_size, object_count, 5]
    first_iteration = 0

    for (class_id, cx, cy, w, h) in y_true:
        responsive_x =

    true_matrix[:, :, cx]

    cx = y_pred[:, :, :, 0]
    cy = y_pred[:, :, :, 1]
    w = y_pred[:, :, :, 2]
    h = y_pred[:, :, :, 3]
    c = y_pred[:, :, :, 4]

    loss_directives = [

    ]


def Yolo3Pred(y_true, y_pred):
    bbox_1 = y_pred[:, :, :, :5]
    bbox_2 = y_pred[:, :, :, 5:10]
    classes = y_pred[:, :, :, 10:]

    confidance_score_bbox_1 = tf.math.multiply(bbox_1[:, :, :, 4:5], classes)
    confidance_score_bbox_2 = tf.math.multiply(bbox_2[:, :, :, 4:5], classes)

    # SCORE_THRESH 이하는 0으로 변경한다.
    tf.cast(confidance_score_bbox_1 < SCORE_THRESH, confidance_score_bbox_1.dtype)
    tf.cast(confidance_score_bbox_2 < SCORE_THRESH, confidance_score_bbox_2.dtype)

    # return tf.reduce_sum(y_pred)
