import tensorflow as tf

def Yolov3Loss(y_true, y_pred):
    '''
    YOLOv3 Loss Function
    
    :param y_true: 정답값 (7*7*6)
    :param y_pred: 예측값 (7*7*30)
    :return: Loss의 결과
    '''
    
    
    # return tf.reduce_sum(y_pred)