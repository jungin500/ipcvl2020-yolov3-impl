from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, BatchNormalization, Input, \
    LeakyReLU, Reshape, Flatten, Softmax, Lambda, Concatenate, UpSampling2D, Add, Layer

import tensorflow as tf


class YoloReshape(Layer):
    def __init__(self, target_shape, **kwargs):
        super(YoloReshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.target_shape)

    # inputs = [B * S * S * 75]
    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]
        S = [self.target_shape[0], self.target_shape[1]]
        C = 20
        A = 3

        # Class probability -> Softmax!
        class_prob = tf.reshape(inputs[..., :C * A], shape=(batch_size, S[0], S[1], A, C))

        # Confidence -> Sigmoid!
        confidence = tf.reshape(inputs[..., C * A:(C + 1) * A], shape=(batch_size, S[0], S[1], A, 1))

        # Boxes -> Sigmoid!
        box = tf.reshape(inputs[..., (C + 1) * A:], shape=(batch_size, S[0], S[1], A, 4))

        return tf.concat([
            tf.nn.softmax(class_prob),  # Still class probability needs softmax activation
            tf.sigmoid(box),  # YOLOv2 introduces sigmoid to box predicator
            tf.sigmoid(confidence)   # YOLOv2 introduces sigmoid to confidence predicator. not in loss fn.
        ], axis=4)


def ConvBlock(x, filters, kernel_size, name, strides=(1, 1)):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation=None,
               name=name + '_conv')(x)
    x = BatchNormalization(name=name + '_bn')(x)
    x = LeakyReLU(alpha=.1, name=name + '_lru')(x)

    return x


def ResBlock(x, filters, kernel_size, name):
    x_branch = ConvBlock(x, filters=filters[0], kernel_size=kernel_size[0], name=name + '_0cb')
    x_branch = ConvBlock(x_branch, filters=filters[1], kernel_size=kernel_size[1], name=name + '_1cb')
    x_branch = Add(name=name + '_2add')([x, x_branch])

    return x_branch


def DualConvBlock(x, filters, kernel_size, name):
    x = ConvBlock(x, filters=filters[0], kernel_size=kernel_size[0], name=name + '_0cb')
    x = ConvBlock(x, filters=filters[1], kernel_size=kernel_size[1], name=name + '_1cb')

    return x


def Yolov3Model():
    input = Input(shape=(416, 416, 3), name='input')

    '''
    Feature Extraction network (L1~L7)
    '''
    # L1
    x = ConvBlock(input, filters=32, kernel_size=3, name='l1_0cb')
    # (None, 416, 416, 32)

    # L2
    x = ConvBlock(x, filters=64, kernel_size=3, strides=2, name='l2_0cb')
    x = ResBlock(x, filters=[32, 64], kernel_size=[1, 3], name='l2_1rb')
    # (None, 208, 208, 64)

    # L3
    x = ConvBlock(x, filters=128, kernel_size=3, strides=2, name='l3_0cb')
    x = ResBlock(x, filters=[64, 128], kernel_size=[1, 3], name='l3_1rb')
    x = ResBlock(x, filters=[64, 128], kernel_size=[1, 3], name='l3_2rb')
    # (None, 104, 104, 128)

    # L4
    x = ConvBlock(x, filters=256, kernel_size=3, strides=2, name='l4_0cb')
    x = ResBlock(x, filters=[128, 256], kernel_size=[1, 3], name='l4_1rb')
    x = ResBlock(x, filters=[128, 256], kernel_size=[1, 3], name='l4_2rb')
    x = ResBlock(x, filters=[128, 256], kernel_size=[1, 3], name='l4_3rb')
    x = ResBlock(x, filters=[128, 256], kernel_size=[1, 3], name='l4_4rb')
    x = ResBlock(x, filters=[128, 256], kernel_size=[1, 3], name='l4_5rb')
    x = ResBlock(x, filters=[128, 256], kernel_size=[1, 3], name='l4_6rb')
    x = ResBlock(x, filters=[128, 256], kernel_size=[1, 3], name='l4_7rb')
    l4_skip = x = ResBlock(x, filters=[128, 256], kernel_size=[1, 3], name='l4_8rb')
    # (None, 52, 52, 256)

    # L5
    x = ConvBlock(x, filters=512, kernel_size=3, strides=2, name='l5_0cb')
    x = ResBlock(x, filters=[256, 512], kernel_size=[1, 3], name='l5_1rb')
    x = ResBlock(x, filters=[256, 512], kernel_size=[1, 3], name='l5_2rb')
    x = ResBlock(x, filters=[256, 512], kernel_size=[1, 3], name='l5_3rb')
    x = ResBlock(x, filters=[256, 512], kernel_size=[1, 3], name='l5_4rb')
    x = ResBlock(x, filters=[256, 512], kernel_size=[1, 3], name='l5_5rb')
    x = ResBlock(x, filters=[256, 512], kernel_size=[1, 3], name='l5_6rb')
    x = ResBlock(x, filters=[256, 512], kernel_size=[1, 3], name='l5_7rb')
    l5_skip = x = ResBlock(x, filters=[256, 512], kernel_size=[1, 3], name='l5_8rb')
    # (None, 26, 26, 512)

    # L6
    x = ConvBlock(x, filters=1024, kernel_size=3, strides=2, name='l6_0cb')
    x = ResBlock(x, filters=[512, 1024], kernel_size=[1, 3], name='l6_1rb')
    x = ResBlock(x, filters=[512, 1024], kernel_size=[1, 3], name='l6_2rb')
    x = ResBlock(x, filters=[512, 1024], kernel_size=[1, 3], name='l6_3rb')
    x = ResBlock(x, filters=[512, 1024], kernel_size=[1, 3], name='l6_4rb')
    # (None, 13, 13, 1024)

    # 아래 부분은 Feature Map만 학습시키키 위한 것
    # # L7
    # x = GlobalAveragePooling2D(name='l7_0gap')(x)  # (None, 1024)
    # x = Dense(units=1000, name='l7_1fc')(x)        # (None, 1000)
    # x = Softmax(name='l7_2softmax')(x)             # (None, 1000)

    # L7 - Branch 1 - plagiarized from Darknet
    x = DualConvBlock(x, filters=[512, 1024], kernel_size=[1, 3], name='l7_b01_0dcb')
    l7_br1_skip = x = DualConvBlock(x, filters=[512, 1024], kernel_size=[1, 3], name='l7_b01_1dcb')
    x = DualConvBlock(x, filters=[512, 1024], kernel_size=[1, 3], name='l7_b01_2dcb')
    x = ConvBlock(x, filters=75, kernel_size=1, name='l7_b01_3cb_out0')
    output_scale_0 = YoloReshape(name='netout_scale13', target_shape=(13, 13, 3, 25))(x)

    # L7 - Branch 2 - same, as above.
    x = ConvBlock(l7_br1_skip, filters=256, kernel_size=1, name='l7_b02_0cb')
    x = UpSampling2D(data_format='channels_last', name='l7_b02_1us2x')(x)
    x = Concatenate(axis=3, name='l7_b02_2concat')([x, l5_skip])  # route  85 61, B * 26 * 26 * 768
    x = DualConvBlock(x, filters=[256, 512], kernel_size=[1, 3], name='l7_b02_3dcb')
    l7_br2_skip = x = DualConvBlock(x, filters=[256, 512], kernel_size=[1, 3], name='l7_b02_4dcb')
    x = DualConvBlock(x, filters=[256, 512], kernel_size=[1, 3], name='l7_b02_5dcb')
    x = ConvBlock(x, filters=75, kernel_size=1, name='l7_b02_6cb_out1')
    output_scale_1 = YoloReshape(name='netout_scale26', target_shape=(26, 26, 3, 25))(x)

    # L7 - Branch 3 - same.
    x = ConvBlock(l7_br2_skip, filters=128, kernel_size=1, name='l7_b03_0cb')
    x = UpSampling2D(data_format='channels_last', name='l7_b03_1us2x')(x)
    x = Concatenate(axis=3, name='l7_b03_2concat')([x, l4_skip])  # route  97 36, B * 26 * 26 * 768
    x = DualConvBlock(x, filters=[128, 256], kernel_size=[1, 3], name='l07_b03_3dcb')
    x = DualConvBlock(x, filters=[128, 256], kernel_size=[1, 3], name='l07_b03_4dcb')
    x = DualConvBlock(x, filters=[128, 256], kernel_size=[1, 3], name='l07_b03_5dcb')
    x = ConvBlock(x, filters=75, kernel_size=1, name='l07_b03_6cb_out2')
    output_scale_2 = YoloReshape(name='netout_scale52', target_shape=(52, 52, 3, 25))(x)

    model = Model(inputs=[input], outputs=[output_scale_0, output_scale_1, output_scale_2])
    return model
