from tensorflow.keras.initializers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def Conv2DLRU(x, filters, kernel_size, strides=(1, 1), name=None):
    x = Conv2D(filters, kernel_size, strides, use_bias=False, padding='same', name=None if name is None else 'conv_' + name)(x)
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


def YoloModel():
    input = Input(shape=(256, 256, 3))

    # L1 Outer
    x = Conv2DLRU(input, filters=32, kernel_size=3)
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

    # L9 Outer
    x = Conv2DLRU(x, filters=1024, kernel_size=3, strides=2)

    # L10 Residual (x4)
    x = ResidualIDBlock(x, filters=[512, 1024], block='a', stage=10)
    x = ResidualIDBlock(x, filters=[512, 1024], block='b', stage=10)
    x = ResidualIDBlock(x, filters=[512, 1024], block='c', stage=10)
    x = ResidualIDBlock(x, filters=[512, 1024], block='d', stage=10)

    # L11 Final
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000)(x)
    x = Softmax()(x)

    model = Model(input, x)

    return model
