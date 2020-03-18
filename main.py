from tensorflow.keras.preprocessing.image import ImageDataGenerator
from DataLoader import VOCDataset
from Model import YoloModel
from Loss import Yolov3Loss
from PIL import Image, ImageDraw

import numpy as np
import tensorflow as tf
import random


def test_display_dataset(train_dataset):
    item_no = random.randrange(0, train_dataset.__len__() - 1)

    print("Length of train dataset: ", train_dataset.__len__())
    print("--- Displaying random data ", item_no, " ---")
    img, target, current_shape = train_dataset.__getitem__(item_no)
    draw = ImageDraw.Draw(img)
    for item in target:
        converted_target_rect = [int(k * 448) for k in item[1:]]
        converted_target_rect[0] -= converted_target_rect[2] / 2
        converted_target_rect[1] -= converted_target_rect[3] / 2
        converted_target_rect[2] += converted_target_rect[0]
        converted_target_rect[3] += converted_target_rect[1]
        draw.rectangle(converted_target_rect, outline='red')

    print("Original image shape: ", current_shape)
    print("Image target: ", target)
    print("Converted Image target: class ", int(target[0][0]), ", rect ", converted_target_rect)
    print("Image:")

    img.show()


def detection_collate(batch):
    targets = []
    imgs = []
    sizes = []

    for sample in batch:
        imgs.append(sample[0])
        sizes.append(sample[2])

        np_label = np.zeros((7, 7, 6), dtype=np.float32)
        for item in sample[1]:
            objectness = 1
            classes = item[0]
            x_ratio = item[1]
            y_ratio = item[2]
            w_ratio = item[3]
            h_ratio = item[4]

            scale_factor = (1 / 7)
            grid_x_index = int(x_ratio // scale_factor)
            grid_y_index = int(y_ratio // scale_factor)
            x_offset = (x_ratio / scale_factor) - grid_x_index
            y_offset = (y_ratio / scale_factor) - grid_y_index

            np_label[grid_x_index][grid_y_index] = np.array([objectness, x_offset, y_offset, w_ratio, h_ratio, classes])

        label = tf.convert_to_tensor(np_label)
        targets.append(label)

    return tf.stack(label, 0), tf.stack(targets, 0), sizes


data_path = './VOCdevkit/VOC2007'
class_path = './voc.names'

train_dataset = VOCDataset(
    root=data_path,
    transform=None,
    class_path=class_path
)

test_display_dataset(train_dataset)

# model = YoloModel()
# model.compile(loss=Yolov3Loss)
#
# model.summary()
