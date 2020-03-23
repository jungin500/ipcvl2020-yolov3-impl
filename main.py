from tensorflow.keras.preprocessing.image import ImageDataGenerator
from DataLoader import VOCDataset
from Model import YoloModel
from Loss import Yolov3Loss
from PIL import Image, ImageDraw
from augmentation import *

import imgaug as ia
import imgaug.augmenters as iaa
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


def test_display_augment_dataset(train_dataset):
    seq = iaa.SomeOf(2, [
        # iaa.Multiply((1.2, 1.5)),
        iaa.Affine(
            translate_px={"x": 100, "y": 200},
            scale=(0.9, 0.9)
        ),
        # iaa.AdditiveGaussianNoise(scale=0.1 * 255),
        # iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
        # iaa.Affine(rotate=45)
        # iaa.Sharpen(alpha=0.5)
    ])

    print("Length of train dataset: ", train_dataset.__len__())
    items = [train_dataset.__getitem__(random.randrange(0, train_dataset.__len__())) for k in range(20)]
    # items = [ train_dataset.__getitem__(0) ]
    batch_size = 4
    aug_det = seq.to_deterministic()
    for img, target, current_shape in items:
        imgaug_target = convert_voc_bbox_to_iaa_bbox(target, train_dataset.classes, 448, 448)
        print("--- Displaying random data  ---")
        print("Original image shape: ", current_shape)
        print("Original target output (YOLOv3 output): ", target)
        print("Image target bbox(input to imgaug): ", imgaug_target)

        image_aug, bbs_aug = seq(image=np.array(img), bounding_boxes=imgaug_target)
        bbs_aug = bbs_aug.remove_out_of_image()
        bbs_aug = bbs_aug.clip_out_of_image()

        bbs_aug_yolo = [
            [
                float(bbox.label),
                bbox.center_x / 448,
                bbox.center_y / 448,
                bbox.width / 448,
                bbox.height / 448
            ] for bbox in bbs_aug.bounding_boxes
        ]

        print("Augmented Image: ", image_aug.shape)
        print("Augmented BBox: ", bbs_aug)
        print("Augmented and YOLOized BBox: ", bbs_aug_yolo)
        Image.fromarray(bbs_aug.draw_on_image(image_aug)).show()
        Image.fromarray(imgaug_target.draw_on_image(np.array(img))).show()
        input()

    return

    print("Original image shape: ", current_shape)
    print("Image target: ", target)
    print("Converted Image target: class ", int(target[0][0]), ", rect ", converted_target_rect)
    print("Image: ", img)

    # seq_det = seq.to_deterministic()
    # image_aug = seq_det.augment_image([img])

    Image.fromarray(image_aug).show()


data_path = './VOCdevkit/VOC2007'
class_path = './voc.names'

train_dataset = VOCDataset(
    root=data_path,
    # transform=augmentImage,  # Image augmenter
    transform=None,
    class_path=class_path
)

# test_display_dataset(train_dataset)
test_display_augment_dataset(train_dataset)

# model = YoloModel()
# model.compile(loss=Yolov3Loss)
#
# model.summary()
