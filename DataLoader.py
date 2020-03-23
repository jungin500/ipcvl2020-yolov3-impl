from tensorflow.keras.utils import Sequence

from PIL import Image
from convert2Yolo.Format import YOLO as cvtYOLO
from convert2Yolo.Format import VOC as cvtVOC

import imgaug.augmenters as iaa
import tensorflow as tf
import numpy as np
import os


class VOCDataset(Sequence):
    IMAGE_FOLDER = "JPEGImages"
    LABEL_FOLDER = "Annotations"
    IMG_EXTENSIONS = ".jpg"

    AUGMENTER = iaa.SomeOf(2, [
        iaa.Multiply((1.2, 1.5)),
        iaa.Affine(
            translate_px={"x": 3, "y": 10},
            scale=(0.9, 0.9)
        ),
        iaa.AdditiveGaussianNoise(scale=0.1 * 255),
        iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
        iaa.Affine(rotate=45),
        iaa.Sharpen(alpha=0.5)
    ])

    def __init__(self, root, batch_size=32, train=True, transform=None, target_transform=None, resize=448, shuffle=True,
                 class_path='./voc.names'):
        self.root = root
        self.batch_size = batch_size
        self.model_input_dim = (448, 448, 3)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize_factor = resize
        self.shuffle = shuffle
        self.class_path = class_path

        with open(class_path) as f:
            self.classes = f.read().splitlines()
            self.indexes = np.arange(len(self.classes))

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        self.data = self.cvtData()

    def _check_exists(self):
        print("Image folder: {}".format(os.path.join(self.root, self.IMAGE_FOLDER)))
        print("Label folder: {}".format(os.path.join(self.root, self.LABEL_FOLDER)))

        return os.path.exists(os.path.join(self.root, self.IMAGE_FOLDER)) and \
               os.path.exists(os.path.join(self.root, self.LABEL_FOLDER))

    def cvtData(self):
        result = []
        voc = cvtVOC()

        yolo = cvtYOLO(os.path.abspath(self.class_path))
        flag, self.dict_data = voc.parse(os.path.join(self.root, self.LABEL_FOLDER))

        try:
            if flag:
                flag, data = yolo.generate(self.dict_data)

                keys = list(data.keys())
                keys = sorted(keys, key=lambda key: int(key.split("_")[-1]))

                for key in keys:
                    contents = list(filter(None, data[key].split("\n")))
                    target = []
                    for i in range(len(contents)):
                        tmp = contents[i]
                        tmp = tmp.split(" ")
                        for j in range(len(tmp)):
                            tmp[j] = float(tmp[j])
                        target.append(tmp)

                    result.append(
                        {os.path.join(self.root, self.IMAGE_FOLDER, "".join([key, self.IMG_EXTENSIONS])): target})

                return result

        except Exception as e:
            raise RuntimeError("Error: {}".format(e))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key = list(self.data[index].keys())[0]
        print(key)
        img = Image.open(key).convert('RGB')
        current_shape = img.size
        img = img.resize((self.resize_factor, self.resize_factor))

        target = self.data[index][key]

        if self.transform is not None:
            # img = self.transform(img) # no augmentation
            img, aug_target = self.transform(
                augmenter=self.AUGMENTER,
                image=img,
                target=target,
                class_list=self.classes,
                image_width=self.resize_factor,
                image_height=self.resize_factor
            )

            img = tf.convert_to_tensor(img)
            return img, aug_target, current_shape
        else:
            img = tf.convert_to_tensor(np.array(img))
            return img, target, current_shape  # no augmentation

    # def on_epoch_end(self):
    #     self.indexes = np.arange(len(self.classes))
    #     if self.shuffle:
    #         np.random.shuffle(self.indexes)
