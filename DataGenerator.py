from tensorflow.keras import utils
import math
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa

from PIL import Image
from YoloLoss import ANCHOR_BOXES, SCALES


def test_augmented_items(image_aug, bbs_aug):
    bbs_aug = bbs_aug.remove_out_of_image()
    bbs_aug = bbs_aug.clip_out_of_image()

    Image.fromarray(bbs_aug.draw_on_image(np.array(image_aug)), 'RGB').show()
    pass


class Labeler():
    def __init__(self, names_filename):
        self.names_list = {}

        with open(names_filename) as f:
            idx = 0
            for line in f:
                self.names_list[idx] = line
                idx += 1

    def get_name(self, index):
        return self.names_list[index].replace("\n", "")


# Necessary directives
ANCHOR_BOXES_NP = np.array(ANCHOR_BOXES).reshape((-1, 3, 2))


def get_best_iou_anchor_idx(width, height, scale_index):
    height_ratios = (ANCHOR_BOXES_NP[scale_index] / np.reshape(ANCHOR_BOXES_NP[scale_index, :, 0], (-1, 1)))[:, 1]
    current_heights = np.tile(height / width, np.shape(ANCHOR_BOXES_NP)[0])

    height_differences = np.abs(height_ratios - current_heights)
    return np.argmin(height_differences)


class Yolov3Dataloader(utils.Sequence):
    DEFAULT_AUGMENTER = iaa.SomeOf(2, [
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.Affine(
            translate_px={"x": 3, "y": 10},
            scale=(1.2, 1.2)
        ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
        iaa.AdditiveGaussianNoise(scale=0.1 * 255),
        iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
        # iaa.Affine(rotate=45),
        iaa.Sharpen(alpha=0.5)
    ])

    def __init__(self, file_name, dim=(416, 416, 3), batch_size=1, numClass=1, augmentation=False, shuffle=True, verbose=False):
        self.image_list, self.label_list = self.GetDataList(file_name)
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmenter = self.DEFAULT_AUGMENTER if augmentation else False
        self.outSize = 5 + numClass
        self.labeler = Labeler('voc.names')
        self.verbose = verbose
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.image_list) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = [self.image_list[k] for k in indexes]
        batch_y = [self.label_list[k] for k in indexes]

        X, Y = self.__data_generation(batch_x, batch_y)
        self.verbose = False  # one-time

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def GetDataList(self, folder_path: str):
        train_list = []
        lable_list = []

        f = open(folder_path, 'r')
        while True:
            line = f.readline()
            if not line: break
            train_list.append(line.replace("\n", ""))
            label_text = line.replace(".jpg", ".txt")
            # label_text = label_text.replace(
            #     'C:\\Users\\jungin500\\Desktop\\Study\\2020-yolov3-impl\\VOCdevkit\\VOC2007\\JPEGImages\\',
            #     "C:\\Users\\jungin500\\Desktop\\Study\\2020-yolov3-impl\\VOCyolo\\")
            label_text = label_text.replace("\n", "")
            lable_list.append(label_text)

        return train_list, lable_list

    def convert_yololabel_to_iaabbs(self, yolo_raw_label):
        # raw_label = [bboxes, 5], np.array([center_x, center_y, w, h, c])
        return ia.BoundingBoxesOnImage([
            ia.BoundingBox(
                x1=yolo_raw_bbox[0],
                y1=yolo_raw_bbox[1],
                x2=yolo_raw_bbox[2],
                y2=yolo_raw_bbox[3],
                # label=class_list[int(yolo_bbox[0])] # Label을 id로 활용하자
                label=yolo_raw_bbox[4]
            ) for yolo_raw_bbox in yolo_raw_label
        ], shape=(self.dim[0], self.dim[1]))

    def batch_convert_iaabbs_to_yololabel(self, augmented_labels, label_scale, scale_index):
        batch_size = len(augmented_labels)
        gt_label = np.empty((batch_size, label_scale, label_scale, 3, 25), dtype=np.float32)

        for i in range(batch_size):
            object_label = np.zeros((label_scale, label_scale, 3, 25), dtype=np.float32)
            for bbox in augmented_labels[i].bounding_boxes:
                center_x = bbox.center_x
                center_y = bbox.center_y
                width = bbox.width
                height = bbox.height
                class_id = int(float(bbox.label))  # Explicit

                anchor_idx = get_best_iou_anchor_idx(width, height, scale_index)

                scale_factor = (1 / label_scale)

                grid_x_index = int((center_x / 416) // scale_factor)
                grid_y_index = int((center_y / 416) // scale_factor)
                grid_x_index, grid_y_index = \
                    np.clip([grid_x_index, grid_y_index], a_min=0, a_max=label_scale - 1)  # 13이면 12까지만...

                if object_label[grid_y_index][grid_x_index][anchor_idx][class_id] == 0.:
                    object_label[grid_y_index][grid_x_index][anchor_idx][class_id] = 1.
                    object_label[grid_y_index][grid_x_index][anchor_idx][20:] = np.array(
                        [center_x, center_y, width, height, 1])
            gt_label[i] = object_label

        return gt_label

    def convert_iaabbs_to_yololabel(self, iaa_bbs_out, label_scale, scale_index):
        label = np.zeros((label_scale, label_scale, 3, 25), dtype=np.float32)
        raw_label = []

        for bbox in iaa_bbs_out.bounding_boxes:
            center_x = bbox.center_x
            center_y = bbox.center_y
            width = bbox.width
            height = bbox.height
            class_id = int(float(bbox.label))  # Explicit

            anchor_idx = get_best_iou_anchor_idx(width, height, scale_index)

            scale_factor = (1 / label_scale)

            grid_x_index = int((center_x / 416) // scale_factor)
            grid_y_index = int((center_y / 416) // scale_factor)
            grid_x_index, grid_y_index = \
                np.clip([grid_x_index, grid_y_index], a_min=0, a_max=label_scale - 1)  # 13이면 12까지만...

            if label[grid_y_index][grid_x_index][anchor_idx][class_id] == 0.:
                label[grid_y_index][grid_x_index][anchor_idx][class_id] = 1.
                label[grid_y_index][grid_x_index][anchor_idx][20:] = np.array([center_x, center_y, width, height, 1])

                raw_label.append(np.array([center_x, center_y, width, height, class_id]))

        return label, np.array(raw_label)

    def __data_generation(self, list_img_path, list_label_path):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # YOLOv3 Initialization
        # Generate data
        SCALED_X = []
        SCALED_Y = []

        # 지정된 batch size 혹은,
        # Boundary에 있다면 그보다 작은 값이 들어갈 것이다.
        batch_size = len(list_img_path)

        for scale_id in range(len(SCALES)):
            scale = SCALES[scale_id]

            X = np.empty((batch_size, *self.dim), dtype=np.float32)
            rX = [None] * len(list_img_path)
            Y = np.empty((batch_size, *(scale, scale, 3, 25)), dtype=np.float32)

            # 데이터 가져오기
            for i, path in enumerate(list_img_path):
                # raw_label은 x_1, y_1, x_2, y_2, c를 가지고 있다.
                label, raw_label = self.GetLabel(list_label_path[i], scale, scale_id)
                X[i,] = np.array(Image.open(path).resize((self.dim[0], self.dim[1])), dtype=np.float32) / 255.
                Y[i,] = label
                rX[i] = raw_label

            # 가져온 데이터 Augmentation
            if self.augmenter:
                # 바운딩박스(Y)는 형식을 바꿔준다.
                iaa_bbs_list = [self.convert_yololabel_to_iaabbs(k) for k in rX]

                augmented_images = self.augmenter.augment_images(X)
                augmented_labels = self.augmenter.augment_bounding_boxes(iaa_bbs_list)

                augmented_labels_converted = self.batch_convert_iaabbs_to_yololabel(augmented_labels, scale, scale_id)

                # 기존 값을 대치한다.
                X = augmented_images
                Y = augmented_labels_converted

            SCALED_X.append(X)
            SCALED_Y.append(Y)

        return SCALED_X, SCALED_Y

    def GetLabel(self, label_path, label_scale, scale_index):
        f = open(label_path, 'r')
        label = np.zeros((label_scale, label_scale, 3, 25), dtype=np.float32)
        raw_label = []
        while True:
            line = f.readline()
            if not line: break

            split_line = line.split()
            c, x, y, w, h = split_line

            x = float(x)  # global-relative x, y
            y = float(y)
            w = float(w)  # global-relative w, h
            h = float(h)
            c = int(c)

            anchor_idx = get_best_iou_anchor_idx(w, h, scale_index)
            scale_factor = (1 / label_scale)

            # // : 몫
            grid_x_index = int(x // scale_factor)
            grid_y_index = int(y // scale_factor)

            if self.verbose:
                print("[%d] GT Grid [y=%d, x=%d, a=%d] %s" % (label_scale, grid_y_index, grid_x_index, anchor_idx, self.labeler.get_name(c)))

            # x와 y는 해당 grid cell-relative value이다.
            cell_relative_x, cell_relative_y = (x * label_scale, y * label_scale)
            cell_relative_x, cell_relative_y = \
                (cell_relative_x - int(cell_relative_x), cell_relative_y - int(cell_relative_y))

            # 레이블은 하나만 지정한다.
            # 같은 Cell에 두 개 이상의 레이블이 들어가게 되면,
            # 하나의 객체만 사용한다.
            if label[grid_y_index][grid_x_index][anchor_idx][c] == 0.:
                label[grid_y_index][grid_x_index][anchor_idx][c] = 1.
                label[grid_y_index][grid_x_index][anchor_idx][20:] = np.array([cell_relative_x, cell_relative_y, w, h, 1])

                raw_label.append(np.array([
                    x - w / 2,
                    y - h / 2,
                    x + w / 2,
                    y + h / 2,
                    c
                ]))
            else:
                # print("Skipping labeling ... two or more bbox in same cell")
                pass

        return label, np.array(raw_label)

    def GetLabelName(self, label_id):
        return self.labeler.get_name(label_id)
