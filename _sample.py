import imgaug as ia
from imgaug import augmenters as iaa
from YoloLabelDecoder import YoloLabelDecoder
import numpy as np


class DataArgument():
    def __init__(self, bDataArgument, GridX, GridY, scale_factor):
        self.GridX = GridX
        self.GridY = GridY
        self.scale_factor = 1 / scale_factor

        self.YoloLabelDecoder = YoloLabelDecoder()
        ia.seed(1)
        if bDataArgument:
            self.seq = iaa.Sequential(
                [
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-10, 10)
                    ),
                    iaa.SomeOf((0, 5),
                               [
                                   iaa.OneOf([
                                       iaa.GaussianBlur((0, 3.0)),
                                       iaa.AverageBlur(k=(2, 7)),
                                       iaa.MedianBlur(k=(3, 11)),
                                   ]),
                                   iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                                   iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                                   iaa.OneOf([
                                       iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                   ]),
                                   iaa.Add((-10, 10), per_channel=0.5),
                                   iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                   iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                               ],
                               random_order=True
                               )
                ],
                random_order=True)  # apply augmenters in random order
        else:
            self.seq = iaa.Sequential([])

    def to_deterministic(self):
        self.seq_det = self.seq.to_deterministic()

    # 나중에 배치로 넣어야함
    def ImageArg(self, img, target):
        argImg = self.seq_det.augment_images([img])
        lt_rbs = self.YoloLabelDecoder.CoordToLTRB(target, argImg[0].shape, self.GridY, self.GridX)

        image = ia.quokka(size=(argImg[0].shape[0], argImg[0].shape[1]))

        # Bound Box Argument
        # Return LT, RB
        for idxY in range(int(self.GridY)):
            for idxX in range(int(self.GridX)):
                if lt_rbs[idxY, idxX, 0] == 1.:
                    lr_rb_bbox = lt_rbs[idxY, idxX, 1: 5]

                    b = ia.BoundingBox(x1=lr_rb_bbox[0], y1=lr_rb_bbox[1], x2=lr_rb_bbox[2], y2=lr_rb_bbox[3])

                    bbs = ia.BoundingBoxesOnImage([b], shape=image.shape)

                    bbs_aug = self.seq_det.augment_bounding_boxes([bbs])[0]
                    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

                    if len(bbs_aug.bounding_boxes) == 0:
                        lt_rbs[idxY, idxX, 0: 6] = [0, 0, 0, 0, 0, 0]
                        continue

                    after = bbs_aug.bounding_boxes[0]

                    if after.width < 20 or after.height < 20:
                        lt_rbs[idxY, idxX, 0: 6] = [0, 0, 0, 0, 0, 0]
                        continue

                    after.x1 = int(np.clip(after.x1, 0, argImg[0].shape[0] - 1))
                    after.x2 = int(np.clip(after.x2, 0, argImg[0].shape[0] - 1))
                    after.y1 = int(np.clip(after.y1, 0, argImg[0].shape[0] - 1))
                    after.y2 = int(np.clip(after.y2, 0, argImg[0].shape[0] - 1))

                    # print("x : {}, y : {}".format(after.x1, after.y1))
                    # print("w : {}, h : {}".format(after.x2 - after.x1, after.y2 - after.y1))
                    # print("LBx : {}, LBy : {}".format(after.x2, after.y2))

                    if after.x1 is 0 and after.x2 is 0 and after.y1 is 0 and after.y2 is 0:
                        lt_rbs[idxY, idxX, 0: 6] = [0, 0, 0, 0, 0, 0]
                    else:
                        lt_rbs[idxY, idxX, 1: 5] = [after.x1, after.y1, after.x2, after.y2]

        y = np.zeros(((int(self.GridY), int(self.GridX), 6)))

        count = 0
        for idxY in range(int(self.GridY)):
            for idxX in range(int(self.GridX)):
                if lt_rbs[idxY, idxX, 0] == 1.:
                    cls = lt_rbs[idxY, idxX, 5]
                    lr_rb_bbox = lt_rbs[idxY, idxX, 1: 5] / argImg[0].shape[0]

                    width = lr_rb_bbox[2] - lr_rb_bbox[0]
                    height = lr_rb_bbox[3] - lr_rb_bbox[1]

                    centerX = lr_rb_bbox[0] + (width / 2)
                    centerY = lr_rb_bbox[1] + (height / 2)

                    x_offset = (centerX / self.scale_factor)
                    y_offset = (centerY / self.scale_factor)

                    grid_y = int(y_offset)
                    grid_x = int(x_offset)

                    y[grid_y, grid_x, 0:6] = [1, x_offset, y_offset, width, height, cls]

        return argImg, y

