import imgaug as ia
import imgaug.augmenters as iaa

import numpy as np


def convert_voc_bbox_to_iaa_bbox(yolo_target, class_list, image_width, image_height):
    return ia.BoundingBoxesOnImage([
        ia.BoundingBox(
            x1=(yolo_bbox[1] - (yolo_bbox[3] / 2)) * image_width,
            y1=(yolo_bbox[2] - (yolo_bbox[4] / 2)) * image_height,
            x2=(yolo_bbox[1] + (yolo_bbox[3] / 2)) * image_width,
            y2=(yolo_bbox[2] + (yolo_bbox[4] / 2)) * image_height,
            # label=class_list[int(yolo_bbox[0])] # Label을 id로 활용하자
            label=yolo_bbox[0]
        ) for yolo_bbox in yolo_target
    ], shape=(image_width, image_height))


def augmentImage(augmenter, image, target, class_list, image_width, image_height):
    imgaug_target = convert_voc_bbox_to_iaa_bbox(target, class_list, image_width, image_height)
    image_aug, bbs_aug = augmenter(image=np.array(image), bounding_boxes=imgaug_target)
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
    bbs_aug_yolo = [
        [
            float(bbox.label),
            bbox.center_x / image_width,
            bbox.center_y / image_height,
            bbox.width / image_width,
            bbox.height / image_height
        ] for bbox in bbs_aug.bounding_boxes
    ]

    return image_aug, bbs_aug_yolo
