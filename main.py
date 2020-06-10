# INFO까지의 로그 Suppress하기
import datetime
import os.path
import numpy as np
from PIL import Image, ImageDraw

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
# cumulative GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda
from YoloLoss import CreateYolov3Loss, SCALES, generate_index_matrix, get_anchor_box
from YoloModel import Yolov3Model
from YoloMetrics import *
from YoloCallbacks import *
from DataGenerator import Yolov3Dataloader

import argparse


# Read class list from file
class_list = []
with open('voc.names', 'r') as f:
    while True:
        line = f.readline()
        if not line: break
        class_list.append(line)


def get_list_length(list_file_name):
    list_length = 0
    with open(list_file_name, "r") as f:
        while True:
            line = f.readline   ()
            if not line: break
            list_length += 1

    return list_length


def visualize_v2_nms(image, gt_label, out_label=None, batch_range=[0], scale_range=range(3), anchor_range=range(3), pred_threshold=0.2, verbose=False):
    anchor_merge = False
    if anchor_range == 'merge':
        anchor_merge = True
        anchor_range = range(3)

    pred_display_only_gt = False
    if pred_threshold == 'gtonly':
        pred_display_only_gt = True
        pred_threshold = 1.0

    image_size = 416

    # [y_true_13, y_true_26, y_true_52] = gt_label
    # if out_label is not None:
    #     [y_pred_13, y_pred_26, y_pred_52] = out_label

    if out_label is not None:
        pred_gt_pair = [(gt_label[i], out_label[i]) for i in range(3)]
    else:
        pred_gt_pair = [(gt_label[i], None) for i in range(3)]

    for scale_id in scale_range:
        y_true, y_pred = pred_gt_pair[scale_id]

        pred_class = y_pred[..., :20]
        pred_bbox = y_pred[..., 20:24]
        pred_conf = y_pred[..., 24:25]
        pred_conf_nonexpand = y_pred[..., 24]

        label_class = y_true[..., :20]
        label_bbox = y_true[..., 20:24]
        label_conf = y_true[..., 24:25]
        label_conf_nonexpand = y_true[..., 24]

        # 인덱스 매트릭스를 만든다
        # (cell-based가 아닌 global-based bbox coordinate를 구하기 위해)
        batch_size = tf.shape(y_pred)[0]
        scale_size = tf.cast(tf.shape(y_true)[1], tf.float32)  # B * S * S * 3 * 25에서 S!
        scale_size_np = int(scale_size.numpy())
        scale_index = tf.cast(scale_size / 13 / 2, tf.int32) # 13, 26, 52 -> 0, 1, 2
        anchor_box = get_anchor_box(scale_index)

        image_size = 416
        width_block = 416 / scale_size
        height_block = 416 / scale_size

        index_matrix = generate_index_matrix(scale_size)

        pred_bbox_xy_glr = (tf.sigmoid(pred_bbox[..., :2]) + index_matrix) / scale_size  # normalized from 0.0 to 1.0
        pred_bbox_wh_glr = (anchor_box * tf.exp(pred_bbox[..., 2:])) / image_size

        label_bbox_xy_glr = (label_bbox[..., :2] + index_matrix) / scale_size
        label_bbox_wh = label_bbox[..., 2:]

        pred_bbox_xy_min = pred_bbox_xy_glr - (pred_bbox_wh_glr / 2)
        pred_bbox_xy_max = pred_bbox_xy_glr + (pred_bbox_wh_glr / 2)
        pred_volume = pred_bbox_wh_glr[..., 0] * pred_bbox_wh_glr[..., 1]

        label_bbox_xy_min = label_bbox_xy_glr - (label_bbox_wh / 2)
        label_bbox_xy_max = label_bbox_xy_glr + (label_bbox_wh / 2)
        label_volume = label_bbox_wh[..., 0] * label_bbox_wh[..., 1]

        intersact_min = tf.maximum(pred_bbox_xy_min, label_bbox_xy_min)
        intersact_max = tf.minimum(pred_bbox_xy_max, label_bbox_xy_max)
        intersact_wh = tf.maximum(intersact_max - intersact_min, 0)  # clip to zero
        intersact_volume = intersact_wh[..., 0] * intersact_wh[..., 1]

        union_volume = pred_volume + label_volume - intersact_volume

        # 검출된 모든 셀의 모든 IoU를 계산한 값이다.
        # 이 값들 중, TP, FP를 구해서 "Precision Metric"을 구한다.
        iou_volume = intersact_volume / union_volume  # B * S * S * 3
        max_iou_volume = tf.reduce_max(iou_volume, axis=3, keepdims=True)  # B * S * S

        # Metric 계산
        # NMS (Non Maximum Suppression)
        cell_count = scale_size * scale_size
        pred_conf_class_prob = pred_class * tf.cast(pred_conf > 0.5, tf.float32)  # B * S * S * 3 * 20

        for batch_index in batch_range:
            if image[0].shape[0] <= batch_index:
                print("!! Batch index too large: ", str(batch_index))
                continue

            bbox_yxminmax = tf.concat([
                pred_bbox_xy_min[batch_index, ..., 1],
                pred_bbox_xy_min[batch_index, ..., 0],
                pred_bbox_xy_max[batch_index, ..., 1],
                pred_bbox_xy_max[batch_index, ..., 0]
            ], -1)  # S * S * 3 * 4
            bbox_flatten_yxminmax = tf.reshape(bbox_yxminmax, [-1, 4])  # (S * S * 3) * 4
            # 507 * 4, 2028 * 4, 8112 * 4

            class_score_per_batch = tf.zeros([scale_size, scale_size, 3, 0], tf.float32)

            for class_id in range(20):
                score_per_class = pred_conf_class_prob[batch_index, ..., class_id]  # S * S * 3
                score_flatten_indexes = tf.reshape(score_per_class, [-1])  # (S * S * 3)
                # 507, 2028, 8112

                selected_box_mask = tf.zeros([scale_size, scale_size, 3], tf.bool)
                selected_box_mask_flatten = tf.reshape(selected_box_mask, [-1])

                selected_box_indicies = tf.image.non_max_suppression(
                    boxes=bbox_flatten_yxminmax,
                    scores=score_flatten_indexes,
                    iou_threshold=.5,
                    score_threshold=.5,
                    max_output_size=16  # tf.cast(cell_count, tf.int32)
                )

                # 선택된 Cell들을 Assign하고 다시 덮어씌운다.
                selected_box_indicies_count = tf.shape(selected_box_indicies)[0]
                selected_box_mask_flatten = tf.tensor_scatter_nd_update(
                    tensor=selected_box_mask_flatten,
                    indices=tf.expand_dims(selected_box_indicies, 1),
                    updates=tf.tile([True], [selected_box_indicies_count])
                )
                # tf.gather(selected_box_mask_flatten, selected_box_indicies).assign(True) # alternatvies upper
                selected_box_mask = tf.reshape(selected_box_mask_flatten, [scale_size, scale_size, 3])
                # selected_box_mask_expand = tf.expand_dims(selected_box_mask, 3)

                score_per_class = score_per_class * tf.cast(selected_box_mask, tf.float32)  # S * S * 3
                score_per_class_expand = tf.expand_dims(score_per_class, axis=3)  # S * S * 3 * 1

                class_score_per_batch = tf.concat([class_score_per_batch, score_per_class_expand], -1)

            # tf.print("tf.shape(class_score_per_batch):", tf.shape(class_score_per_batch))

            pred_nms_class_id_per_batch = tf.cast(tf.argmax(class_score_per_batch, -1), tf.float32)  # S * S * 3, tf.float32
            pred_nms_class_prob_per_batch = tf.reduce_max(class_score_per_batch, 3)  # S * S * 3, tf.float32, 0~1
            pred_nms_obj_mask_per_batch = pred_nms_class_prob_per_batch > .5  # S * S * 3, tf.bool


            # pred_nms_total_mask: NMS 결과로 Box가 존재하는 Mask를 의미함
            # (S * S * 3)
            if anchor_merge:
                image_pil = Image.fromarray((image[scale_id][batch_index] * 255).astype(np.int8), 'RGB')

            for a in range(3):
                if not anchor_merge:
                    image_pil = Image.fromarray((image[scale_id][batch_index] * 255).astype(np.int8), 'RGB')

                for y in range(scale_size_np):
                    for x in range(scale_size_np):
                        # for GT values
                        if y_true[batch_index, y, x, a, 24] != 0.:
                            # Responsible Cell
                            r_xmin, r_xmax = (width_block * x, width_block * (x + 1))
                            r_ymin, r_ymax = (height_block * y, height_block * (y + 1))

                            transparant_pil = Image.new('RGBA', (416, 416))
                            image_pil_draw = ImageDraw.Draw(transparant_pil)
                            image_pil_draw.rectangle([r_xmin, r_ymin, r_xmax, r_ymax], fill=(255, 0, 0, 150))
                            image_pil_draw.text([r_xmin + 2, r_ymin + 2], text='R', fill='black')
                            image_pil.paste(transparant_pil, mask=transparant_pil)

                            # GT Image BBox
                            global_relative_xy = (y_true[batch_index, y, x, a, 20:22] + np.array([x, y])) / np.array(
                                [scale_size_np, scale_size_np])
                            gt_xmin, gt_ymin = (global_relative_xy - (y_true[batch_index, y, x, a, 22:24] / 2)) * image_size
                            gt_xmax, gt_ymax = (global_relative_xy + (y_true[batch_index, y, x, a, 22:24] / 2)) * image_size

                            transparant_pil = Image.new('RGBA', (416, 416))
                            image_pil_draw = ImageDraw.Draw(transparant_pil)
                            image_pil_draw.rectangle([gt_xmin, gt_ymin, gt_xmax, gt_ymax],
                                                     fill=(0, 255, 255, 100), outline=(0, 255, 255, 255))
                            image_pil_draw.text([gt_xmin + 4, gt_ymin + 2], text='O', fill='white')
                            image_pil.paste(transparant_pil, mask=transparant_pil)

                        elif not anchor_merge or (anchor_merge and a == 0):
                            # Just draw grid box
                            r_xmin, r_xmax = (width_block * x, width_block * (x + 1))
                            r_ymin, r_ymax = (height_block * y, height_block * (y + 1))

                            transparant_pil = Image.new('RGBA', (416, 416))
                            image_pil_draw = ImageDraw.Draw(transparant_pil)
                            image_pil_draw.rectangle([r_xmin, r_ymin, r_xmax, r_ymax], outline=(255, 255, 0, 50))
                            image_pil.paste(transparant_pil, mask=transparant_pil)

                        # for ModelOut values
                        if ((not pred_display_only_gt and pred_nms_obj_mask_per_batch[y, x, a]) or
                                (pred_display_only_gt and y_true[batch_index, y, x, a, 24] == 1.)):  # NMS의 결과로 해당 Cell에 Object가 있는 경우!

                            object_class_id = int(pred_nms_class_id_per_batch[y, x, a].numpy())

                            # Print class prob on result view
                            class_prob_per_cell = y_pred[batch_index, y, x, a, :20] * y_pred[batch_index, y, x, a, 24]
                            max_class_prob = float(pred_nms_class_prob_per_batch[y, x, a].numpy())
                            if verbose:
                                print("[{}, {}, anchor={}] Class {} probability {}".format(
                                    y, x, a, class_list[object_class_id], max_class_prob
                                ))

                            # Constants (due to anchor boxes)
                            pred_xy_glr = (tf.sigmoid(y_pred[batch_index, y, x, a, 20:22]) + tf.constant([x, y],
                                                                                                        dtype=tf.float32)) / scale_size_np * image_size
                            pred_wh_glr = (tf.exp(y_pred[batch_index, y, x, a, 22:24]) * get_anchor_box(scale_id)[
                                0, 0, 0, a])

                            # ModelOut Image BBox

                            pred_xy_min = (pred_xy_glr - (pred_wh_glr / 2))
                            pred_xy_max = (pred_xy_glr + (pred_wh_glr / 2))
                            pred_xmin, pred_ymin = pred_xy_min[..., 0], pred_xy_min[..., 1]
                            pred_xmax, pred_ymax = pred_xy_max[..., 0], pred_xy_max[..., 1]
                            threshold_value_str = str(round(max_class_prob, 2))

                            transparant_pil = Image.new('RGBA', (416, 416))
                            image_pil_draw = ImageDraw.Draw(transparant_pil)
                            image_pil_draw.rectangle([pred_xmin, pred_ymin, pred_xmax, pred_ymax],
                                                     fill=(255, 255, 0, 100), outline=(255, 255, 0, 255))
                            image_pil_draw.text([pred_xmin + 4, pred_ymin + 2], text=threshold_value_str, fill='white')
                            image_pil.paste(transparant_pil, mask=transparant_pil)

                            # Responsible Cell
                            r_xmin, r_xmax = (width_block * x, width_block * (x + 1))
                            r_ymin, r_ymax = (height_block * y, height_block * (y + 1))

                            transparant_pil = Image.new('RGBA', (416, 416))
                            image_pil_draw = ImageDraw.Draw(transparant_pil)
                            image_pil_draw.rectangle([r_xmin, r_ymin, r_xmax, r_ymax], fill=(255, 0, 0, 150))
                            # image_pil_draw.text([r_xmin + 2, r_ymin + 2], text='P_R', fill='black')
                            image_pil.paste(transparant_pil, mask=transparant_pil)

                if not anchor_merge:
                    image_pil.show()

            if anchor_merge:
                image_pil.show()



def visualize(image, gt_label, out_label=None, batch_range=[0], scale_range=range(3), anchor_range=range(3), pred_threshold=0.2, verbose=False):
    anchor_merge = False
    if anchor_range == 'merge':
        anchor_merge = True
        anchor_range = range(3)

    pred_display_only_gt = False
    if pred_threshold == 'gtonly':
        pred_display_only_gt = True
        pred_threshold = 1.0

    image_size = 416

    for batch_index in batch_range:
        if image[0].shape[0] <= batch_index:
            print("!! Batch index too large: ", str(batch_index))
            continue

        for scale_id in scale_range:
            scale_size = SCALES[scale_id]

            raw_gt_image = image[scale_id][batch_index]
            raw_gt_label = gt_label[scale_id][batch_index]
            if out_label is not None:
                raw_modelout_label = out_label[scale_id][batch_index]

            (width, height, anchors, _) = np.shape(raw_gt_label)
            width_block = 416 / width
            height_block = 416 / height

            if anchor_merge:
                image_pil = Image.fromarray((raw_gt_image * 255).astype(np.int8), 'RGB')

            for a in anchor_range:
                if not anchor_merge:
                    image_pil = Image.fromarray((raw_gt_image * 255).astype(np.int8), 'RGB')

                # Fill in x-y ranges
                for y in range(height):
                    for x in range(width):

                        # for GT values
                        if raw_gt_label[y, x, a, 24] == 1.:
                            # Responsible Cell
                            r_xmin, r_xmax = (width_block * x, width_block * (x + 1))
                            r_ymin, r_ymax = (height_block * y, height_block * (y + 1))

                            transparant_pil = Image.new('RGBA', (416, 416))
                            image_pil_draw = ImageDraw.Draw(transparant_pil)
                            image_pil_draw.rectangle([r_xmin, r_ymin, r_xmax, r_ymax], fill=(255, 0, 0, 150))
                            image_pil_draw.text([r_xmin + 2, r_ymin + 2], text='R', fill='black')
                            image_pil.paste(transparant_pil, mask=transparant_pil)

                            # GT Image BBox
                            global_relative_xy = (raw_gt_label[y, x, a, 20:22] + np.array([x, y])) / np.array([width, height])
                            gt_xmin, gt_ymin = (global_relative_xy - (raw_gt_label[y, x, a, 22:24] / 2)) * image_size
                            gt_xmax, gt_ymax = (global_relative_xy + (raw_gt_label[y, x, a, 22:24] / 2)) * image_size

                            transparant_pil = Image.new('RGBA', (416, 416))
                            image_pil_draw = ImageDraw.Draw(transparant_pil)
                            image_pil_draw.rectangle([gt_xmin, gt_ymin, gt_xmax, gt_ymax],
                                                     fill=(0, 255, 255, 100), outline=(0, 255, 255, 255))
                            image_pil_draw.text([gt_xmin + 4, gt_ymin + 2], text='O', fill='white')
                            image_pil.paste(transparant_pil, mask=transparant_pil)

                        elif not anchor_merge or (anchor_merge and a == 0):
                            # Just draw grid box
                            r_xmin, r_xmax = (width_block * x, width_block * (x + 1))
                            r_ymin, r_ymax = (height_block * y, height_block * (y + 1))

                            transparant_pil = Image.new('RGBA', (416, 416))
                            image_pil_draw = ImageDraw.Draw(transparant_pil)
                            image_pil_draw.rectangle([r_xmin, r_ymin, r_xmax, r_ymax], outline=(255, 255, 0, 50))
                            image_pil.paste(transparant_pil, mask=transparant_pil)

                        # for ModelOut values
                        if out_label is not None and (
                                (
                                    raw_modelout_label[y, x, a, 24] >= pred_threshold
                                    and np.max(raw_modelout_label[y, x, a, 24] * raw_modelout_label[y, x, a, :20])
                                    >= pred_threshold
                                ) or
                                (pred_display_only_gt and raw_gt_label[y, x, a, 24] == 1.)
                        ):
                            # Print class prob on result view
                            class_prob_per_cell = raw_modelout_label[y, x, a, :20] * raw_modelout_label[y, x, a, 24]
                            max_class_id = np.argmax(class_prob_per_cell)
                            max_class_prob = np.max(class_prob_per_cell)
                            if verbose:
                                print("[{}, {}, anchor={}] Class {} probability {}".format(
                                    y, x, a, class_list[max_class_id], max_class_prob
                                ))

                            # Constants (due to anchor boxes)
                            pred_xy_glr = (tf.sigmoid(raw_modelout_label[y, x, a, 20:22]) + tf.constant([x, y], dtype=tf.float32)) / scale_size * image_size
                            pred_wh_glr = (tf.exp(raw_modelout_label[y, x, a, 22:24]) * get_anchor_box(scale_id)[0, 0, 0, a])

                            # ModelOut Image BBox

                            pred_xy_min = (pred_xy_glr - (pred_wh_glr / 2))
                            pred_xy_max = (pred_xy_glr + (pred_wh_glr / 2))
                            pred_xmin, pred_ymin = pred_xy_min[..., 0], pred_xy_min[..., 1]
                            pred_xmax, pred_ymax = pred_xy_max[..., 0], pred_xy_max[..., 1]
                            threshold_value_str = str(round(max_class_prob, 2))

                            transparant_pil = Image.new('RGBA', (416, 416))
                            image_pil_draw = ImageDraw.Draw(transparant_pil)
                            image_pil_draw.rectangle([pred_xmin, pred_ymin, pred_xmax, pred_ymax],
                                                     fill=(255, 255, 0, 100), outline=(255, 255, 0, 255))
                            image_pil_draw.text([pred_xmin + 4, pred_ymin + 2], text=threshold_value_str, fill='white')
                            image_pil.paste(transparant_pil, mask=transparant_pil)

                            # Responsible Cell
                            r_xmin, r_xmax = (width_block * x, width_block * (x + 1))
                            r_ymin, r_ymax = (height_block * y, height_block * (y + 1))

                            transparant_pil = Image.new('RGBA', (416, 416))
                            image_pil_draw = ImageDraw.Draw(transparant_pil)
                            image_pil_draw.rectangle([r_xmin, r_ymin, r_xmax, r_ymax], fill=(255, 0, 0, 150))
                            # image_pil_draw.text([r_xmin + 2, r_ymin + 2], text='P_R', fill='black')
                            image_pil.paste(transparant_pil, mask=transparant_pil)

                if not anchor_merge:
                    image_pil.show()

            if anchor_merge:
                image_pil.show()

def __main__(args):
    import random

    MODE_TRAIN = args.train
    INTERACTIVE_TRAIN = args.interactive
    LOAD_WEIGHT = True if args.weight_file is not None else False

    train_data = Yolov3Dataloader(file_name=args.manifest_train, numClass=20, batch_size=args.batch_size,
                                  augmentation=args.augmentation, verbose=True if args.verbosity > 1 else False)
    valid_train_data = None
    if args.manifest_valid is not None:
        valid_train_data = Yolov3Dataloader(file_name=args.manifest_valid, numClass=20, batch_size=args.batch_size)
    test_data = Yolov3Dataloader(file_name='manifest-test.txt', numClass=20, batch_size=2)

    LOG_NAME = datetime.datetime.now().strftime("%Y%m%d\\%H%M%S-") + "%s-%s-%depochs-lr%f-%s-loss-v200605" % (
        str(get_list_length(args.manifest_train)) + "item",
        "novalid" if args.manifest_valid is None else "valid",
        args.epoches,
        args.learning_rate,
        "augm" if args.augmentation else "noaugm"
    )

    CHECKPOINT_SAVE_DIR = "D:\\ModelCheckpoints\\2020-yolov3-impl\\"
    CHECKPOINT_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-")

    GLOBAL_EPOCHS = args.epoches
    SAVE_PERIOD_SAMPLES = len(train_data.image_list) * args.save_epoches  # n epoches
    DISPLAY_FREQ_EPOCHES = args.display_freq
    VERBOSE_LOSS = True if args.verbosity > 0 else False

    # References
    LEARNING_RATE = args.learning_rate
    DECAY_RATE = LEARNING_RATE * 0.01
    
    model = Yolov3Model()
    optimizer = Adam(learning_rate=LEARNING_RATE, decay=DECAY_RATE)
    # optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=[CreateYolov3Loss(SCALES[i], i, verbose=VERBOSE_LOSS) for i in range(len(SCALES))],
        metrics=[YoloMetric()] if args.enable_metric is True else []
    )

    # model.summary()

    if LOAD_WEIGHT and (args.weight_file is not None):
        print("[Train] Loading weight filr \"%s\" ..." % args.weight_file)
        model.load_weights(CHECKPOINT_SAVE_DIR + args.weight_file)
        print("[Train] Weight file loaded")

    model_checkpoint = ModelCheckpoint(
        CHECKPOINT_SAVE_DIR + CHECKPOINT_TIMESTAMP + 'weights.epoch{epoch:02d}-loss{loss:.2f}.hdf5',  # 'weights.epoch{epoch:02d}-loss{loss:.2f}-validloss{val_loss:.2f}.hdf5'
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
        # monitor='loss',
        # mode='min',
        # save_freq=save_frequency
        save_freq=SAVE_PERIOD_SAMPLES
    )

    tensor_board = TensorBoard(
        log_dir="logs\\" + LOG_NAME,
        write_graph=True,
        update_freq=1,
        profile_batch=0
    )

    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10,
        baseline=5e-1
    )

    visualizer = VisualizeYolo(
        test_dataset=valid_train_data,
        visualizer_fn=visualize_v2_nms,
        model=model,
        display_count=args.display_count,
        display_on_begin=args.display_on_begin,
        display_freq_epoches=DISPLAY_FREQ_EPOCHES
    )

    callback_list = [tensor_board]

    if INTERACTIVE_TRAIN:
        print("[Train] Interactive train display frequency is {} epoches, batch_size={}.".format(DISPLAY_FREQ_EPOCHES, train_data.batch_size))
        callback_list += [visualizer]

    if args.save_weight:
        print("[Train] Model save frequency is {} sample, batch_size={}.".format(SAVE_PERIOD_SAMPLES, train_data.batch_size))
        callback_list += [model_checkpoint]
    else:
        print("[Train] WARNING: Model not saved from here! try \"-s\" flag!")

    if MODE_TRAIN:
        model.fit(
            train_data,
            epochs=GLOBAL_EPOCHS,
            validation_data=valid_train_data,
            shuffle=True,
            callbacks=callback_list,
            verbose=1
        )
    else:
        import random

        data_iterations = 1
        for _ in range(data_iterations):
            image, gt_label = train_data.__getitem__(random.randrange(0, train_data.__len__()))
            net_out = model.predict(image)
            visualize(image, gt_label, net_out, batch_range=[0, 1], scale_range=[0], anchor_range='merge', pred_threshold=0.2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv3 Trainer")
    parser.add_argument('--manifest-train', dest='manifest_train', default='manifest-train.txt', type=str, help='Trainset manifests (Default: manifest-train.txt)')
    parser.add_argument('--manifest-valid', dest='manifest_valid', default=None, type=str, help='Validset manifests (Default: None)')
    parser.add_argument('--augmentation', dest='augmentation', default=False, action='store_true', help='Enable augmentation')
    parser.add_argument('-t', dest='train', default=True, type=bool, help='Train Mode (Default: True)')
    parser.add_argument('-b', dest='batch_size', type=int, default=16, help='Batch size (Default: 16)')
    parser.add_argument('-e', dest='epoches', type=int, default=200, help='Epoches ( /5 if interactive ) (Default: 200)')
    parser.add_argument('-l', dest='learning_rate', type=float, default=1e-5, help='Learnig rate (Default: 5e-3)')
    parser.add_argument('-m', dest='enable_metric', default=False, action='store_true', help='Enable YOLO metric (anyway slows down training speed)')
    parser.add_argument('-i', dest='interactive', default=False, action='store_true', help='Interactive Mode - display results every N epoches (Default: False)')
    parser.add_argument('--display-freq', dest='display_freq', type=int, default=10,
                        help='Interactive Mode - display batch frequency (Default: 10)')
    parser.add_argument('--display-count', dest='display_count', type=int, default=1,
                        help='Interactive Mode - how much to display on interactive display time (Default: 1)')
    parser.add_argument('--display-on-begin', dest='display_on_begin', default=False, action='store_true',
                        help='Interactive mode - display first output on begin (Default: False)')
    parser.add_argument('-w', dest='weight_file', type=str, help='(Optional) Weight file to load (Default: None)')
    parser.add_argument('-v', dest='verbosity', type=int, default=0, help='Verbosity (0: None, 1: Loss verb, 2: Loss+DataLoader verb)')
    parser.add_argument('-s', dest='save_weight', default=False, action='store_true',
                        help='Save weight to file every n epoches')
    parser.add_argument('--save-epoches', dest='save_epoches', type=int, default=400, help='Every epoches after saving weights (Default: 400)')
    __main__(parser.parse_args())