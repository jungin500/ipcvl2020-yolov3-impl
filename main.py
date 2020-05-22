# INFO까지의 로그 Suppress하기
import datetime
import os.path
import numpy as np
from PIL import Image, ImageDraw

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda
from YoloLoss import CreateYolov3Loss, SCALES, generate_index_matrix, get_anchor_box
from YoloModel import Yolov3Model
from YoloMetrics import *
from DataGenerator import Yolov3Dataloader

import tensorflow as tf
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
    # test_data = Yolov3Dataloader(file_name='manifest-test.txt', numClass=20, batch_size=2)

    LOG_NAME = datetime.datetime.now().strftime("%Y%m%d\\%H%M%S-") + "%s-%s-%depochs-lr%f-%s" % (
        str(get_list_length(args.manifest_train)) + "item",
        "novalid" if args.manifest_valid is None else "valid",
        args.epoches,
        args.learning_rate,
        "augm" if args.augmentation else "noaugm"
    )

    CHECKPOINT_SAVE_DIR = "D:\\ModelCheckpoints\\2020-yolov3-impl\\"
    LOAD_CHECKPOINT_FILENAME = CHECKPOINT_SAVE_DIR + "20200518-150343-weights.epoch400-loss20.38.hdf5"
    CHECKPOINT_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-")

    GLOBAL_EPOCHS = args.epoches
    SAVE_PERIOD_SAMPLES = len(train_data.image_list) * 10  # 10 epoches
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
        # metrics=[YoloMetric()]
    )

    # model.summary()

    if args.save_weight:
        print("Save frequency is {} sample, batch_size={}.".format(SAVE_PERIOD_SAMPLES, train_data.batch_size))

    if LOAD_WEIGHT and (LOAD_CHECKPOINT_FILENAME is not None):
        model.load_weights(LOAD_CHECKPOINT_FILENAME)

    model_checkpoint = ModelCheckpoint(
        CHECKPOINT_SAVE_DIR + CHECKPOINT_TIMESTAMP + 'weights.epoch{epoch:02d}-loss{loss:.2f}.hdf5',  # 'weights.epoch{epoch:02d}-loss{loss:.2f}-validloss{val_loss:.2f}.hdf5'
        save_best_only=False,
        save_weights_only=True,
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

    callback_list = []

    if not INTERACTIVE_TRAIN:
        callback_list += [tensor_board]

    if args.save_weight:
        callback_list += [model_checkpoint]

    if MODE_TRAIN:
        if INTERACTIVE_TRAIN:

            epoch_divide_by = 5
            epoch_iteration = 0
            while epoch_iteration * (GLOBAL_EPOCHS / epoch_divide_by) < GLOBAL_EPOCHS:

                # Train <GLOBAL_EPOCHS / epoch_divide_by> epoches
                model.fit(
                    train_data,
                    epochs=int(GLOBAL_EPOCHS / epoch_divide_by),
                    validation_data=valid_train_data,
                    shuffle=True,
                    callbacks=callback_list,
                    verbose=1
                )

                image, gt_label = train_data.__getitem__(random.randrange(0, train_data.__len__()))
                net_out = model.predict(image)
                visualize(image, gt_label, net_out, batch_range=[0], scale_range=[0, 1, 2],
                          anchor_range='merge', pred_threshold=0.5, verbose=True)

                epoch_iteration += 1
        else:
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
    parser.add_argument('-t', dest='train', default=True, action='store_true', help='Train Mode (Default: True)')
    parser.add_argument('-b', dest='batch_size', type=int, default=16, help='Batch size (Default: 16)')
    parser.add_argument('-e', dest='epoches', type=int, default=200, help='Epoches ( /5 if interactive ) (Default: 200)')
    parser.add_argument('-l', dest='learning_rate', type=float, default=1e-5, help='Learnig rate (Default: 1e-5)')
    parser.add_argument('-i', dest='interactive', default=False, action='store_true', help='Interactive Mode - display results every /5 epoches (Default: False)')
    parser.add_argument('-w', dest='weight_file', type=str, help='(Optional) Weight file to load (Default: None)')
    parser.add_argument('-v', dest='verbosity', type=int, default=0, help='Verbosity (0: None, 1: Loss verb, 2: Loss+DataLoader verb)')
    parser.add_argument('-s', dest='save_weight', default=False, action='store_true',
                        help='Save weight to file every 10 epoches')
    __main__(parser.parse_args())