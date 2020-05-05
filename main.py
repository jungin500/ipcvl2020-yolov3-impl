# INFO까지의 로그 Suppress하기
import datetime
import os.path
import random
import numpy as np
from PIL import Image, ImageDraw

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda
from YoloLoss import CreateYolov3Loss, ANCHOR_BOXES, SCALES
from YoloModel import Yolov3Model
from DataGenerator import Yolov3Dataloader

import tensorflow as tf

# Read class list from file
class_list = []
with open('voc.names', 'r') as f:
    while True:
        line = f.readline()
        if not line: break
        class_list.append(line)


def visualize(image, gt_label, out_label=None, batch_index=0, scale_range=range(3), anchor_range=range(3), pred_threshold=0.2, verbose=False):
    anchor_merge = False
    if anchor_range == 'merge':
        anchor_merge = True
        anchor_range = range(3)

    pred_display_only_gt = False
    if pred_threshold == 'gtonly':
        pred_display_only_gt = True
        pred_threshold = 1.0

    image_size = 416

    for scale_id in scale_range:
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
                        # GT Image BBox
                        global_relative_xy = (raw_gt_label[y, x, a, 20:22] + np.array([x, y])) / np.array([width, height])
                        gt_xmin, gt_ymin = (global_relative_xy - (raw_gt_label[y, x, a, 22:24] / 2)) * image_size
                        gt_xmax, gt_ymax = (global_relative_xy + (raw_gt_label[y, x, a, 22:24] / 2)) * image_size

                        transparant_pil = Image.new('RGBA', (416, 416))
                        image_pil_draw = ImageDraw.Draw(transparant_pil)
                        image_pil_draw.rectangle([gt_xmin, gt_ymin, gt_xmax, gt_ymax],
                                                 fill=(255, 255, 0, 100), outline=(255, 255, 0, 255))
                        image_pil_draw.text([gt_xmin + 4, gt_ymin + 2], text='O', fill='white')
                        image_pil.paste(transparant_pil, mask=transparant_pil)

                        # Responsible Cell
                        r_xmin, r_xmax = (width_block * x, width_block * (x + 1))
                        r_ymin, r_ymax = (height_block * y, height_block * (y + 1))

                        transparant_pil = Image.new('RGBA', (416, 416))
                        image_pil_draw = ImageDraw.Draw(transparant_pil)
                        image_pil_draw.rectangle([r_xmin, r_ymin, r_xmax, r_ymax], fill=(255, 0, 0, 150))
                        image_pil_draw.text([r_xmin + 2, r_ymin + 2], text='R', fill='black')
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
                            raw_modelout_label[y, x, a, 24] >= pred_threshold or
                            (pred_display_only_gt and raw_gt_label[y, x, a, 24] == 1.)
                    ):
                        # Print class prob on result view
                        if verbose:
                            class_prob_per_cell = raw_modelout_label[y, x, a, :20] * raw_modelout_label[y, x, a, 24]
                            max_class_id = np.argmax(class_prob_per_cell)
                            max_class_prob = np.max(class_prob_per_cell)
                            print("[{}, {}, anchor={}] Class {} probability {}".format(
                                y, x, a, class_list[max_class_id], max_class_prob
                            ))

                        # ModelOut Image BBox
                        global_relative_xy = (raw_modelout_label[y, x, a, 20:22] + np.array([x, y])) / np.array([width, height])
                        pred_xmin, pred_ymin = (global_relative_xy - (raw_modelout_label[y, x, a, 22:24] / 2)) * image_size
                        pred_xmax, pred_ymax = (global_relative_xy + (raw_modelout_label[y, x, a, 22:24] / 2)) * image_size
                        threshold_value_str = str(int(max_class_prob * 10) / 10)

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


MODE_TRAIN = True
INTERACTIVE_TRAIN = True
LOAD_WEIGHT = False

train_data = Yolov3Dataloader(file_name='manifest-train.txt', numClass=20, batch_size=16, augmentation=True)
train_data_no_augmentation = Yolov3Dataloader(file_name='manifest-train.txt', numClass=20, batch_size=16,
                                              augmentation=False)
valid_train_data = Yolov3Dataloader(file_name='manifest-valid.txt', numClass=20, batch_size=16)
test_data = Yolov3Dataloader(file_name='manifest-test.txt', numClass=20, batch_size=2)

dev_1 = Yolov3Dataloader(file_name='manifest-1.txt', numClass=20, batch_size=1, augmentation=False)
dev_2 = Yolov3Dataloader(file_name='manifest-2.txt', numClass=20, batch_size=2, augmentation=False)
dev_8 = Yolov3Dataloader(file_name='manifest-8.txt', numClass=20, batch_size=8, augmentation=True)
dev_16 = Yolov3Dataloader(file_name='manifest-16.txt', numClass=20, batch_size=16, shuffle=False, augmentation=False)

dev_64 = Yolov3Dataloader(file_name='manifest-64.txt', numClass=20, batch_size=16, augmentation=True)
dev_64_validset = Yolov3Dataloader(file_name='manifest-64-valid.txt', numClass=20, batch_size=16)

# image, label = dev_1.__getitem__(0)
# visualize(image, label, batch_index=0, scale_range=range(3), anchor_range='merge', pred_threshold=0.2, verbose=True)
# exit(0)

TARGET_TRAIN_DATA = dev_1

LOG_NAME = "1items-novalid-500epochs-lr0.1-decay0.01"

CHECKPOINT_SAVE_DIR = "D:\\ModelCheckpoints\\2020-yolov3-impl\\"
LOAD_CHECKPOINT_FILENAME = CHECKPOINT_SAVE_DIR + "20200421-235535-weights.epoch300-loss42.33.hdf5"
CHECKPOINT_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-")

GLOBAL_EPOCHS = 2000
SAVE_PERIOD_SAMPLES = len(TARGET_TRAIN_DATA.image_list) * 2000  # 2000 epoch

# With BN
LEARNING_RATE = 0.1
DECAY_RATE = 1e-2  # ref: 1e-5

# No BN
# LEARNING_RATE = 1e-5
# DECAY_RATE = 1e-6  # ref: 1e-5

# No BN, Interactive 1-by-1
# LEARNING_RATE = 1e-4
# DECAY_RATE = 1e-5  # ref: 1e-5

thresh1 = 0.2
thresh2 = 0.2

model = Yolov3Model()
optimizer = Adam(learning_rate=LEARNING_RATE, decay=DECAY_RATE)
model.compile(optimizer=optimizer, loss=[CreateYolov3Loss(SCALES[i], i) for i in range(len(SCALES))])

# model.summary()

save_frequency_raw = SAVE_PERIOD_SAMPLES
print("Save frequency is {} sample, batch_size={}.".format(save_frequency_raw, TARGET_TRAIN_DATA.batch_size))

if LOAD_WEIGHT and (LOAD_CHECKPOINT_FILENAME is not None):
    model.load_weights(LOAD_CHECKPOINT_FILENAME)

if LOG_NAME is not None:
    log_dir = "logs\\" + LOG_NAME + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")
else:
    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

model_checkpoint = ModelCheckpoint(
    CHECKPOINT_SAVE_DIR + CHECKPOINT_TIMESTAMP + 'weights.epoch{epoch:02d}-loss{loss:.2f}.hdf5',
    save_best_only=True,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    # save_freq=save_frequency
    save_freq=save_frequency_raw
)

tensor_board = TensorBoard(
    log_dir=log_dir,
    write_graph=True,
    update_freq=1,
    profile_batch=0
)

early_stopping = EarlyStopping(
    monitor='loss',
    patience=10,
    baseline=5e-1
)

callback_list = None
if INTERACTIVE_TRAIN:
    callback_list = [model_checkpoint]
else:
    callback_list = [model_checkpoint, tensor_board]

# force
callback_list = []

if MODE_TRAIN:
    if INTERACTIVE_TRAIN:

        epoch_divide_by = 5
        epoch_iteration = 0
        while epoch_iteration * (GLOBAL_EPOCHS / epoch_divide_by) < GLOBAL_EPOCHS:
            # Train <GLOBAL_EPOCHS / epoch_divide_by> epoches
            model.fit(
                TARGET_TRAIN_DATA,
                epochs=int(GLOBAL_EPOCHS / epoch_divide_by),
                # validation_data=valid_train_data,
                shuffle=False,
                callbacks=callback_list,
                verbose=1
            )

            image, gt_label = TARGET_TRAIN_DATA.__getitem__(random.randrange(0, TARGET_TRAIN_DATA.__len__()))
            net_out = model.predict(image)
            visualize(image, gt_label, net_out, batch_index=0, scale_range=[0], anchor_range='merge', pred_threshold='gtonly', verbose=True)

            epoch_iteration += 1
    else:
        model.fit(
            TARGET_TRAIN_DATA,
            epochs=GLOBAL_EPOCHS,
            # validation_data=valid_train_data,
            shuffle=False,
            callbacks=callback_list,
            verbose=1
        )
else:
    import random

    data_iterations = 1
    for _ in range(data_iterations):
        image, gt_label = TARGET_TRAIN_DATA.__getitem__(random.randrange(0, TARGET_TRAIN_DATA.__len__()))
        net_out = model.predict(image)
        visualize(image, gt_label, net_out, batch_index=0, scale_range=[0], anchor_range='merge', pred_threshold=0.2)
