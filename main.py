# INFO까지의 로그 Suppress하기
import os
import os.path
import numpy as np
from PIL import Image, ImageDraw

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from DataLoader import Yolov2Dataloader
from Model import YoloModel
from Loss import Yolov3Loss

GLOBAL_EPOCHS = 5
SAVE_PERIOD_EPOCHS = 1
CHECKPOINT_FILENAME = "yolov2-training.hdf5"
MODE_TRAIN = True
LOAD_WEIGHT = True

LEARNING_RATE = 5e-6
DECAY_RATE = 5e-5
thresh1 = 0.2
thresh2 = 0.2
BATCH_SIZE = 4

# train_data = Yolov2Dataloader(file_name='manifest-train.txt', numClass=20, batch_size=BATCH_SIZE, augmentation=True)
# train_data_no_augmentation = Yolov2Dataloader(file_name='manifest-train.txt', numClass=20, batch_size=BATCH_SIZE,
#                                               augmentation=False)
# valid_train_data = Yolov2Dataloader(file_name='manifest-valid.txt', numClass=20, batch_size=2)
# test_data = Yolov2Dataloader(file_name='manifest-test.txt', numClass=20, batch_size=4)

train_data_two = Yolov2Dataloader(file_name='manifest-two.txt', numClass=20, batch_size=BATCH_SIZE)

TARGET_TRAIN_DATA = train_data_two

# 6 anchors
# [https://github.com/Jumabek/darknet_scripts/blob/master/generated_anchors/voc-anchors-reproduce/anchors3.txt]
# ANCHORS = [1.44, 2.42, 4.04, 6.30, 9.58, 9.66]

# Original anchors
# [https://github.com/experiencor/keras-yolo3/blob/master/config.json]
ANCHORS = [55, 69, 75, 234, 133, 240, 136, 129, 142, 363, 203, 290, 228, 184, 285, 359, 341, 260]

train_model, infer_model = YoloModel(
    nb_class=20,
    anchors=ANCHORS,
    max_box_per_image=5,
    max_grid=(448, 448),
    batch_size=BATCH_SIZE,
    warmup_batches=3,
    ignore_thresh=0.5,
    grid_scales=[1, 1, 1],
    obj_scale=5,
    noobj_scale=1,
    xywh_scale=1,
    class_scale=1
)
optimizer = Adam(learning_rate=LEARNING_RATE, decay=DECAY_RATE)
train_model.compile(optimizer=optimizer, loss=Yolov3Loss)
infer_model.compile(optimizer=optimizer, loss=Yolov3Loss)

train_model.summary()

save_frequency = int(
    SAVE_PERIOD_EPOCHS * TARGET_TRAIN_DATA.__len__() / TARGET_TRAIN_DATA.batch_size *
    (1 if TARGET_TRAIN_DATA.augmenter else TARGET_TRAIN_DATA.augmenter_size)
)
print("Save frequency is {} sample, batch_size={}.".format(save_frequency, TARGET_TRAIN_DATA.batch_size))

save_best_model = ModelCheckpoint(
    CHECKPOINT_FILENAME,
    save_best_only=True,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_freq=save_frequency
)

if LOAD_WEIGHT:
    if os.path.isfile(CHECKPOINT_FILENAME):
        train_model.load_weights(CHECKPOINT_FILENAME)
    else:
        print("Checkpoint file not found: ".format(CHECKPOINT_FILENAME))

if MODE_TRAIN:
    train_model.fit(
        TARGET_TRAIN_DATA,
        epochs=GLOBAL_EPOCHS,
        validation_data=TARGET_TRAIN_DATA,
        shuffle=True,
        callbacks=[save_best_model],
        verbose=1
    )
else:
    import random

    data_iterations = 1
    result_set = []
    for _ in range(data_iterations):
        image, _, _ = test_data.__getitem__(random.randrange(0, test_data.__len__()))
        result = infer_model.predict(image)
        # postprocess_non_nms_result(image, result)

    print(result_set)
