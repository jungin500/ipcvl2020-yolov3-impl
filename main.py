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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def parse_gtlabel(gt_label, data_generator, scale_size):
    '''
    gt_label -> [S * S * 5 * 25]
    - 13*13의 cell 중에 Confidence가 0이 아닌 값이 있는 곳(confidence mask가 1인 곳)만 보면 됨
    - 마찬가지로 5개의 Anchor 중 Confidence가 0이 아닌 값이 있는 곳만 보면 됨
    - Class ID는 25개 나열 시 앞 20개, 맨 뒤 5개(x, y, w, h, c=1)는 제외한다.
      이 값들은 x_c, y_c, w, h로 parse한다.
    '''

    confidence_mask = gt_label[:, :, :, 24]  # 13 * 13 * 5
    anchor_mask = np.argmax(confidence_mask, axis=2)  # 13 * 13. 5개의 anchor 중엔 무조건 1개만 있기 때문이다.

    parsed_objects = []

    for y in range(scale_size):
        for x in range(scale_size):

            # 값이 0이면 skip이다...
            anchor_idx = anchor_mask[y, x]

            if confidence_mask[y, x, anchor_idx] != 1.0:
                continue

            # 값이 있다면 해당 anchor는 cell의 5개 anchor중 유일하게 값을 가지며,
            # 해당 cell은 해당 object에 responsible하다.

            x_center = gt_label[y][x][anchor_idx][20]
            y_center = gt_label[y][x][anchor_idx][21]
            width = gt_label[y][x][anchor_idx][22]
            height = gt_label[y][x][anchor_idx][23]

            cell_x1 = x_center - (width / 2)
            cell_y1 = y_center - (height / 2)
            cell_x2 = x_center + (width / 2)
            cell_y2 = y_center + (height / 2)

            object_bbox = {
                'cell_x1': cell_x1,
                'cell_y1': cell_y1,
                'cell_x2': cell_x2,
                'cell_y2': cell_y2,
                'cell_id_x': x,
                'cell_id_y': y,
                'class_id': np.argmax(gt_label[y][x][anchor_idx][:20], axis=0),
                'class_name': data_generator.GetLabelName(np.argmax(gt_label[y][x][anchor_idx][:20], axis=0)),
                'anchor_idx': anchor_idx
            }

            parsed_objects.append(object_bbox)

    return parsed_objects


def check_if_responsible_cell_anchor(x, y, anchor_id, parsed_gt):
    for parsed_object in parsed_gt:
        label_anchor_id = parsed_object['anchor_idx']
        cell_x = parsed_object['cell_id_x']
        cell_y = parsed_object['cell_id_y']

        if x == cell_x and y == cell_y and anchor_id == label_anchor_id:
            return True
    return False


# result: b * S * S * 3 * 25
def display_result_image_v3(input_image, network_output, label, data_generator, scale_size, scale_index,
                            no_suppress=False, display_all=True,
                            display_by_anchors=False):
    classes = network_output[..., :20]  # ? * S * S * 3 * 20
    bbox = network_output[..., 20:24]  # ? * S * S * 3 * 4
    confidence = network_output[..., 24]  # ? * S * S * 3

    class_score_bbox = np.expand_dims(confidence, axis=4) * classes  # ? * S * S * 3 * 20

    # Set zero if core < thresh1 (0.2)
    class_score_bbox[np.where(class_score_bbox < thresh1)] = 0.

    # class_score 중에서 가장 높은 class id
    class_score_bbox_max_class = np.argmax(class_score_bbox, axis=4)
    class_score_bbox_max_score = np.amax(class_score_bbox, axis=4)

    batch_size = np.shape(input_image)[0]

    if not display_all:
        display_range = list(range(batch_size))
        random.shuffle(display_range)
        display_range = display_range[:4]
    else:
        display_range = range(batch_size)

    for batch in display_range:
        input_image_single = input_image[batch]

        if display_by_anchors:
            input_images = [
                Image.fromarray((input_image_single * 255).astype(np.uint8), 'RGB')
                for _ in range(3)
            ]
        else:
            input_image_pil = Image.fromarray((input_image_single * 255).astype(np.uint8), 'RGB')

        max_anchor_id_per_cell = np.argmax(confidence, axis=3)  # ? * S * S

        # GT를 그린다.
        # 현재는 display_by_anchors 상태에서만 가능하다.
        parsed_gt = parse_gtlabel(label[batch], data_generator, scale_size)
        for parsed_object in parsed_gt:
            anchor_id = parsed_object['anchor_idx']
            x_1 = parsed_object['cell_x1']
            y_1 = parsed_object['cell_y1']
            x_2 = parsed_object['cell_x2']
            y_2 = parsed_object['cell_y2']
            class_name = parsed_object['class_name']

            outline_mask = Image.new('RGBA', (416, 416))
            outline_mask_draw = ImageDraw.Draw(outline_mask)
            outline_mask_draw.rectangle([x_1, y_1, x_2, y_2], outline=(0, 0, 255, 255), width=3)  # Blue
            outline_mask_draw.text([x_1 + 5, y_1 + 5], text='GT-' + class_name, fill='blue')

            if display_by_anchors:
                input_images[anchor_id].paste(outline_mask, mask=outline_mask)
            else:
                input_image_pil.paste(outline_mask, mask=outline_mask)

        # 모델의 Inference 결과를 그린다.
        for y in range(scale_size):
            for x in range(scale_size):
                if no_suppress:
                    anchor_range = range(3)
                else:
                    anchor_range = [max_anchor_id_per_cell[batch][y][x]]  # 하나만 넣기...!

                for anchor_id in anchor_range:
                    class_id = class_score_bbox_max_class[batch][y][x][anchor_id]
                    class_score_bbox = class_score_bbox_max_score[batch][y][x][anchor_id]

                    if not no_suppress and class_score_bbox_max_score[batch][y][x][anchor_id] == 0:
                        continue

                    if not no_suppress and class_score_bbox < thresh2:
                        continue

                    # # Confidence를 그린다.
                    # confidence_value = int(confidence[batch][y][x][anchor_id] * 100) / 100

                    (t_x, t_y, t_w, t_h) = bbox[batch][y][x][anchor_id]
                    diff = (1 / scale_size * 416)

                    '''
                    scale_size = 13 -> 32
                    scale_size = 26 -> 16
                    scale_size = 52 -> 8
                    '''
                    x_c = (sigmoid(t_x) * (416 / scale_size)) + (x * diff)
                    y_c = (sigmoid(t_y) * (416 / scale_size)) + (y * diff)
                    w = ANCHOR_BOXES[(2 * 3) * scale_index + 2 * anchor_id] * np.exp(t_w)
                    h = ANCHOR_BOXES[(2 * 3) * scale_index + 2 * anchor_id + 1] * np.exp(t_h)

                    x_1 = (x_c - (w / 2))
                    y_1 = (y_c - (h / 2))
                    y_2 = (y_c + (h / 2))
                    x_2 = (x_c + (w / 2))

                    # class_score_bbox 값에 따라 투명도를 달리 한다.
                    outline_mask = Image.new('RGBA', (416, 416))
                    outline_mask_draw = ImageDraw.Draw(outline_mask)

                    # supress_text = '' if no_suppress else '[' + str(class_score_bbox) + ']'
                    supress_text = str(int(class_score_bbox * 100) / 100) + '\n'

                    fill_style = (255, 0, 0)
                    if class_score_bbox >= thresh2:
                        fill_style = (0, 255, 0)

                    # Red
                    if check_if_responsible_cell_anchor(x, y, anchor_id, parsed_gt):
                        outline_mask_draw.rectangle([x_1, y_1, x_2, y_2],
                                                    outline=(*fill_style, 255), width=3)
                        outline_mask_draw.text([x_1 + 5, y_1 + 5],
                                               text=supress_text + train_data.GetLabelName(class_id),
                                               fill=fill_style)
                        # print("y={}, x={}, anchor={}, confidence={}, class_score_bbox={}".format(y, x, anchor_id, confidence_value, class_score_bbox))
                    else:
                        outline_mask_draw.rectangle([x_1, y_1, x_2, y_2],
                                                    # outline=(255, 0, 0, int(class_score_bbox * 255)), width=1)
                                                    outline=(*fill_style, 155 + int(class_score_bbox * 100)), width=1)
                        outline_mask_draw.text([x_1 + 5, y_1 + 5],
                                               text=supress_text + train_data.GetLabelName(class_id),
                                               fill=(255, 255, 0, 155 + int(class_score_bbox * 100)))

                    # outline_mask_draw.text([x * 32, y * 32], text=str(confidence_value), fill='white')

                    if display_by_anchors:
                        input_images[anchor_id].paste(outline_mask, mask=outline_mask)
                    else:
                        input_image_pil.paste(outline_mask, mask=outline_mask)

        if display_by_anchors:
            for image in input_images:
                image.show()
        else:
            input_image_pil.show()


def display_dataset_image(dataset):
    item_index = random.randint(0, dataset.__len__())

    # __getitem__
    indexes = dataset.indexes[item_index * dataset.batch_size:(item_index + 1) * dataset.batch_size]

    list_img_path = [dataset.image_list[k] for k in indexes]
    list_label_path = [dataset.label_list[k] for k in indexes]

    # __data_generation
    batch_size = len(list_img_path)

    X = np.empty((batch_size, *dataset.dim), dtype=np.float32)
    rX = [None] * len(list_img_path)
    Y = np.empty((batch_size, *(26, 26, 3, 25)), dtype=np.float32)

    # 데이터 가져오기
    for i, path in enumerate(list_img_path):
        # raw_label은 x_1, y_1, x_2, y_2, c를 가지고 있다.
        label, raw_label = dataset.GetLabel(list_label_path[i], 26, 1)  # scale 26, id 1
        X[i,] = np.array(Image.open(path).resize((dataset.dim[0], dataset.dim[1])), dtype=np.float32) / 255.
        Y[i,] = label
        rX[i] = raw_label

    # 바운딩박스(Y)는 형식을 바꿔준다.
    iaa_bbs_list = [dataset.convert_yololabel_to_iaabbs(k) for k in rX]

    augmented_images = dataset.augmenter.augment_images(X)
    augmented_labels = dataset.augmenter.augment_bounding_boxes(iaa_bbs_list)

    for label_id in len(augmented_labels):
        image = Image.fromarray((augmented_images[label_id] * 255.).astype(np.int8), 'rgb')  # 416 * 416 * 3, np.array
        label = augmented_labels[label_id]

        label.draw_on_image(image)
        image.show()

MODE_TRAIN = True
INTERACTIVE_TRAIN = False
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

TARGET_TRAIN_DATA = dev_1

LOG_NAME = "1items-validset-500epochs-lr0.1-decay0.01"

CHECKPOINT_SAVE_DIR = "D:\\ModelCheckpoints\\2020-yolov3-impl\\"
LOAD_CHECKPOINT_FILENAME = CHECKPOINT_SAVE_DIR + "20200421-235535-weights.epoch300-loss42.33.hdf5"
CHECKPOINT_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-")

GLOBAL_EPOCHS = 500
SAVE_PERIOD_SAMPLES = len(TARGET_TRAIN_DATA.image_list) * 2000  # 2000 epoch

# With BN
# LEARNING_RATE = 0.1
# DECAY_RATE = 1e-2  # ref: 1e-5

# No BN
LEARNING_RATE = 1e-5
DECAY_RATE = 1e-6  # ref: 1e-5

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
            image, label = TARGET_TRAIN_DATA.__getitem__(random.randrange(0, TARGET_TRAIN_DATA.__len__()))
            net_out = model.predict(image)
            out_scale_13, out_scale_26, out_scale_52 = model.predict(image)  # B * S * S * 3 * 25
            # net_out = [out_scale_13, out_scale_26, out_scale_52]
            net_out = [out_scale_13]
            for scale_index in range(len(net_out)):
                print("Displaying scale " + str(SCALES[scale_index]))
                display_result_image_v3(image[scale_index], net_out[scale_index], label[scale_index], TARGET_TRAIN_DATA,
                                        SCALES[scale_index], scale_index, no_suppress=True,
                                        display_all=True,
                                        display_by_anchors=True)

            model.fit(
                TARGET_TRAIN_DATA,
                epochs=int(GLOBAL_EPOCHS / epoch_divide_by),
                # validation_data=valid_train_data,
                shuffle=False,
                callbacks=callback_list,
                verbose=1
            )

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
        image, label = TARGET_TRAIN_DATA.__getitem__(random.randrange(0, TARGET_TRAIN_DATA.__len__()))
        net_out = model.predict(image)
        out_scale_13, out_scale_26, out_scale_52 = model.predict(image)  # B * S * S * 3 * 25
        net_out = [out_scale_13, out_scale_26, out_scale_52]
        for scale_index in range(len(net_out)):
            display_result_image_v3(image[scale_index], net_out[scale_index], label[scale_index], TARGET_TRAIN_DATA,
                                    SCALES[scale_index], scale_index, no_suppress=False,
                                    display_all=False,
                                    display_by_anchors=True)