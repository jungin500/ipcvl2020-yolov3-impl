import numpy as np
from PIL import Image, ImageDraw

ANCHORS = np.array([1.19,1.99, 2.79,4.60, 4.54,8.93, 8.06,5.29, 10.33,10.65]).reshape((5, 2))
ANCHOR_X, ANCHOR_Y = ANCHORS[:, 0], ANCHORS[:, 1]
ANCHOR_RATIO = ANCHOR_Y / ANCHOR_X
AVG_IOUS = 0.615864

image = Image.new('RGB', (416, 416))
imageDraw = ImageDraw.Draw(image)

rect_beg_idx_x = 10
rect_beg_idx_y = 10
PX_SIZE = 1 / AVG_IOUS

for anchor_idx in range(np.shape(ANCHOR_RATIO)[0]):
    anchor = ANCHOR_RATIO[anchor_idx]

    target_draw_x = PX_SIZE
    target_draw_y = target_draw_x * anchor
    imageDraw.rectangle(
        xy=[rect_beg_idx_x, rect_beg_idx_y, rect_beg_idx_x + target_draw_x, rect_beg_idx_y + target_draw_y],
        outline='red'
        # fill='yellow'
    )

    rect_beg_idx_x += 3
    rect_beg_idx_y += 10

image.show()
image.save("C:/Users/jungin500/Desktop/image.png", format='png')