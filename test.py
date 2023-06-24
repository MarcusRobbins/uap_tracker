import torch
from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import select_device
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from yolov7.utils.datasets import letterbox
import numpy as np
import cv2
import random

label_map = {
    0: 'Bird',
    1: 'Insect',
    2: 'Plane',
    3: 'UAP',
    # and so on for the rest of the labels
}

# 'plot_one_box' is a helper function to draw the detection bounding boxes on the image
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

# Load the model
device = select_device('0')
weights = 'models/best.pt'  # replace with your weights file
imgsz = 640  # replace with the size of your images
model = attempt_load(weights, map_location=device)  
imgsz = check_img_size(imgsz, s=model.stride.max())
if device.type != 'cpu':
    model.half()

# Load image
image = cv2.imread('test.jpg') # replace with your image file
# Apply letterbox for maintaining aspect ratio
img = letterbox(image, imgsz, stride=int(model.stride.max()))[0]
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)

# Convert
img = torch.from_numpy(img).to(device)
img = img.half() if device.type != 'cpu' else img.float()  # uint8 to fp16/32
img /= 255.0 # Normalize 0 - 255 to 0.0 - 1.0

if img.ndimension() == 3:
    img = img.unsqueeze(0)

with torch.no_grad():
    pred = model(img, augment=False)[0]

# Apply NMS
pred = non_max_suppression(pred, 0.1, 0.45, classes=None, agnostic=False)
predn = pred[0]
scale_coords(img.shape[2:], predn[:, :4], image.shape).round()

# Draw the bounding boxes
for *xyxy, conf, cls in reversed(predn):
    cls = int(cls)
    label = '%s %.2f' % (label_map[cls], conf)
    plot_one_box(xyxy, image, label=label, color=(255, 0, 0), line_thickness=3)

# Use Opencv to show the image in a window
cv2.imshow('image', image)
cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image
