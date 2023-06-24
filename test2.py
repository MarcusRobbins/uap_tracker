import torch
from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import select_device
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from yolov7.utils.datasets import letterbox
import numpy as np
import cv2
import random
import time

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

# Create a VideoCapture object
cap = cv2.VideoCapture(0)
# print(cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840))
# print(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160))

while True:
    # Record start time for FPS calculation
    start_time = time.time()

    # Capture frame-by-frame
    ret, image = cap.read()
    if not ret:
        break

    # Same processing as before...
    image = cv2.imread('test.jpg')
    img = letterbox(image, imgsz, stride=int(model.stride.max()))[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if device.type != 'cpu' else img.float()
    img /= 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False)[0]

    pred = non_max_suppression(pred, 0.001, 0.45, classes=None, agnostic=False)
    predn = pred[0]
    scale_coords(img.shape[2:], predn[:, :4], image.shape).round()

    for *xyxy, conf, cls in reversed(predn):
        cls = int(cls)
        label = '%s %.2f' % (label_map[cls], conf)
        plot_one_box(xyxy, image, label=label, color=(255, 0, 0), line_thickness=3)

    # Display the resulting frame
    cv2.imshow('image', image)

    # Calculate and print FPS in the console
    fps = 1.0 / (time.time() - start_time)
    print(f"FPS: {fps:.2f}")

    # Press 'q' on keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
