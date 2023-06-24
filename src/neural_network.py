import threading
import torch
from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import select_device
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from yolov7.utils.datasets import letterbox
import numpy as np
import cv2
import random
import multiprocessing
import time
from .shared_variables import SharedVariables

label_map = {
    0: 'Bird',
    1: 'Insect',
    2: 'Plane',
    3: 'UAP',
    # and so on for the rest of the labels
}

# def plot_one_box(x, img, color=None, label=None, line_thickness=None):
#     # same as before

class NeuralNetworkProcessing:
    def __init__(self, shared_variables: SharedVariables):
        self.shared_vars = shared_variables


        # self.processing_thread = threading.Thread(target=self._process_frames)
        # self.processing_thread.daemon = True
        # self.processing_thread.start()
        self.processing_process = multiprocessing.Process(target=self._process_frames)
        self.processing_process.start()

    def _process_frames(self):

        # Load the model once
        self.device = select_device('0')
        self.weights = 'models/best.pt'
        # self.imgsz = 640
        self.imgsz = 1600
        self.model = attempt_load(self.weights, map_location=self.device)
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())
        
        if self.device.type != 'cpu':
            self.model.half()

        while True:
            if self.shared_vars.tracking_enabled and not self.shared_vars.frame_queue.empty():
                frame = self.shared_vars.frame_queue.get()
                # cv2.imshow('image', frame)
                cv2.waitKey(1)
                # cv2.waitKey(0)  # waits until a key is pressed
                

                track_data, image = self.detect_and_draw(frame)
                self.shared_vars.tracking_queue.put(track_data)
                self.shared_vars.filtered_frame_queue.put(image)
                # self.detect_and_draw(frame)

    # 'plot_one_box' is a helper function to draw the detection bounding boxes on the image
    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
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
    
    def draw_fps(self, image, fps, location=(10, 50), color=(255, 0, 0), thickness=2, fontScale=1):
        """Draw the FPS on an image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(image, fps_text, location, font, fontScale, color, thickness, cv2.LINE_AA)
        return image


    def detect_and_draw(self, image):
        # # Convert the image array to a format expected by the model
        # image_array = cv2.imread('test.jpg')
        # img = letterbox(image_array, self.imgsz, stride=int(self.model.stride.max()))[0]
        # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # img = np.ascontiguousarray(img)

        # # Convert
        # img = torch.from_numpy(img).to(self.device)
        # img = img.half() if self.device.type != 'cpu' else img.float()  # uint8 to fp16/32
        # img /= 255.0 # Normalize 0 - 255 to 0.0 - 1.0

        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)

        # with torch.no_grad():
        #     pred = self.model(img, augment=False)[0]

        # # Apply NMS
        # pred = non_max_suppression(pred, 0.001, 0.45, classes=None, agnostic=False)
        # predn = pred[0]
        # scale_coords(img.shape[2:], predn[:, :4], image_array.shape).round()

        # # Initialize track_data
        # track_data = {'boxes': [], 'scores': [], 'labels': []}

        # # Draw the bounding boxes
        # for *xyxy, conf, cls in reversed(predn):
        #     cls = int(cls)
        #     label = '%s %.2f' % (label_map[cls], conf)
        #     # plot_one_box(xyxy, image_array, label=label, color=(255, 0, 0), line_thickness=3)

        #     track_data['boxes'].append(xyxy)
        #     track_data['scores'].append(conf.item())
        #     track_data['labels'].append(label_map[cls])

        # Record start time for FPS calculation
        start_time = time.time()

        # # Capture frame-by-frame
        # ret, image = cap.read()
        # if not ret:
        #     break

        # Same processing as before...
        # image = cv2.imread('test.jpg')
        img = letterbox(image, self.imgsz, stride=int(self.model.stride.max()))[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.device.type != 'cpu' else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img, augment=False)[0]

        pred = non_max_suppression(pred, 0.1, 0.45, classes=None, agnostic=False)
        predn = pred[0]
        scale_coords(img.shape[2:], predn[:, :4], image.shape).round()

        # Initialize track_data
        track_data = {'boxes': [], 'scores': [], 'labels': []}

        for *xyxy, conf, cls in reversed(predn):
            cls = int(cls)
            label = '%s %.2f' % (label_map[cls], conf)
            self.plot_one_box(xyxy, image, label=label, color=(255, 0, 0), line_thickness=3)
            for i in range(len(xyxy)):
                xyxy[i] = xyxy[i].cpu().numpy()

            conf = conf.cpu().numpy()

            track_data['boxes'].append(xyxy)
            track_data['scores'].append(conf)
            track_data['labels'].append(label_map[cls])

        # self.shared_vars.tracking_queue.put(track_data)



        # for *xyxy, conf, cls in reversed(predn):
        #     cls = int(cls)
        #     label = '%s %.2f' % (label_map[cls], conf)
        #     #plot_one_box(xyxy, image, label=label, color=(255, 0, 0), line_thickness=3)
        #     track_data['boxes'].append(xyxy)
        #     track_data['scores'].append(conf.item())
        #     track_data['labels'].append(label_map[cls])

        # # Initialize track_data
        # track_data = {'boxes': [], 'scores': [], 'labels': []}

        # # Draw the bounding boxes
        # for *xyxy, conf, cls in reversed(predn):
        #     cls = int(cls)
        #     label = '%s %.2f' % (label_map[cls], conf)
        #     # plot_one_box(xyxy, image_array, label=label, color=(255, 0, 0), line_thickness=3)

        #     track_data['boxes'].append(xyxy)
        #     track_data['scores'].append(conf.item())
        #     track_data['labels'].append(label_map[cls])

        # Display the resulting frame
        # cv2.imshow('image', img)
        # img_show = img.cpu().numpy()
        # img_show = np.transpose(img_show.squeeze(), (1, 2, 0))  # Move color channels to the end
        # img_show *= 255.0  # Denormalize if your image was normalized to 0-1 range
        # img_show = img_show.astype(np.uint8)  # Convert to uint8
        # cv2.imshow('image', img_show)


        # Calculate and print FPS in the console
        # fps = 1.0 / (time.time() - start_time)
        # print(f"FPS: {fps:.2f}")
        fps = 1.0 / (time.time() - start_time)
        self.draw_fps(image, fps)

        # Display the resulting frame
        # cv2.imshow('image', image)

        return track_data, image

