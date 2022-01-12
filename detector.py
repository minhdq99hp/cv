import os
from collections import deque
from datetime import datetime

import numpy as np
import torch
# import torchvision.transforms.functional as F
import torchvision.ops.boxes as bops
from PIL import Image
from torchvision.models.detection import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes

# model_path = ''
# num_classes = 4
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model = fasterrcnn_resnet50_fpn(pretrained=True)
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# model.load_state_dict(torch.load(model_path, map_location=device))

# model.eval()


# image_path = ""
# image = Image.open(image_path)

# làm nào mà nó thành ảnh tensor scale 0-1 thì làm nhé <3 


# images = [image.to(device)]
# with torch.no_grad():
#     predictions = model(images)

# predictions có dạng [{'boxes': torch.Tensor(), 'labels': torch.Tensor(), 'scores': torch.Tensor()}]

# print(predictions)


class Detector:

    def __init__(self, model_name='yolo5s', device='gpu', confidence=0.5, iou=0.75):
        self.model = None
        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou
        if device in ('gpu', 'cuda'):
            self.device = 'cuda'
            self.torch_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = 'cpu'
            self.torch_device = torch.device('cpu')

        self.colors = [(255, 0, 0), (0, 255, 0), (128, 0, 128)]

        if self.model_name == 'yolo5m':
            weight_path = os.path.join('weights', 'yolo5m_best')
            self.model = torch.hub.load('yolov5', 'custom', path=weight_path, source='local')

        elif self.model_name == 'yolo5s':
            weight_path = os.path.join('weights', 'yolo5s_best')
            self.model = torch.hub.load('yolov5', 'custom', path=weight_path, source='local')
        
        elif self.model_name == 'yolo5l':
            weight_path = os.path.join('weights', 'yolo5l_best')
            self.model = torch.hub.load('yolov5', 'custom', path=weight_path, source='local')

        elif self.model_name == 'faster_rcnn':
            weight_path = os.path.join('weights', 'faster_rcnn.pt')
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            num_classes = 4
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            self.model.load_state_dict(torch.load(weight_path, map_location=self.torch_device))
            self.model.eval()
            self.model.to(self.torch_device)


        # if self.device == 'cuda':
        #     self.model.cuda()
        # else:
        #     self.model.cpu()

        # torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.queue = deque()
        self.total = 0.0

        self.names = ['no_mask', 'mask', 'incorrect_mask']

        
    def _detect_yolov5(self, img):
        """
        
        Input: img - RGB numpy array
        """

        output = self.model(img)

        # annotate
        input_tensor = torch.from_numpy(img).permute(2, 0, 1)

        pred = output.pred[0]
        pred = pred[pred[:, -2] > self.confidence]

        if pred.size(0) >= 2:
            # filter by iou

            valid_tensor = torch.tensor([[True]] * pred.size(0), device=self.device)

            # print(valid_tensor.size())

            for i in range(pred.size(0) - 1):
                for j in range(i+1, pred.size(0)):
                    if bool(valid_tensor[i][0]) and bool(valid_tensor[j][0]):
                        # calculate IoU
                        iou = float(bops.box_iou(pred[i:i+1, :-2], pred[j:j+1, :-2])[0][0])

                        if iou > self.iou:
                            print('IOU: ', iou)

                            print(pred[i], pred[j])
                            if float(pred[i, -2]) > float(pred[j, -2]):
                                valid_tensor[i][0] = False
                            else:
                                valid_tensor[j][0] = False
            
            if not all(valid_tensor):
                pred = pred[valid_tensor.repeat(1, pred.size(1))]

                pred = torch.unsqueeze(pred, dim=0)


            pass
        
        labels = [output.names[int(x[-1])] for x in pred]
        box_colors = [self.colors[int(x[-1])] for x in pred]

        out = draw_bounding_boxes(input_tensor, boxes=pred[:, :-2], labels=labels, colors=box_colors, width=3)

        
        
        out = out.permute(1, 2, 0).numpy()
        return out


    def _detect_faster_rcnn(self, img):
        """
        
        Input: numpy array
        """

        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        input_tensor = (img_tensor / 255).to(self.torch_device)

        predictions = []

        with torch.no_grad():
            predictions = self.model([input_tensor])
        
        if not predictions:
            return None
        
        # annotate
        boxes = predictions[0]['boxes']
        scores = predictions[0]['scores']
        labels = predictions[0]['labels'] - 1

        mask = scores > self.confidence

        mask = mask[:, None].repeat(1, 4)

        masked_boxes = boxes[mask]
        masked_boxes = torch.reshape(masked_boxes, (masked_boxes.size(0) // 4, 4))

        if masked_boxes.size(0) >= 2:
            # filter by iou

            valid_tensor = torch.tensor([[True]] * masked_boxes.size(0), device=self.torch_device)

            for i in range(masked_boxes.size(0) - 1):
                for j in range(i+1, masked_boxes.size(0)):
                    if bool(valid_tensor[i][0]) and bool(valid_tensor[j][0]):
                        # calculate IoU
                        iou = float(bops.box_iou(masked_boxes[i:i+1, :], masked_boxes[j:j+1, :])[0][0])

                        if iou > self.iou:
                            print('IOU: ', iou)

                            print(masked_boxes[i], masked_boxes[j])
                            if float(scores[i]) > float(scores[j]):
                                valid_tensor[i][0] = False
                            else:
                                valid_tensor[j][0] = False
            
            if not all(valid_tensor):
                masked_boxes = masked_boxes[valid_tensor.repeat(1, 4)]
                masked_boxes = torch.reshape(masked_boxes, (masked_boxes.size(0) // 4, 4))
            

        labels = labels[:masked_boxes.size(0)]
        
        box_labels = [self.names[int(x)] for x in labels]
        box_colors = [self.colors[int(x)] for x in labels]

        out = draw_bounding_boxes(img_tensor, boxes=masked_boxes, labels=box_labels, colors=box_colors, width=3)

        out = out.permute(1, 2, 0).cpu().numpy()
        return out


    def detect(self, img):
        """
        Input: PIL image/RGB numpy array

        Return: PIL image/RGB numpy array
        """

        start = datetime.now()

        if self.model_name in ('yolo5m', 'yolo5s', 'yolo5l'):
            res = self._detect_yolov5(img)
        elif self.model_name in ('faster_rcnn'):
            res = self._detect_faster_rcnn(img)
        
        end = datetime.now()

        a = (end-start).total_seconds()

        if len(self.queue) < 10:
            
            print(f'Total duration: {a * 1000} ms')
            self.queue.append(a)
            self.total += a
        else:
            b = self.queue.popleft()
            self.queue.append(a)
            self.total = self.total - b + a

            print(f"Mean: {self.total / 10 * 1000} ms")

        return res
