import os
import torch
import numpy as np
from PIL import Image
from torchvision.utils import draw_bounding_boxes
# import torchvision.transforms.functional as F
import torchvision.ops.boxes as bops

# box1 = torch.tensor([[511, 41, 577, 76]], dtype=torch.float)
# box2 = torch.tensor([[544, 59, 610, 94]], dtype=torch.float)
# iou = bops.box_iou(box1, box2)


class Detector:

    def __init__(self, model_name='yolo5s', device='gpu', confidence=0.5, iou=0.75):
        self.model = None
        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou

        self.colors = [(255, 0, 0), (0, 255, 0), (128, 0, 128)]

        if self.model_name == 'yolo5m':
            weight_path = os.path.join('weights', 'yolo5m_best')
            self.model = torch.hub.load('yolov5', 'custom', path=weight_path, source='local')
        elif self.model_name == 'yolo5s':
            weight_path = os.path.join('weights', 'yolo5s_best')
            self.model = torch.hub.load('yolov5', 'custom', path=weight_path, source='local')

        if device == 'gpu':
            self.model.cuda()
        else:
            self.model.cpu()
        
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

            valid_tensor = torch.tensor([[True]] * pred.size(0), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

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


    def detect(self, img):
        """
        Input: PIL image/RGB numpy array

        Return: PIL image/RGB numpy array
        """

        if self.model_name in ('yolo5m', 'yolo5s'):
            res = self._detect_yolov5(img)
        
        return res
