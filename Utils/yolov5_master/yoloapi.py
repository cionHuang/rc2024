import os
import platform
import sys
from pathlib import Path
import time
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (Profile, check_img_size, cv2
                           , non_max_suppression,scale_boxes,xyxy2xywh)
from utils.torch_utils import select_device

class yolo:

    def __init__(self,weights=ROOT / 'best.pt', data=ROOT / 'data/coco128.yaml', source = 0, conf_thres=0.25,
                 iou_thres=0.45, max_det=1000, device='', view_img=True, save_txt=False,
                 save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False,
                 visualize=False, update=False, project=ROOT / 'runs/detect', name='exp', exist_ok=False, line_thickness=3,
                 hide_labels=False, hide_conf=False, half=False, dnn=False):

        self.weights = weights
        self.data = data
        self.source = source
        self.imgsz = [640, 640]
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.update = update
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.dnn = dnn
        self.vid_stride = 1
        self.result = []

        self.source = str(self.source)
        # Load model
        device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=device,dnn=self.dnn,data=self.data, fp16=self.half)
        stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        # Dataloader
        bs = 1  # batch_size
        self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=stride, auto=pt, vid_stride=self.vid_stride)
          
        # Run inference
        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *self.imgsz))  # warmup
        self.seen, self.windows, self.dt = 0, [], (Profile(device=self.device), Profile(device=self.device), Profile(device=self.device))


    def run(self):
        start = time.time()
        self.result = []

        for path, im, im0s, vid_cap, s in self.dataset:

            with self.dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with self.dt[1]:
               
                pred = self.model(im, augment=self.augment, visualize=self.visualize)
      
            with self.dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            # print(pred)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                self.seen += 1

                p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                s += f'{i}: '
               
                
                # 原图
                defaultPic = im0.copy()

                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    
                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class

                        if int(cls) == 0:
                            # print(list(map(int,((xyxy2xywh(torch.tensor(xyxy))).view(-1).tolist()))))
                            xPositionT12 = list(
                                map(int, ((xyxy2xywh(torch.tensor(xyxy))
                                        ).view(-1).tolist())))[0]  # 识别框的中心x坐标
                            yPositionT12 = list(
                                map(int, ((xyxy2xywh(torch.tensor(xyxy))
                                        ).view(-1).tolist())))[1]  # 识别框的中心y坐标
                            widthT12 = list(
                                map(int, ((xyxy2xywh(torch.tensor(xyxy))
                                        ).view(-1).tolist())))[2]  # 识别框的宽
                            heightT12 = list(
                                map(int, ((xyxy2xywh(torch.tensor(xyxy))
                                        ).view(-1).tolist())))[3]  # 识别框的高
                            LUT12_2 = [
                                list(
                                    map(int, (
                                        torch.tensor(xyxy).view(-1).tolist())))[0],
                                list(
                                    map(int,
                                        (torch.tensor(xyxy).view(-1).tolist())))[1]
                            ]  # 识别框的左上角坐标
                            RDT12_2 = [
                                list(
                                    map(int, (
                                        torch.tensor(xyxy).view(-1).tolist())))[2],
                                list(
                                    map(int,
                                        (torch.tensor(xyxy).view(-1).tolist())))[3]
                            ]  # 识别框的右下角坐标
                            a = [
                                xPositionT12, yPositionT12, widthT12, heightT12,
                                LUT12_2, RDT12_2]  # 将以上数据打包并存储在arraryT12数组内z
                            self.result.append(a)




                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        confidence = float(conf)
                        confidence_str = f'{confidence:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))


                # 显示推理速度FPS
                end = time.time()
                inf_end = end - start
                fps = 1 / inf_end
                fps_label = "FPS: %.2f" % fps
                

                # Stream results
                im0 = annotator.result()
                cv2.putText(im0, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if self.view_img:
                    if platform.system() == 'Linux' and p not in self.windows:
                        self.windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)

                # return self.result
               








