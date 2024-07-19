# Do the inference by OpenVINO2022.1
from pyexpat import model
import cv2
import numpy as np
import time
import yaml
from openvino.runtime import Core  # the version of openvino >= 2022.1

# 载入COCO Label
with open('/home/huang/rc/Utils/yolov5_master/data/myvoc.yaml','r', encoding='utf-8') as f:
    result = yaml.load(f.read(),Loader=yaml.FullLoader)
class_list = result['names']

# YOLOv5s输入尺寸
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# 目标检测函数，返回检测结果
def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    preds = net([blob])[next(iter(net.outputs))] # API version>=2022.1
    print(preds)
    return preds

# YOLOv5的后处理函数，解析模型的输出
def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []
    #print(output_data.shape)
    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        print(confidence)
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

# 按照YOLOv5 letterbox resize的要求，先将图像长:宽 = 1:1，多余部分填充黑边
def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

# 载入yolov5s xml or onnx模型
model_path = "/home/huang/rc/Utils/yolov5_master/best_openvino_model/best.xml"
ie = Core() #Initialize Core version>=2022.1
net = ie.compile_model(model=model_path, device_name="CPU")

# 开启Webcam，并设置为1280x720
cap = cv2.VideoCapture('/home/huang/rc/data/original/ball_video/231128/1.avi')
# 调色板
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
 
# 开启检测循环
while True:
    start = time.time()
    _, frame = cap.read()
    if frame is None:
        print("End of stream")
        break
    # 将图像按最大边1:1放缩
    # inputImage = format_yolov5(frame)
    inputImage = frame
    # 执行推理计算
    outs = detect(inputImage, net)
    # 拆解推理结果
    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

    # 显示检测框bbox
    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        color = colors[int(classid) % len(colors)]
        cv2.rectangle(frame, box, color, 2)
        cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
    
    # 显示推理速度FPS
    end = time.time()
    inf_end = end - start
    fps = 1 / inf_end
    fps_label = "FPS: %.2f" % fps
    cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(fps_label+ "; Detections: " + str(len(class_ids)))
    cv2.imshow("output", frame)

    if cv2.waitKey(1) > -1:
        print("finished by user")
        break


###  以上为onnx的openvino部署


# ###  以下为bin xml的openvino部署

# # import cv2
# # import numpy as np
# # from openvino.inference_engine import IENetwork, IECore
 
# # # 定义模型和权重路径
# # model_xml = "/home/huang/rc/sim_best.xml"
# # model_bin = "/home/huang/rc/sim_best.bin"
 
# # # 初始化IECore对象
# # ie = IECore()
 
# # # 读取IR模型
# # net = ie.read_network(model= model_xml, weights= model_bin)  
 
# # # 获取输入和输出的节点名称
# # input_blob = next(iter(net.inputs))
# # out_blob = next(iter(net.outputs))
 
# # # 加载IR模型到设备上
# # exec_net = ie.load_network(network=net, device_name="CPU")

# # try :
# #     while True:
# #         # 读取输入图像
# #         cap = cv2.VideoCapture(0)
# #         ret,image = cap.read()
        
# #         # 对输入图像进行预处理
# #         n, c, h, w = net.inputs[input_blob].shape
# #         image = cv2.resize(image, (w, h))
# #         image = image.transpose((2, 0, 1))
# #         image = image.reshape((n, c, h, w))
        
# #         # 执行推理
# #         res = exec_net.infer(inputs={input_blob: image})
        
# #         # 解析输出结果
# #         boxes = res[out_blob][0][0]
# #         for box in boxes:
# #             if box[2] > 0.5:
# #                 x_min = int(box[3] * image.shape[3])
# #                 y_min = int(box[4] * image.shape[2])
# #                 x_max = int(box[5] * image.shape[3])
# #                 y_max = int(box[6] * image.shape[2])
# #                 cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
# #         # 显示结果图像
# #         cv2.imshow("Result", image)
# #         cv2.waitKey(0)
# # finally:
# #     cv2.destroyAllWindows()

# # import cv2  
# # import numpy as np  
# # from openvino.inference_engine import IECore  
  
# # # 定义模型和权重路径  
# # model_xml = "/home/huang/rc/sim_best.xml"  
# # model_bin = "/home/huang/rc/sim_best.bin"  
  
# # # 初始化IECore对象  
# # ie = IECore()  
  
# # # 读取IR模型  
# # net = ie.read_network(model=model_xml, weights=model_bin)  
  
# # # 获取输入和输出的节点名称  
# # input_blob = next(iter(net.input_info))  
# # out_blob = next(iter(net.outputs))  
  
# # # 获取输入的形状  
# # input_shape = net.input_info[input_blob].input_data.shape  
  
# # # 加载IR模型到设备上  
# # exec_net = ie.load_network(network=net, device_name="CPU")  
  
# # try:  
# #     cap = cv2.VideoCapture(0)  
# #     while cap.isOpened():  
# #         ret, image = cap.read()  
# #         if not ret:  
# #             break  
  
# #         # 对输入图像进行预处理  
# #         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式，如果模型需要的话  
# #         image_resized = cv2.resize(image_rgb, (input_shape[3], input_shape[2]))  # 调整图像大小以匹配模型输入  
# #         image_preprocessed = np.expand_dims(image_resized, axis=0)  # 添加批量维度  
# #         image_preprocessed = image_preprocessed.transpose((0, 3, 1, 2))  # 如果模型需要特定的数据布局，则进行转置  
# #         image_preprocessed = image_preprocessed.astype(np.float32)  # 确保数据类型与模型输入一致  
  
# #         # 执行推理  
# #         res = exec_net.infer(inputs={input_blob: image_preprocessed})  
  
# #         # 解析输出结果（这里需要根据YOLOv5的实际输出格式进行调整）  
# #         # 假设输出是一个包含边界框、置信度和类别分数的数组  
# #         # 注意：这里只是一个示例，实际的解析方式可能不同  
# #         output = res[out_blob]  
# #         for obj in output[0]:  
# #             print(obj)
# #             if obj[4] > 0.5:  # 假设obj[4]是置信度  
# #                 box = obj[:4]  # 假设obj[:4]是边界框坐标  
# #                 x_min, y_min, x_max, y_max = box * [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]  
# #                 # cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)  
  
# #         # 显示结果图像  
# #         cv2.imshow("Result", image)  
# #         if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下'q'键退出  
# #             break  
# # finally:  
# #     cap.release()  # 释放摄像头资源  
# #     cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

# # import cv2
# # import time
# # import yaml
# # import torch
# # from openvino.runtime import Core
# # # https://github.com/zhiqwang/yolov5-rt-stack



# # from yolort.v5 import non_max_suppression, scale_coords
# # # Load COCO Label from yolov5/data/coco.yaml
# # with open('/home/huang/rc/Utils/yolov5_master/data/myvoc.yaml','r', encoding='utf-8') as f:
# #     result = yaml.load(f.read(),Loader=yaml.FullLoader)
# # class_list = result['names']
# # # Step1: Create OpenVINO Runtime Core
# # core = Core()
# # # Step2: Compile the Model, using dGPU
# # net = core.compile_model("/home/huang/rc/sim_best.xml")
# # output_node = net.outputs[0]
# # # color palette
# # colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
# # #import the letterbox for preprocess the frame
# # from utils.augmentations import letterbox 
# # cap = cv2.VideoCapture(0)
# # while True:
    
# #     ret,frame = cap.read()
# #     start = time.time() # total excution time =  preprocess + infer + postprocess
# #     # frame = cv2.imread("./data/images/zidane.jpg")
# #     # preprocess frame by letterbox
# #     letterbox_img, _, _= letterbox(frame, auto=False)
# #     # Normalization + Swap RB + Layout from HWC to NCHW
# #     blob = cv2.dnn.blobFromImage(letterbox_img, 1/255.0, swapRB=True)
# #     # Step 3: Do the inference
# #     outs = torch.tensor(net([blob])[output_node]) 
# #     # Postprocess of YOLOv5:NMS
# #     dets = non_max_suppression(outs)[0].numpy()
# #     bboxes, scores, class_ids= dets[:,:4], dets[:,4], dets[:,5]
# #     # rescale the coordinates
# #     bboxes = scale_coords(letterbox_img.shape[:-1], bboxes, frame.shape[:-1]).astype(int)
# #     end = time.time()
# #     #Show bbox
# #     for bbox, score, class_id in zip(bboxes, scores, class_ids):
# #         color = colors[int(class_id) % len(colors)]
# #         cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2], bbox[3]), color, 2)
# #         cv2.rectangle(frame, (bbox[0], bbox[1] - 20), (bbox[2], bbox[1]), color, -1)
# #         cv2.putText(frame, class_list[class_id], (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
# #     # show FPS
# #     fps = (1 / (end - start)) 
# #     fps_label = "FPS: %.2f" % fps
# #     cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# #     print(fps_label+ "; Detections: " + str(len(class_ids)))
# #     cv2.imshow("output", frame)
# #     cv2.waitKey(1)



# # from pathlib import Path

# # import openvino.runtime as ov
# # from openvino.preprocess import PrePostProcessor
# # from openvino.preprocess import ColorFormat
# # from openvino.runtime import Layout, Type

# # import numpy as np
# # import cv2


# # SCORE_THRESHOLD = 0.2
# # NMS_THRESHOLD = 0.4
# # CONFIDENCE_THRESHOLD = 0.4


# # def resize_and_pad(image, new_shape):
# #     old_size = image.shape[:2] 
# #     ratio = float(new_shape[-1]/max(old_size))#fix to accept also rectangular images
# #     new_size = tuple([int(x*ratio) for x in old_size])

# #     image = cv2.resize(image, (new_size[1], new_size[0]))
    
# #     delta_w = new_shape[1] - new_size[1]
# #     delta_h = new_shape[0] - new_size[0]
    
# #     color = [100, 100, 100]
# #     new_im = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=color)
    
# #     return new_im, delta_w, delta_h


# # def main( ):

# #     # Step 1. Initialize OpenVINO Runtime core
# #     core = ov.Core()
# #     # Step 2. Read a model
# #     model = core.read_model(str(Path("/home/huang/rc/sim_best.xml")))

# #     while True:
# #         cap = cv2.VideoCapture(0)
# #         ret,img = cap.read()
# #         # resize image
# #         img_resized, dw, dh = resize_and_pad(img, (640, 640))


# #         # Step 4. Inizialize Preprocessing for the model
# #         ppp = PrePostProcessor(model)
# #         # Specify input image format
# #         ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.BGR)
# #         #  Specify preprocess pipeline to input image without resizing
# #         ppp.input().preprocess().convert_element_type(Type.f32).convert_color(ColorFormat.RGB).scale([255., 255., 255.])
# #         # Specify model's input layout
# #         ppp.input().model().set_layout(Layout("NCHW"))
# #         #  Specify output results format
# #         ppp.output().tensor().set_element_type(Type.f32)
# #         # Embed above steps in the graph
# #         model = ppp.build()
# #         compiled_model = core.compile_model(model, "CPU")


# #         # Step 5. Create tensor from image
# #         input_tensor = np.expand_dims(img_resized, 0)


# #         # Step 6. Create an infer request for model inference 
# #         infer_request = compiled_model.create_infer_request()
# #         infer_request.infer({0: input_tensor})


# #         # Step 7. Retrieve inference results 
# #         output = infer_request.get_output_tensor()
# #         detections = output.data[0]


# #         # Step 8. Postprocessing including NMS  
# #         boxes = []
# #         class_ids = []
# #         confidences = []
# #         for prediction in detections:
# #             confidence = prediction[4].item()
# #             if confidence >= CONFIDENCE_THRESHOLD:
# #                 classes_scores = prediction[5:]
# #                 _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
# #                 class_id = max_indx[1]
# #                 if (classes_scores[class_id] > .25):
# #                     confidences.append(confidence)
# #                     class_ids.append(class_id)
# #                     x, y, w, h = prediction[0].item(), prediction[1].item(), prediction[2].item(), prediction[3].item()
# #                     xmin = x - (w / 2)
# #                     ymin = y - (h / 2)
# #                     box = np.array([xmin, ymin, w, h])
# #                     boxes.append(box)

# #         indexes = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)

# #         detections = []
# #         for i in indexes:
# #             j = i.item()
# #             detections.append({"class_index": class_ids[j], "confidence": confidences[j], "box": boxes[j]})


# #         # Step 9. Print results and save Figure with detections
# #         for detection in detections:
        
# #             box = detection["box"]
# #             classId = detection["class_index"]
# #             confidence = detection["confidence"]

# #             rx = img.shape[1] / (img_resized.shape[1] - dw)
# #             ry = img.shape[0] / (img_resized.shape[0] - dh)
# #             box[0] = rx * box[0]
# #             box[1] = ry * box[1]
# #             box[2] = rx * box[2]
# #             box[3] = ry * box[3]

# #             print( f"Bbox {i} Class: {classId} Confidence: {confidence} Scaled coords: [ cx: {(box[0] + (box[2] / 2)) / img.shape[1]}, cy: {(box[1] + (box[3] / 2)) / img.shape[0]}, w: {box[2]/ img.shape[1]}, h: {box[3] / img.shape[0]} ]" )
# #             xmax = box[0] + box[2]
# #             ymax = box[1] + box[3]
# #             img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(xmax), int(ymax)), (0, 255, 0), 3)
# #             img = cv2.rectangle(img, (int(box[0]), int(box[1]) - 20), (int(xmax), int(box[1])), (0, 255, 0), cv2.FILLED)
# #             img = cv2.putText(img, str(classId), (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
# #             cv2.imshow(img)
# #             cv2.waitKey(1)
    
# # if __name__ == '__main__':

# #     main( )


# #!/usr/bin/env python
# """
#  Copyright (C) 2018-2019 Intel Corporation
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# """
# from __future__ import print_function, division
 
# import logging
# import os
# import sys
# from argparse import ArgumentParser, SUPPRESS
# from math import exp as exp
# from time import time
# import numpy as np
 
# import cv2
# from openvino.inference_engine import IENetwork, IECore
 
# logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
# log = logging.getLogger()
 
 
# def build_argparser():
#     parser = ArgumentParser(add_help=False)
#     args = parser.add_argument_group('Options')
#     args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
#     args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
#                       required=True, type=str)
#     args.add_argument("-i", "--input", help="Required. Path to an image/video file. (Specify 'cam' to work with "
#                                             "camera)", required=True, type=str)
#     args.add_argument("-l", "--cpu_extension",
#                       help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
#                            "the kernels implementations.", type=str, default=None)
#     args.add_argument("-d", "--device",
#                       help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
#                            " acceptable. The sample will look for a suitable plugin for device specified. "
#                            "Default value is CPU", default="CPU", type=str)
#     args.add_argument("--labels", help="Optional. Labels mapping file", default=None, type=str)
#     args.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
#                       default=0.5, type=float)
#     args.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
#                                                        "detections filtering", default=0.4, type=float)
#     args.add_argument("-ni", "--number_iter", help="Optional. Number of inference iterations", default=1, type=int)
#     args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=False,
#                       action="store_true")
#     args.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
#                       default=False, action="store_true")
#     args.add_argument("--no_show", help="Optional. Don't show output", action='store_true')
#     return parser
 
 
# class YoloParams:
#     # ------------------------------------------- Extracting layer parameters ------------------------------------------
#     # Magic numbers are copied from yolo samples
#     def __init__(self,  side):
#         self.num = 3 #if 'num' not in param else int(param['num'])
#         self.coords = 4 #if 'coords' not in param else int(param['coords'])
#         self.classes = 80 #if 'classes' not in param else int(param['classes'])
#         self.side = side
#         self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
#                         198.0,
#                         373.0, 326.0] #if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]
 
#         #self.isYoloV3 = False
 
#         #if param.get('mask'):
#         #    mask = [int(idx) for idx in param['mask'].split(',')]
#         #    self.num = len(mask)
 
#         #    maskedAnchors = []
#         #    for idx in mask:
#         #        maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
#         #    self.anchors = maskedAnchors
 
#         #    self.isYoloV3 = True # Weak way to determine but the only one.
 
#     def log_params(self):
#         params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
#         [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]
 
 
# def letterbox(img, size=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
#     # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
#     shape = img.shape[:2]  # current shape [height, width]
#     w, h = size
 
#     # Scale ratio (new / old)
#     r = min(h / shape[0], w / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better test mAP)
#         r = min(r, 1.0)
 
#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = w - new_unpad[0], h - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (w, h)
#         ratio = w / shape[1], h / shape[0]  # width, height ratios
 
#     dw /= 2  # divide padding into 2 sides
#     dh /= 2
 
#     if shape[::-1] != new_unpad:  # resize
#         img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
 
#     top2, bottom2, left2, right2 = 0, 0, 0, 0
#     if img.shape[0] != h:
#         top2 = (h - img.shape[0])//2
#         bottom2 = top2
#         img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
#     elif img.shape[1] != w:
#         left2 = (w - img.shape[1])//2
#         right2 = left2
#         img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
#     return img
 
 
# def scale_bbox(x, y, height, width, class_id, confidence, im_h, im_w, resized_im_h=640, resized_im_w=640):
#     gain = min(resized_im_w / im_w, resized_im_h / im_h)  # gain  = old / new
#     pad = (resized_im_w - im_w * gain) / 2, (resized_im_h - im_h * gain) / 2  # wh padding
#     x = int((x - pad[0])/gain)
#     y = int((y - pad[1])/gain)
 
#     w = int(width/gain)
#     h = int(height/gain)
 
#     xmin = max(0, int(x - w / 2))
#     ymin = max(0, int(y - h / 2))
#     xmax = min(im_w, int(xmin + w))
#     ymax = min(im_h, int(ymin + h))
#     # Method item() used here to convert NumPy types to native types for compatibility with functions, which don't
#     # support Numpy types (e.g., cv2.rectangle doesn't support int64 in color parameter)
#     return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id.item(), confidence=confidence.item())
 
 
# def entry_index(side, coord, classes, location, entry):
#     side_power_2 = side ** 2
#     n = location // side_power_2
#     loc = location % side_power_2
#     return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)
 
 
# def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
#     # ------------------------------------------ Validating output parameters ------------------------------------------    
#     out_blob_n, out_blob_c, out_blob_h, out_blob_w = blob.shape
#     predictions = 1.0/(1.0+np.exp(-blob)) 
                   
#     assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
#                                      "be equal to width. Current height = {}, current width = {}" \
#                                      "".format(out_blob_h, out_blob_w)
 
#     # ------------------------------------------ Extracting layer parameters -------------------------------------------
#     orig_im_h, orig_im_w = original_im_shape
#     resized_image_h, resized_image_w = resized_image_shape
#     objects = list()
 
#     side_square = params.side * params.side
 
#     # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
#     bbox_size = int(out_blob_c/params.num) #4+1+num_classes
 
#     for row, col, n in np.ndindex(params.side, params.side, params.num):
#         bbox = predictions[0, n*bbox_size:(n+1)*bbox_size, row, col]
        
#         x, y, width, height, object_probability = bbox[:5]
#         class_probabilities = bbox[5:]
#         if object_probability < threshold:
#             continue
#         x = (2*x - 0.5 + col)*(resized_image_w/out_blob_w)
#         y = (2*y - 0.5 + row)*(resized_image_h/out_blob_h)
#         if int(resized_image_w/out_blob_w) == 8 & int(resized_image_h/out_blob_h) == 8: #80x80, 
#             idx = 0
#         elif int(resized_image_w/out_blob_w) == 16 & int(resized_image_h/out_blob_h) == 16: #40x40
#             idx = 1
#         elif int(resized_image_w/out_blob_w) == 32 & int(resized_image_h/out_blob_h) == 32: # 20x20
#             idx = 2
 
#         width = (2*width)**2* params.anchors[idx * 6 + 2 * n]
#         height = (2*height)**2 * params.anchors[idx * 6 + 2 * n + 1]
#         class_id = np.argmax(class_probabilities)
#         confidence = object_probability
#         objects.append(scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence,
#                                   im_h=orig_im_h, im_w=orig_im_w, resized_im_h=resized_image_h, resized_im_w=resized_image_w))
#     return objects
 
 
# def intersection_over_union(box_1, box_2):
#     width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
#     height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
#     if width_of_overlap_area < 0 or height_of_overlap_area < 0:
#         area_of_overlap = 0
#     else:
#         area_of_overlap = width_of_overlap_area * height_of_overlap_area
#     box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
#     box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
#     area_of_union = box_1_area + box_2_area - area_of_overlap
#     if area_of_union == 0:
#         return 0
#     return area_of_overlap / area_of_union
 
 
# def main():
#     args = build_argparser().parse_args()
 
 
#     # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
#     log.info("Creating Inference Engine...")
#     ie = IECore()
#     if args.cpu_extension and 'CPU' in args.device:
#         ie.add_extension(args.cpu_extension, "CPU")
 
#     # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
#     model = args.model
#     log.info(f"Loading network:\n\t{model}")
#     net = ie.read_network(model=model)
 
#     # ---------------------------------- 3. Load CPU extension for support specific layer ------------------------------
# #    if "CPU" in args.device:
# #        supported_layers = ie.query_network(net, "CPU")
# #        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
# #        if len(not_supported_layers) != 0:
# #            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
# #                     format(args.device, ', '.join(not_supported_layers)))
# #            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
# #                      "or --cpu_extension command line argument")
# #            sys.exit(1)
 
#     assert len(net.input_info.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"
 
#     # ---------------------------------------------- 4. Preparing inputs -----------------------------------------------
#     log.info("Preparing inputs")
#     input_blob = next(iter(net.input_info))
 
#     #  Defaulf batch_size is 1
#     net.batch_size = 1
 
#     # Read and pre-process input images
#     n, c, h, w = net.input_info[input_blob].input_data.shape
 
#     if args.labels:
#         with open(args.labels, 'r') as f:
#             labels_map = [x.strip() for x in f]
#     else:
#         labels_map = None
 
#     input_stream = 0 if args.input == "cam" else args.input
 
#     is_async_mode = True
#     cap = cv2.VideoCapture(input_stream)
#     number_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames
 
#     wait_key_code = 1
 
#     # Number of frames in picture is 1 and this will be read in cycle. Sync mode is default value for this case
#     if number_input_frames != 1:
#         ret, frame = cap.read()
#     else:
#         is_async_mode = False
#         wait_key_code = 0
 
#     # ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
#     log.info("Loading model to the plugin")
#     exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)
 
#     cur_request_id = 0
#     next_request_id = 1
#     render_time = 0
#     parsing_time = 0
 
#     # ----------------------------------------------- 6. Doing inference -----------------------------------------------
#     log.info("Starting inference...")
#     print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
#     print("To switch between sync/async modes, press TAB key in the output window")
#     while cap.isOpened():
#         # Here is the first asynchronous point: in the Async mode, we capture frame to populate the NEXT infer request
#         # in the regular mode, we capture frame to the CURRENT infer request
#         if is_async_mode:
#             ret, next_frame = cap.read()
#         else:
#             ret, frame = cap.read()
 
#         if not ret:
#             break
 
#         if is_async_mode:
#             request_id = next_request_id
#             in_frame = letterbox(frame, (w, h))
#         else:
#             request_id = cur_request_id
#             in_frame = letterbox(frame, (w, h))
 
#         in_frame0 = in_frame
#         # resize input_frame to network size
#         in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
#         in_frame = in_frame.reshape((n, c, h, w))
 
#         # Start inference
#         start_time = time()
#         exec_net.start_async(request_id=request_id, inputs={input_blob: in_frame})
#         det_time = time() - start_time
 
#         # Collecting object detection results
#         objects = list()
#         if exec_net.requests[cur_request_id].wait(-1) == 0:
#             output = exec_net.requests[cur_request_id].output_blobs
#             start_time = time()
#             for layer_name, out_blob in output.items():
#                 layer_params = YoloParams(side=out_blob.buffer.shape[2])
#                 log.info("Layer {} parameters: ".format(layer_name))
#                 layer_params.log_params()
#                 objects += parse_yolo_region(out_blob.buffer, in_frame.shape[2:],
#                                              #in_frame.shape[2:], layer_params,
#                                              frame.shape[:-1], layer_params,
#                                              args.prob_threshold)
#             parsing_time = time() - start_time
 
#         # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
#         objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
#         for i in range(len(objects)):
#             if objects[i]['confidence'] == 0:
#                 continue
#             for j in range(i + 1, len(objects)):
#                 if intersection_over_union(objects[i], objects[j]) > args.iou_threshold:
#                     objects[j]['confidence'] = 0
 
#         # Drawing objects with respect to the --prob_threshold CLI parameter
#         objects = [obj for obj in objects if obj['confidence'] >= args.prob_threshold]
 
#         if len(objects) and args.raw_output_message:
#             log.info("\nDetected boxes for batch {}:".format(1))
#             log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")
 
#         origin_im_size = frame.shape[:-1]
#         print(origin_im_size)
#         for obj in objects:
#             # Validation bbox of detected object
#             if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
#                 continue
#             color = (int(min(obj['class_id'] * 12.5, 255)),
#                      min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
#             det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
#                 str(obj['class_id'])
 
#             if args.raw_output_message:
#                 log.info(
#                     "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],
#                                                                               obj['ymin'], obj['xmax'], obj['ymax'],
#                                                                               color))
 
#             cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
#             cv2.putText(frame,
#                         "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
#                         (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
 
#         # Draw performance stats over frame
#         inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
#             "Inference time: {:.3f} ms".format(det_time * 1e3)
#         render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1e3)
#         async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
#             "Async mode is off. Processing request {}".format(cur_request_id)
#         parsing_message = "YOLO parsing time is {:.3f} ms".format(parsing_time * 1e3)
 
#         cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
#         cv2.putText(frame, render_time_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
#         cv2.putText(frame, async_mode_message, (10, int(origin_im_size[0] - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
#                     (10, 10, 200), 1)
#         cv2.putText(frame, parsing_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
 
#         start_time = time()
#         if not args.no_show:
#             cv2.imshow("DetectionResults", frame)
#         render_time = time() - start_time
 
#         if is_async_mode:
#             cur_request_id, next_request_id = next_request_id, cur_request_id
#             frame = next_frame
 
#         if not args.no_show:
#             key = cv2.waitKey(wait_key_code)
    
#             # ESC key
#             if key == 27:
#                 break
#             # Tab key
#             if key == 9:
#                 exec_net.requests[cur_request_id].wait()
#                 is_async_mode = not is_async_mode
#                 log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))
 
#     cv2.destroyAllWindows()
 
 
# if __name__ == '__main__':
#     sys.exit(main() or 0)
