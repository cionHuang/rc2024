import os
import time
from functools import partial

import numpy as np
import cv2
import torch
from openvino.inference_engine import IECore

from Utils.new_vino.utils.detector_utils import scale_coords, non_max_suppression, preprocess_image

# 目标标签
CLASS_LABELS = ['blue','red','purple','basket']

COLOR = [
    [255, 0, 0],
    [0, 0, 255],
    [128, 0, 128],
    [0, 0, 0]]


# 类定义
class vinodetect:
    """
    定义新的目标检测功能类
    """
    def __init__(self, model, device,threshold,showinfo=True):
        self.model_xml = os.path.join(model, 'best.xml') # xml文件地址
        self.model_bin = os.path.join(model, 'best.bin') # bin文件地址
        self.target_device = device # 使用设备
        self.threshold = threshold # 识别置信度
        self.showInfo = showinfo # 是否显示识别框

        # Initialize OpenVINO  读取模型相关信息
        self.OVIE, self.OVNet, self.OVExec = self.get_openvino_core_net_exec()
        self.InputLayer = next(iter(self.OVNet.input_info)) # 输入层信息
        self.OutputLayer = list(self.OVNet.outputs)[-1] # 输出层信息
        _, self.C, self.H, self.W = self.OVNet.input_info[self.InputLayer].input_data.shape # 获取输入层的通道数、宽、高


    def get_openvino_core_net_exec(self):
        # Load IECore object
        OVIE = IECore() # 实例化IECore

        # Load OpenVINO network  加载模型
        OVNet = OVIE.read_network(
            model=self.model_xml, weights=self.model_bin)

        # Create executable network 将模型部署到硬件上
        OVExec = OVIE.load_network(
            network=OVNet, device_name=self.target_device)

        return OVIE, OVNet, OVExec

    def output_process(self,detections,image_src,threshold,model_in_HW,container):

        labels = detections[..., -1].numpy() # 读取标签
        boxs = detections[..., :4].numpy() # 读取识别框坐标
        confs = detections[..., 4].numpy() # 读取置信度

        mh, mw = model_in_HW
        h, w = image_src.shape[:2]
        boxs[:, :] = scale_coords((mh, mw), boxs[:, :], (h, w)).round() # 将模型中的坐标转化到原图中

        # 遍历所有识别框
        for i, box in enumerate(boxs):
            # 凡是置信度大于预设值的均显示
            if confs[i] >= threshold:
                
                x1, y1, x2, y2 = map(int, box) # 提取识别框坐标信息
                class_info = int(labels[i]) # 提取识别框标签信息

                data_array = [x1, y1, x2, y2, class_info] # 将识别框信息整合为数组
                container.append(data_array)
                
                # 是否绘制识别框
                if self.showInfo == True: 
                    cv2.rectangle(image_src, (x1, y1), (x2, y2), COLOR[int(labels[i])], thickness=max(
                        int((w + h) / 600), 1), lineType=cv2.LINE_AA)
                    label = '%s %.2f' % (CLASS_LABELS[int(labels[i])], confs[i])
                    t_size = cv2.getTextSize(
                        label, 0, fontScale=0.5, thickness=1)[0]
                    c2 = x1 + t_size[0] + 3, y1 - t_size[1] - 5
                    cv2.rectangle(image_src, (x1 - 1, y1), c2,
                                COLOR[int(labels[i])], cv2.FILLED, cv2.LINE_AA)
                    cv2.putText(image_src, label, (x1 + 3, y1 - 4), 0, 0.5, [255, 255, 255],
                                thickness=1, lineType=cv2.LINE_AA)
                    # print("bbox:", box, "conf:", confs[i],
                    #     "class:", CLASS_LABELS[int(labels[i])])
                    
        return container

    def infer_frame(self, frame):
        
        container=[]
        preprocess_func = partial(preprocess_image, in_size=(self.W, self.H)) # 实例一个裁切输入图像为输入层所要求宽高的函数
        orig_input = frame.copy()
        model_input = preprocess_func(frame) # 裁切画面

        start = time.time()
        results = self.OVExec.infer(inputs={self.InputLayer: model_input}) # 输入画面开始预测，并保存推理结果
        end = time.time()

        inf_time = end - start
        fps = 1. / inf_time
        text = 'FPS: {}, INF: {}'.format(round(fps, 2), round(inf_time, 2))
        cv2.putText(orig_input, text, (0, 20), cv2.FONT_HERSHEY_COMPLEX,
                    0.6, (0, 125, 255), 1)

        # 读取结果，数据形式转换，再进行非极大值抑制
        detections = results[self.OutputLayer] 
        detections = torch.from_numpy(detections)
        detections = non_max_suppression(
            detections, conf_thres=self.threshold, iou_thres=0.5, agnostic=False)
        
        self.output_process(detections[0],orig_input,self.threshold,(self.H,self.W),container) # 处理各识别框

        if self.showInfo == True:
            cv2.imshow('Detection', orig_input)
            cv2.waitKey(1)
        
        if len(container) == 0:
            return None
        else:    
            return np.vstack(container)


# # 调试用
# detector = vinodetect(
#     model='/Users/cionhuang/Documents/rc2024/Models/0426v5n/best_openvino_model/',
#     # model='/home/spr/rc2024/Models/0426v5n/best_openvino_model',
#     device='CPU',
#     threshold=0.65,
#     showinfo=True)

# cap0 = cv2.VideoCapture('/Users/cionhuang/Documents/rc2024/recordings/recording_001.avi')

# while True:

#     _,frame = cap0.read()
#     # _,frame = cap.read()

#     print(detector.infer_frame(frame))






    



