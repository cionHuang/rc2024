from vinodetect import vinodetect
import cv2
import numpy as np
import time


# vino初始化选项
# MODEL = "/home/spr/RC2024/Models/0426v5n/best_openvino_model/best.xml" # 模型地址
# MODEL = "/home/spr/RC2024/Utils/yolo_openvino/v5n_openvino_model/v5n.xml" # 模型地址
# MODEL = "/home/spr/RC2024/Models/0507v5n_pro/best_openvino_model/best.xml" # 模型地址
MODEL = "/Users/cionhuang/Documents/rc2024/Models/0426v5n/best_openvino_model/best.xml"

DEVICE = "CPU" # 设备
start = vinodetect(model_path=MODEL,device_name=DEVICE)
cap = cv2.VideoCapture('/Users/cionhuang/Documents/rc2024/recordings/recording_001.avi')
cap.set(cv2.CAP_PROP_AUTO_WB,0)
while True:
    sta  = time.time()
    _,frame = cap.read()
    start.main(frame)
    fps = 1. / (time.time()-sta)
    text = '{}'.format(round(fps,2))
    cv2.putText(frame, text, (0, 20), cv2.FONT_HERSHEY_COMPLEX,
                    0.6, (0, 125, 255), 1)
    cv2.imshow('test',frame)
    cv2.waitKey(1)

# import openvino as ov

# core = ov.Core()
# print(core.available_devices)


# from openvino.inference_engine import IECore

# # 创建IECore对象
# ie = IECore()

# # 获取可用设备列表
# available_devices = ie.available_devices

# # 打印设备列表
# for device in available_devices:
#     print(device)