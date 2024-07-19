# Extra pack for depth CAM
from openni import openni2
import numpy as np
import cv2
import serial
import time
# End

import argparse
import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadStreams
from utils.general import (LOGGER, Profile, check_img_size, check_requirements,
                           cv2, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

import math

cameraMatrix = np.array([
    [602.60620769, 0., 325.53789961],  # 内参
    [0., 602.664394, 248.45389531],
    [0., 0., 1.]
])

distCoeffs = np.array([
    -1.24723895e-01, 2.24796315e+00, 5.09514713e-04, -2.09903750e-03,
    -8.08329950e+00
])  # 畸变系数

yawCount = 0
yawVector = [0, 0, 0, 0, 0]


def digitArray(num):  # 为串口返回列表数组
    array = [0, 0, 0, 0,
             0]  # 1位：符号、校验位 ； 2位：yaw的千位 ：3为：yaw百位 类推     1负2正3错误4成功
    digitArray = [int(digit) for digit in str(abs(num))]
    lenth = len(digitArray)
    if lenth > 4:
        array = [3, 0, 0, 0, 0]  # 若数组长度大于4,说明yaw结算出错，返回首位为3的数组
        return array
    else:
        if num < 0:
            array[0] = 1  # 若小于0,返回的数组首位为1
            for i in range(lenth):
                array[4 - i] = digitArray[lenth - 1 - i]
            return array
        else:
            array[0] = 2  # 若大于0,返回的数组首位为2
            for i in range(lenth):
                array[4 - i] = digitArray[lenth - 1 - i]
            return array


def movingAverageFilter(data, windowSize):  # 移动平均滤波
    filteredData = []
    for i in range(len(data)):
        if i < windowSize:
            filteredData.append(data[i])
        else:
            average = sum(data[i - windowSize:i]) / windowSize
            filteredData.append(average)
    return sum(filteredData) / len(filteredData)


# def addArray(array,newElement):
#     array = [newElement]+array[:4]
#     return array


def depthFinder(cx, cy, width, height,
                depthArray):  # 如果中心点没有深度信息，就向中心点的下面遍历寻找有值的深度，目前来看一般是上方缺失深度信息
    depthData = 0  # 存储深度信息
    dx = width // 20  # x方向的遍历步长
    dy = height // 20  # y方向的遍历步长
    # x = cx
    # y = cy
    flag = 0
    for y in range(cy, cy + height // 2, dy):  # 遍历中点的右下方，此区域可以调整，看具体情况
        for x in range(cx, cx + width // 2, dx):
            if (depthArray[y, x] / 1000.0) != 0:
                depthData = depthArray[y, x] / 1000.0
                flag = 1
                break
        if flag == 1:
            break
    if flag == 0:
        return -1
    else:
        print('!!!!!!!!!!!!!!!!成功补充深度值!!!!!!!!!!!!!!!!!!!11')
        return depthData


def coordinate(xy, d):  # 像素坐标转世界坐标，并进行yaw结算
    # 世界坐标
    x = (xy[0] - cameraMatrix[0, 2]) * d / cameraMatrix[0, 0]
    y = (xy[1] - cameraMatrix[1, 2]) * d / cameraMatrix[1, 1]
    z = d

    yaw = math.degrees(math.atan(x / z))

    # print([x,y,z],'yaw=',yaw)

    return yaw


def mainFunction(arrary, flag, defaultPic, ser, depthArrary):  # 判断是否对正
    # print('######################################################################################')
    # order = -1
    # while order == -1 : # find T12
    global yawCount
    global yawVector
    signal = -1
    if len(arrary) > 0:  # 如果这一帧有框（7个信息）返回，开始运行
        arrary = sorted(
            arrary, key=lambda x: abs(x[0] - 320))  # 按照距离中心的角度排序，且只对最近的一个进行结算

        ######################yaw的结算判定##########################
        if arrary[0][6] > 0:
            dist = arrary[0][6]
            yaw = round(coordinate([arrary[0][0], arrary[0][1]], dist), 2)
            signal = 1
        elif arrary[0][6] == 0:
            dist = depthFinder(arrary[0][0], arrary[0][1], arrary[0][2],
                               arrary[0][3], depthArrary)
            if dist > 0:  # 成功获取缺失的深度值
                yaw = round(coordinate([arrary[0][0], arrary[0][1]], dist), 2)
                signal = 1
            else:
                print(
                    "#############################################################测距失效1"
                )  #  识别框内没有可用的深度信息
                ser.write([3, 0, 0, 0, 0])
        elif arrary[0][6] < 0:
            print(
                '#############################################################测距失效2'
            )  # 深度摄像头异常
            ser.write([3, 0, 0, 0, 0])

        ##########################滤波与串口通信########################
        if signal == 1:
            if abs(yaw) > 0.2:  # 大于0.2度视为没有对正，该参数可调
                # print(flag,'未对正','yaw=',yaw)
                if yawCount < 5:
                    yawVector.append(yaw)
                    yawCount += 1
                else:
                    print(
                        '######################################################################################'
                    )
                    filteredYaw = round(movingAverageFilter(yawVector, 2), 2)
                    print(yawVector)
                    print(filteredYaw)
                    yawVector = []
                    yawCount = 0
                    filteredYaw = int(filteredYaw * 100)
                    yawArray = digitArray(filteredYaw)
                    print(yawArray)
                    ser.write(yawArray)
                    # time.sleep(3)
                    print(
                        '######################################################################################'
                    )
            else:
                print(
                    flag,
                    '######################################################对正',
                    'yaw=', yaw)
                ser.write([4, 0, 0, 0, 0])
                cv2.rectangle(defaultPic, arrary[0][4], arrary[0][5],
                              (0, 255, 0), 2)  # 可注释
                return defaultPic
        # else :
        #     print(flag,'#############################################################测距失效')
        #     ser.write([3,0,0,0,0])
    else:
        print(flag, '未识别到')
        ser.write([3, 0, 0, 0, 0])

    # print('########################################################################################')


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold  置信度可调
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):

    # 一些初始化数据
    xPositionT12 = -1
    yPositionT12 = -1
    # xPositionT3 = -1
    # yPositionT3 = -1
    widthT12 = -1
    heightT12 = -1
    # widthT3 = -1
    # heightT3 = -1
    arraryT12 = []
    # arraryT3 = []
    # filterArray = []
    # count = 0

    ser = serial.Serial('/dev/ttyUSB0', 115200)  # 实例化一个串口

    # print(weights)
    # print(conf_thres)

    ##################################################################OpenNi###########################################

    # Openni Initialize 初始化深度摄像机
    openni2.initialize()
    dev = openni2.Device.open_any()

    # Set video mode & Create videostream
    depth_stream = dev.create_depth_stream()
    depth_stream.set_video_mode(
        openni2.VideoMode(pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM,
                          resolutionX=640,
                          resolutionY=480,
                          fps=30))
    depth_stream.start()
    if dev.is_image_registration_mode_supported(
            openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR):
        dev.set_image_registration_mode(
            openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    # End

################################################################################################################3
# 判断源的格式
    source = str(source)  # 数据输入，配置中的路径

    # Load model 加载模型的权重
    device = select_device(device)  # 选择加载模型的硬件
    model = DetectMultiBackend(weights,
                               device=device,
                               dnn=dnn,
                               data=data,
                               fp16=half)  # 判断后端框架的权重加载方式
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader 加载待预测的数据
    bs = 1  # batch_size 一次传入的数据数量

    dataset = LoadStreams(source,
                          img_size=imgsz,
                          stride=stride,
                          auto=pt,
                          vid_stride=vid_stride)
    bs = len(dataset)

    # Run inference 执行数据推理
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3,
                        *imgsz))  # warmup 初始化了一张空白图片，让参数初始值较小，便于后期训练更快的收敛
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # frep = cv2.getTickFrequency()  # 显示帧率步骤1

    for path, im, im0s, vid_cap, s in dataset:  # 开始遍历

        # startTime = cv2.getTickCount() # 显示帧率步骤2

        with dt[0]:
            im = torch.from_numpy(im).to(
                model.device)  # resize后的图片，并从numpy的格式转换为pytorch支持的格式
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0  归一化
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference 推测
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)

        # NMS 非极大值抑制，过滤掉多余的框，关键参数 conf_thres、iou_thres
        with dt[2]:
            pred = non_max_suppression(pred,
                                       conf_thres,
                                       iou_thres,
                                       classes,
                                       agnostic_nms,
                                       max_det=max_det)
            # print('########################',pred)  # pred中包含所有框的对角点坐标，置信度，类别，以及cuda的设备

        # Process predictions
        for i, det in enumerate(pred):  # per image  第二层：一帧的每个框

            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{i}: '

            # for saving defualt pic
            defaultPic = im0.copy()

            annotator = Annotator(im0,
                                  line_width=line_thickness,
                                  example=str(names))  # 绘图工具Annotator

            ######################################################Depth And Arrary####################################################################
            depth_frame = depth_stream.read_frame()
            depth_data = depth_frame.get_buffer_as_uint16()
            depth_array = np.ndarray((depth_frame.height, depth_frame.width),
                                     dtype=np.uint16,
                                     buffer=depth_data)

            depth_array = cv2.flip(depth_array, 1)  # 所有深度信息的数组
            # Convert depth frame to CV_8U format for visualization
            max_depth = depth_stream.get_max_pixel_value()  #最大深度值，单位mm 可注释
            # print(max_depth)
            depth_visual = cv2.convertScaleAbs(depth_array,
                                               alpha=255 / max_depth)  # 可注释
            ############################################################################################################################################

            if len(det):  # 所有框的信息
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                         im0.shape).round()  # 原图中的边界信息
                # print(det)

                # Write results
                for *xyxy, conf, cls in reversed(
                        det):  # 此处 int(cls) 的值就是对应的class  第三层：获取每个框的信息
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
                        depth_valueT12 = depth_array[
                            yPositionT12,
                            xPositionT12] / 1000.0  # 识别框中点位置的深度信息
                        a = [
                            xPositionT12, yPositionT12, widthT12, heightT12,
                            LUT12_2, RDT12_2, depth_valueT12
                        ]  # 将以上数据打包并存储在arraryT12数组内z
                        arraryT12.append(a)

                    # if int(cls) == 1 :
                    #     # print(list(map(int,((xyxy2xywh(torch.tensor(xyxy))).view(-1).tolist()))))
                    #     xPositionT3 = list(map(int,((xyxy2xywh(torch.tensor(xyxy))).view(-1).tolist())))[0] # Extra process:get the information of the box
                    #     yPositionT3 = list(map(int,((xyxy2xywh(torch.tensor(xyxy))).view(-1).tolist())))[1]
                    #     widthT3 = list(map(int,((xyxy2xywh(torch.tensor(xyxy))).view(-1).tolist())))[2]
                    #     heightT3 = list(map(int,((xyxy2xywh(torch.tensor(xyxy))).view(-1).tolist())))[3]
                    #     LUT3_2 = [list(map(int,(torch.tensor(xyxy).view(-1).tolist())))[0],list(map(int,(torch.tensor(xyxy).view(-1).tolist())))[1]]
                    #     RDT3_2 = [list(map(int,(torch.tensor(xyxy).view(-1).tolist())))[2],list(map(int,(torch.tensor(xyxy).view(-1).tolist())))[3]]
                    #     depth_valueT3 = depth_array[yPositionT3,xPositionT3] / 1000.0
                    #     b = [xPositionT3,yPositionT3,widthT3,heightT3,LUT3_2,RDT3_2,depth_valueT3]
                    #     arraryT3.append(b)

                    c = int(cls)  # integer class
                    label = None if hide_labels else (
                        names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

#########################################################################################################

##################################################################################################################

# Stream results
            im0 = annotator.result()  # 带有yolo检测框的图像

            cv2.imshow(str(p), im0)
            # cv2.waitKey(1)  # 1 millisecond
            # flag = (ser.read(1)).decode()
            # flag = 'A'
            # if flag == 'A':
            mainFunction(arraryT12, 'T12', defaultPic, ser, depth_array)
            # if flag == 'B':
            # mainFunction(arraryT3,'T3',defaultPic,ser)


#########################################################################################################################
        xPositionT12 = -1
        yPositionT12 = -1
        # xPositionT3 = -1
        # yPositionT3 = -1
        widthT12 = -1
        heightT12 = -1
        # widthT3 = -1
        # heightT3 = -1
        arraryT12 = []
        # arraryT3 = []

        # 显示帧率步骤3

        # endTime = cv2.getTickCount()
        # time = (endTime - startTime) /frep
        # fps =1 / time
        # print('fps:',fps)

        cv2.imshow('depth image', depth_visual)
        # cv2.imshow('test',defaultPic)
        # cv2.waitKey(1)

    ser.close()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        nargs='+',
                        type=str,
                        default=ROOT / 'best_sim.onnx',
                        help='model path or triton URL')  # 训练权重
    parser.add_argument('--source',
                        type=str,
                        default='0',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data',
                        type=str,
                        default=ROOT / 'data/myvoc.yaml',
                        help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz',
                        '--img',
                        '--img-size',
                        nargs='+',
                        type=int,
                        default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.35,
                        help='confidence threshold')  # 置信度
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.45,
                        help='NMS IoU threshold')  # iou
    parser.add_argument('--max-det',
                        type=int,
                        default=1000,
                        help='maximum detections per image')
    parser.add_argument('--device',
                        default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt',
                        action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf',
                        action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop',
                        action='store_true',
                        help='save cropped prediction boxes')
    parser.add_argument('--nosave',
                        action='store_false',
                        help='do not save images/videos')
    parser.add_argument(
        '--classes',
        nargs='+',
        type=int,
        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms',
                        action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment',
                        action='store_true',
                        help='augmented inference')
    parser.add_argument('--visualize',
                        action='store_true',
                        help='visualize features')
    parser.add_argument('--update',
                        action='store_true',
                        help='update all models')
    parser.add_argument('--project',
                        default=ROOT / 'runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name',
                        default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness',
                        default=3,
                        type=int,
                        help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels',
                        default=False,
                        action='store_true',
                        help='hide labels')
    parser.add_argument('--hide-conf',
                        default=True,
                        action='store_true',
                        help='hide confidences')
    parser.add_argument('--half',
                        action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--dnn',
                        action='store_true',
                        help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride',
                        type=int,
                        default=1,
                        help='video frame-rate stride')
    opt = parser.parse_args()  # 存储并返回配置信息
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()  # 检测命令行输入的配置信息
    main(opt)
