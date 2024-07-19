import time
from scipy.spatial.transform import Rotation as R
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import numpy as np
import cv2
from openvino.runtime import Core, Model
from typing import Tuple, Dict
import random
from ultralytics.utils import ops
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from ultralytics.utils import ROOT, yaml_load
from ultralytics.utils.checks import check_yaml
from unity_server import Armro_Unity_Server#unity通信类
from WebcamVideoStream import WebcamVideoStream#双线程读取视频类
from Predict.kalmanfilter import MovingAverageFilter,KalmanFilter_low#卡尔曼预测类
from position_solver import PositionSolver
import math
# from my_serial import SerialPort
from inverse_transformation import CameraCalibration
matplotlib.use('TkAgg')
from scipy.optimize import minimize
from test_light import lighter_correct

#储存pitch角度
pitch=[]

#标签和关键点
label_map = []
kpt_shape: Tuple = []

#可视化处理结果
show_result = True

#储存滤波结果的
t_matrix2=[]
t_matrix2_lvbo=[]
t_matrix2_kal=[]
dx_list=[]
dx_list_lvbo=[]
world_position=[]
anti_top=[]


#接收上一帧数据的列表
last_center=0
last_x=[]
last_z=[]


#储存速度列表
speed_x=[]
speed_z=[]
speed_x_lvbo_kal=[]
speed_x_lvbo_low=[]
speed_absolute=[]
speed_gimbal=[]


#接受时间的列表
time_list=[]

#跟随标志位
flag1=0#预测
flag2=0#跟随

#电控发来的角度
gimbal_angle_list=[]



# 初始化空的数据列表
yaw_data = []
iou_data = []
# 创建初始的Matplotlib图形
fig, ax = plt.subplots()
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
#设置 y 轴范围
y_min = -1# 设置最小值
y_max = 1 # 设置最大值

# x_min=-500
# x_max=500
ax.set_ylim(y_min, y_max)
# ax.set_xlim(x_min, x_max)

line, = ax.plot(yaw_data, iou_data, marker='o', linestyle='-', markersize=5)

#设置储存的缓冲区大小
buffer_size = 100

def update_plot(yaw, iou):
    yaw_data.append(yaw)
    iou_data.append(iou)

    # 如果数据超过缓冲区大小，移除最旧的数据
    if len(yaw_data) > buffer_size:
        yaw_data.pop(0)
        iou_data.pop(0)
    
    # 更新Matplotlib图形数据
    line.set_data(yaw_data, iou_data)
    
    # 重新设置图形的x轴范围，可以根据需要进行调整
    ax.relim()
    ax.autoscale_view()
    
    # 更新图形
    plt.draw()
    plt.pause(0.01)  # 稍微暂停以显示更新

# 示例函数接受两个参数并更新图形
def receive_and_plot(yaw, iou):
    update_plot(yaw, iou)



#计算速度的函数
def calculate_speed(dx,dt):
    relative_speed=dx/1000/dt
    return relative_speed

#调用低通滤波器
def low_filter(filter,x):
    
    x=filter.smooth(x)
    return x

#调用卡尔曼滤波器
def kal_filter(filter,x):
    x=filter.filter(x)
    return x

#目前采取的后一帧代替前一帧的方法，后面可以改为删掉第一帧，最后加一帧
def update(list_,new_value,length):
    for i in range(length-1):
        list_[i]=list_[i+1]
    list_[length-1]=new_value
    

#针对每一个检测框，检测框的数量nms有关
def process_data_img(box: np.ndarray, keypoints:np.ndarray, img: np.ndarray, color: Tuple[int, int, int] = None, label: str = None, line_thickness: int = 5):
    """
    Helper function for drawing single bounding box on image
    Parameters:
        x (np.ndarray): bounding box coordinates in format [x1, y1, x2, y2]
         (no.ndarray): input imagea
        color (Tuple[int, int, int], *optional*, None): color in BGR format for drawing box, if not specified will be selected randomly
        mask (np.ndarray, *optional*, None): instance segmentation mask polygon in format [N, 2], where N - number of points in contour, if not provided, only box will be drawn
        label (str, *optonal*, None): box label string, if not provided will not be provided as drowing result
        line_thickness (int, *optional*, 5): thickness for box drawing lines
    """
    #开火指令
    fire=0
    #小陀螺标志位
    anti_top_flag=0
    #上一帧装甲板的中心点
    global last_center,outpost_center
    
    #储存关键点的数组,默认顺序是左上，右上，右下，左下
    points_2D=np.empty([0,2],dtype=np.float64)
      
    # #相机参数矩阵
    # cameraMatrix = np.array([[2075.23100, 0, 624.212611],
    #                         [ 0, 2073.82280, 514.674148],
    #                          [0, 0, 1]], dtype=np.float64)
    # distCoeffs=np.array([[-0.09584139, 1.10498131, -0.00723334675, -0.00165270614, -8.01363788]],dtype=np.float64)
    
    # #相机参数矩阵(shaobing)
    # cameraMatrix = np.array([[2087.421950, 0, 640.0],
    #                         [ 0, 2087.421950, 512.0],
    #                          [0, 0, 1]], dtype=np.float64)
    # distCoeffs=np.array([[-0.075394, 0.509417, -0.694177, 0.0, 0.0]],dtype=np.float64)
    
    
    #相机参数矩阵(bubing)
    cameraMatrix = np.array([[1550.135539, 0, 640.0],
                            [ 0, 1550.135539, 512.0],
                             [0, 0, 1]], dtype=np.float64)
    distCoeffs=np.array([[-0.088273, 0.55129, 0.005279, 0.0, 0.0]],dtype=np.float64)


    #画框
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    #cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    #解包标签信息
    if label:

        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        #cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        #cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    #[225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
    #解包keypoints信息，添加关键点到二维点
    for i, k in enumerate(keypoints):
            if(k.dim()!=0):
                color_k = color or [random.randint(0, 255) for _ in range(3)]
                x_coord, y_coord = k[0], k[1]#得到x,y的坐标
                #                                1280/416=3.07692  1024/416=2.461538
                points_keypoint=np.array([int(x_coord*3.07692),int(y_coord*2.461538)],dtype=np.float64)#作为二维点进行储存
                points_2D=np.vstack((points_2D,points_keypoint))
                


                if x_coord % img.shape[1] != 0 and y_coord % img.shape[0] != 0:
                    if len(k) == 3:
                        conf = k[2]
                        if conf < 0.6:
                            continue
                        

    if(np.shape(points_2D)==(4,2)):
        ###########################################################判断四个角点姿态，剔除异常装甲板######################################################
        
 
        #############################################对正常装甲板进行处理###################################################################
        armor_box=np.array([points_2D[0],points_2D[1],points_2D[2],points_2D[3]],dtype=np.float64)

        #以四个角点为基准，创建roi区域
        x_min=(armor_box[0][0])/3.07692/1.1
        x_max=(armor_box[2][0])/3.07692*1.1
        y_min=(armor_box[0][1])/2.461538/1.1
        y_max=(armor_box[2][1])/2.461538*1.1
        roi=img[int(y_min):int(y_max),int(x_min):int(x_max)]

        # #将roi区域放大两倍
        # roi=cv2.resize(roi,(roi.shape[1]*2,roi.shape[0]*2))

        #print("开始矫正")
        roi=cv2.resize(roi,(640,640))
        roi_convert=cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
        lighter_angle=lighter_corrector.extract_red_light(roi_convert)

    
        for i in range(4):
            cv2.circle(img, (int(armor_box[i][0]/3.07692), int(armor_box[i][1]/2.461538)), 2, (255,0,0), -1,lineType=cv2.LINE_AA)

            cv2.line(img, (int(armor_box[i][0]/3.07692), int(armor_box[i][1]/2.461538)), (int(armor_box[(i+1)%4][0]/3.07692), int(armor_box[(i+1)%4][1]/2.461538)), color, 1)

        
    

        ###########################################筛选大小装甲板###################################################################
        #print("灯条比值",(points_2D[1][0]-points_2D[0][0])/(points_2D[2][1]-points_2D[1][1]))
        if((points_2D[1][0]-points_2D[0][0])/(points_2D[2][1]-points_2D[1][1])>=5):
            #print("大装甲板")
            #大装甲板尺寸
            points_3D=np.array([[-114.0, -26.0, 0],
                            [114.0,-26.0, 0],
                            [114.0,26.0, 0],
                            [-114.0,26.0, 0]], dtype=np.float64)
            
        elif((points_2D[1][0]-points_2D[0][0])/(points_2D[2][1]-points_2D[1][1])<5):
            #print("小装甲板")
            #小装甲板尺寸,默认顺序是左上，右上，右下，左下,中心
            points_3D = np.array([[-66.0,-25.0, 0],
                            [66.0,-25.0, 0],
                            [66.0,25.0, 0],
                            [-66.0,25.0, 0]], dtype=np.float64)
        
                    
        ######################################################进行坐标结算###################################################################

        P_matrix,T_matrix=po.my_pnp(points_3D,points_2D,cameraMatrix,distCoeffs)
        #print("测距",T_matrix[2])

        #########################################################反前哨站算法(反投影版)###########################################
        
        #目前yaw角度给的0，yaw应为自变量，yaw的取值影响反投影结果,pitch给15，和roll角度给的0
        #三个值依次为pitch，yaw，roll
        rotation_vector = np.array([np.radians(20),np.radians(0),0], dtype=np.float32)

        translation_vector=np.array([[0], [0] ,[0]], dtype=np.float64)
        translation_vector[0]=T_matrix[0]
        translation_vector[1]=T_matrix[1]
        translation_vector[2]=T_matrix[2]
            
        # 设置参数
        calibrator.set_parameters(points_3D, cameraMatrix, rotation_vector, translation_vector)
        
        # # 获取投影后的图像点
        #image_points = calibrator.project_points(0)
        # print("反投影后维点",image_points)
        
        #image_points = calibrator.project_points(0)
        #image_points=calibrator.project_points_3d_to_2d()

        #time1=time.time()
        best_yaw=calibrator.find_best_yaw(points_2D,lighter_angle)#寻找当前装甲板最合适的yaw朝向角
        #best_yaw=0
        #best_yaw=calibrator.find_best_yaw_phi_search(points_2D)#寻找当前装甲板最合适的yaw朝向角
        #time2=time.time()
        #print("寻找最佳yaw时间",time2-time1)
        #print("best_yaw滤波前",best_yaw)

        #对最佳yaw角度进行滤波
        best_yaw=low_filter(filter_x,best_yaw)
        #best_yaw=kal_filter(kal_filter_x,best_yaw)
        # best_yaw=low_filter(filter_x,best_yaw)
        #best_yaw=-math.pi / 4
        #print("best_yaw滤波后",best_yaw)

        #假设子弹飞行时间为0.5s
        #best_yaw=best_yaw+0.5*0.4

        image_points = calibrator.project_points(best_yaw)#反投影后的图像点

        #对敌方车辆进行建模可视化
        if(best_yaw<math.pi/4 and best_yaw>-math.pi/4):
                
            #unity_server.set_parameters(translation_vector[0][0]/100,translation_vector[1][0]/100,translation_vector[2][0]/100,best_yaw)
    
            #画出反投影点构成的矩形
            for i in range(4):
                cv2.circle(img, (int(image_points[i][0]/3.07692), int(image_points[i][1]/2.461538)), 2, (0,255,0), -1,lineType=cv2.LINE_AA)
                cv2.line(img, (int(image_points[i][0]/3.07692), int(image_points[i][1]/2.461538)), (int(image_points[(i+1)%4][0]/3.07692), int(image_points[(i+1)%4][1]/2.461538)), (0,255,0), 1)
        
        print("最佳yaw",best_yaw)
        
        
    
    return img

def analyse_results(results: Dict, source_image: np.ndarray, label_map: Dict):
    """
    Helper function for drawing bounding 
    # plt.subplot(313)
    # plt.plot(pitch,label = "pitch")
    # plt.xlabel("time")
    # plt.ylabel("speed")
    # plt.ylim(-1,1)
    # plt.legend() 
    # plt.show() boxes on image
    Parameters:
        image_res (np.ndarray): detection predictions in format [x1, y1, x2, y2, score, label_id]
        source_image (np.ndarray): input image for drawing
        label_map; (Dict[int, str]): label_id to class name mapping
    Returns:

    """
    boxes = results["det"]
    keypoints = results["keypoints"]
    
    h, w = source_image.shape[:2]
    
    for idx, (*xyxy, conf, lbl) in enumerate(boxes): 
        
        #让每帧图像只有一个输出结果
        if idx == 1:
            break 
        if(np.shape(keypoints)!=[]):
            if(np.shape(keypoints[0]>idx)):
                if(np.shape(keypoints[idx])!=torch.Size([])):#防止keypoints和boxes之间数量不匹配造成的报错
                    #label = f'{label_map[int(lbl)]} {conf:.2f}'
                    label = f'{label_map[int(lbl)]}'
                    *keypoints, = keypoints[idx]
                    #进行最终结算
                    source_image = process_data_img(
                        xyxy, keypoints, source_image, label=label, color=colors(int(lbl)), line_thickness=1)
                    

    return source_image


def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (416, 416), color: Tuple[int, int, int] = (114, 114, 114), auto: bool = False, scale_fill: bool = False, scaleup: bool = False, stride: int = 32):
    """
    Resize image and padding for detection. Takes image as input,
    resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints

    Parameters:
      img (np.ndarray): image for preprocessing
      new_shape (Tuple(int, int)): image size after preprocessing in format [height, width]
      color (Tuple(int, int, int)): color for filling padded area
      auto (bool): use dynamic input size, only padding for stride constrins applied
      scale_fill (bool): scale image to fill new_shape
      scaleup (bool): allow scale image if it is lower then desired input size, can affect model accuracy
      stride (int): input padding stride
    Returns:
      img (np.ndarray): image after preprocessing
      ratio (Tuple(float, float)): hight and width scaling ratio
      padding_size (Tuple(int, int)): height and width padding size
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def preprocess_image(img0: np.ndarray):
    """
    Preprocess image according to YOLOv8 input requirements.
    Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

    Parameters:
      img0 (np.ndarray): image for preprocessing
    Returns:
      img (np.ndarray): image after preprocessing
    """
    # resize
    img = letterbox(img0)[0]

    # Convert HWC to CHW
    if(np.shape(img)==(416,416,3)):
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
    
    return img


def image_to_tensor(image: np.ndarray):
    """
    Preprocess image according to YOLOv8 input requirements.
    Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

    Parameters:
      img (np.ndarray): image for preprocessing
    Returns:
      input_tensor (np.ndarray): input tensor in NCHW format with float32 values in [0, 1] range
    """
    input_tensor = image.astype(np.float32)  # uint8 to fp32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

    # add batch dimension
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor


def postprocess(
    preds: np.ndarray,
    input_hw: Tuple[int, int],
    orig_img: np.ndarray,
    min_conf_threshold: float = 0.3,
    nms_iou_threshold: float = 0.7,
    agnosting_nms: bool = False,
    max_detections: int = 300,
):        # parser.add_argument('--model', default="/home/rc-cv/yolov8-face-main/best_openvino_model/outpost.xml",
    """
    YOLOv8 model postprocessing function. Applied non maximum supression algorithm to detections and rescale boxes to original image size
    Parameters:
        preds (np.ndarray): model output prediction boxes and keypoints in format [x1, y1, x2, y2, score, label, keypoints_x, keypoints_y, keypoints_visible]
        input_hw (np.ndarray): preprocessed image
        orig_image (np.ndarray): image before preprocessing
        min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
        nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
        agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
        max_detections (int, *optional*, 300):  maximum detections after NMS
    Returns:
       pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label] and segment - segmentation polygons for each element in batch
    """
    nms_kwargs = {"agnostic": agnosting_nms, "max_det": max_detections}
    preds = ops.non_max_suppression(
        torch.from_numpy(preds),
        min_conf_threshold,
        nms_iou_threshold,
        nc=len(label_map),
        **nms_kwargs
    )

    #打印预测结果的尺寸信息
    #print(preds)


    results = []
    for i, pred in enumerate(preds):
        shape = orig_img[i].shape if isinstance(
            orig_img, list) else orig_img.shape
        if not len(pred):
            results.append({"det": [], "keypoints": []})
            continue
        else:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            pred_kpts = pred[:, 6:].view(
                len(pred), *kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(input_hw, pred_kpts, shape)
            results.append(
                {"det": pred[:, :6].numpy(), "keypoints": pred_kpts})
    
    #print(results)
    
    return results


def detect(image: np.ndarray, model: Model):
    """
    OpenVINO YOLOv8 model inference function. Preprocess image, runs model inference and postprocess results using NMS.
    Parameters:
        image (np.ndarray): input image.
        model (Model): OpenVINO compiled model.
    Returns:
        detections (np.ndarray): detected boxes in format [x1, y1, x2, y2, score, label]
    """
    preprocessed_image = preprocess_image(image)
    input_tensor = image_to_tensor(preprocessed_image)
    result = model(input_tensor)
    preds = result[model.output(0)]
    # print(preds)
    # print("xxxxx")
    input_hw = input_tensor.shape[2:]
    detections = postprocess(
        preds=preds, input_hw=input_hw, orig_img=image)
    
    return detections


def main(openvino_model, cap):

    global label_map, kpt_shape
    # Load YOLOv8 model for label map, if you needn't display labels, you can remove this part
    # if(port.enemy_color==1):
    #yolo_model = YOLO("D:\\source\\YOLO\\yolo_hero\\red_fina.pt")
    yolo_model=YOLO("D:\\source\\YOLO\\yolov8-face-main\\yolov8-face-main\\runs\\pose\\outpost_only_red3\\best.pt")
    
    # else:
        # yolo_model = YOLO("/home/rc-cv/yolov8-face-main/red_fina.pt")
        
           
    label_map = yolo_model.model.names
    kpt_shape = yolo_model.model.kpt_shape

    core = Core()

    # Load a model
    # Path to the model's XML file
    model = core.read_model(model=openvino_model) 
    compiled_model = core.compile_model(model=model, device_name="CPU")
    input_layer = compiled_model.input(0)
    
    while(1):
        time1=time.time()
        #input_image=cap.read()
        ok,input_image=cap.read()
        #input_image=cv2.imread("D:\\source\\YOLO\yolov8-face-main\\yolov8-face-main\\new_dataset\\images\\train\\6389.0.jpg")
        input_image = cv2.resize(input_image, (416, 416))
        #data=share_queue.shared_queue_back.get()

        #input_image=cap.read()
        # if(np.shape(input_image)==(416,416,3)):
        #input_image = cv2.GaussianBlur(input_image, (3, 3), 0)
        input_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        # input_image = cv2.imread(input_path)
        #input_image = np.array(Image.open(input_path))
        detections = detect(input_image, compiled_model)[0]

        

        boxes = detections["det"]
        keypoints = detections["keypoints"]

        #自瞄处理程序
        if(show_result):
            if ((np.shape(boxes)) and (np.shape(keypoints))):
                image_with_boxes = analyse_results(detections, input_image, label_map)
                #可视化处理结果
                # Image.fromarray(image_with_boxes).show()
                image_with_boxes=cv2.cvtColor(image_with_boxes,cv2.COLOR_RGB2BGR)
                cv2.imshow(" ",image_with_boxes)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            else:
                image_with_boxes = analyse_results(detections, input_image, label_map)
                image_with_boxes=cv2.cvtColor(image_with_boxes,cv2.COLOR_RGB2BGR)
                cv2.imshow(" ",image_with_boxes)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                

        time2=time.time()
                 
        print("运行帧率：",1/(time2-time1))
        #print(anti_top_flag)



#初始化坐标结算器
po=PositionSolver()

calibrator = CameraCalibration()

#初始化滤波器
filter_x=MovingAverageFilter(5)#测距滑动平均滤波器
filter_z=MovingAverageFilter(10)#测距滑动平均滤波器
filter_vx=MovingAverageFilter(4)#速度滑动平均滤波器
filter_vz=MovingAverageFilter(4)#速度滑动平均滤波器
kal_filter_x=KalmanFilter_low()


if __name__ == '__main__':
    #初始化unity可视化
    #unity_server=Armro_Unity_Server().start()

    # port=SerialPort().start()
    #初始化相机    
    #cap=WebcamVideoStream().start()
    cap=cv2.VideoCapture("D:\\MindVision\\Video.mp4")
    
    parser = argparse.ArgumentParser()

    lighter_corrector=lighter_correct()
    
    # print("敌方颜色",port.enemy_color)
    # if(port.enemy_color==1 or port.enemy_color==3):
    #     parser.add_argument('--model', default="/home/rc-cv/yolov8-face-main/best_openvino_model/blue_fina.xml",
    #                 help='Input your openvino model.')
        
    # else:
    # parser.add_argument('--model', default="D:\\source\\YOLO\yolo_hero\\best_openvino_model\\red_fina.xml",
    #                  help='Input your openvino model.')

    parser.add_argument('--model', default="D:\\source\\YOLO\\yolov8-face-main\\yolov8-face-main\\runs\\pose\\outpost_only_red3\\best.xml",
                help='Input your openvino model.')

    args = parser.parse_args()
    

    #主程序
    main(args.model, cap)

