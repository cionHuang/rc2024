import threading
import time
import cv2
import os
import numpy as np
from Utils import mvsdk
from openni import openni2

from collections import deque

# 类定义
class SharedData:
    """
    串口数据共享类，以对全局共享串口信息
    """
    def __init__(self):
        self.flag = 0
        self.yaw = 0
        self.endCode = 0
        self.redArray = []
        self.blueArray = []
        self.dist_vis = None
        self.distArray = None

    
class CameraBufferCleanerThread(threading.Thread):
    """
    摄像头的线程读取类，防止帧堆积而造的画面延迟
    """
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()


class DelayedAverageExecution:
    """
    延迟平均值执行类，用于对函数的参数进行平均值处理，不是最优
    """
    def __init__(self, n, func, threshold_a, threshold_b):
        self.n = n
        self.func = func
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.calls = []
        self.last_args = None

    def __call__(self, *args, **kwargs):
        if self.last_args is not None:
            diff_a = abs(args[0] - self.last_args[0])
            diff_b = abs(args[1] - self.last_args[1])
            if diff_a >= self.threshold_a or diff_b >= self.threshold_b:
                print("Call ignored due to threshold limits.")
                return None, False

        self.calls.append((args, kwargs))
        self.last_args = args
        
        if len(self.calls) > self.n:
            # 计算前n次和第n+1次的平均值
            accumulated_args = [0] * len(args)
            for call in self.calls:
                for i in range(len(call[0])):
                    accumulated_args[i] += call[0][i]
            average_args = [x / (self.n + 1) for x in accumulated_args]
            
            # 清空调用记录
            self.calls = []
            
            # 调用实际函数
            result = self.func(*average_args, **kwargs)
            return result, True

        return None, False


class MultiDataFrameFilter:
    """
    多帧滤波器，用于对多帧信息进行平均值滤波
    """
    def __init__(self, data_frame_count, threshold_a, threshold_b):
        self.data_frame_count = data_frame_count
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.frames = deque(maxlen=data_frame_count)

    def reset(self):
        self.frames.clear()

    # 添加数据帧
    def add_data_frame(self, frame):
        self.frames.append(frame)

    # 滤波操作
    def get_filtered_data(self):
        if len(self.frames) == self.data_frame_count:
            frames_array = np.array(self.frames)
            median_frame = np.median(frames_array, axis=0)

            # 计算每帧数据与中位数的差异
            deviations = np.abs(frames_array - median_frame)

            # 过滤掉差异较大的帧
            filtered_frames = [frame for frame, deviation in zip(frames_array, deviations)
                               if deviation[0] <= self.threshold_a and deviation[1] <= self.threshold_b]

            # 如果所有帧都被过滤掉了，返回 None
            if not filtered_frames:
                self.reset()
                return None

            # 计算过滤后的均值
            filtered_frame = np.mean(filtered_frames, axis=0)
            self.reset()  # 重置变量
            return filtered_frame
        else:
            return None


class mindvisionCamera:
    """
    迈德威视相机类
    """
    def __init__(self, camera_index, exposure_time):
        # 枚举相机
        self.DevList = mvsdk.CameraEnumerateDevice()
        self.nDev = len(self.DevList)
        if self.nDev < 1:
            raise Exception("No camera was found!")

        if camera_index >= self.nDev:
            raise Exception(f"Camera index {camera_index} is out of range. Only {self.nDev} cameras found.")

        self.DevInfo = self.DevList[camera_index]
        print(f"Opening camera {camera_index}: {self.DevInfo.GetFriendlyName()} {self.DevInfo.GetPortType()}")

        # 打开相机
        self.hCamera = 0
        try:
            self.hCamera = mvsdk.CameraInit(self.DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            raise Exception(f"CameraInit Failed({e.error_code}): {e.message}")

        # 获取相机特性描述
        self.cap = mvsdk.CameraGetCapability(self.hCamera)

        # 判断是黑白相机还是彩色相机
        self.monoCamera = (self.cap.sIspCapacity.bMonoSensor != 0)

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if self.monoCamera:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # 相机模式切换成连续采集
        mvsdk.CameraSetTriggerMode(self.hCamera, 0)

        # 手动曝光，曝光时间30ms
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraSetExposureTime(self.hCamera, exposure_time * 1000)

        # 让SDK内部取图线程开始工作
        mvsdk.CameraPlay(self.hCamera)

        # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
        self.FrameBufferSize = self.cap.sResolutionRange.iWidthMax * self.cap.sResolutionRange.iHeightMax * (1 if self.monoCamera else 3)

        # 分配RGB buffer，用来存放ISP输出的图像
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(self.FrameBufferSize, 16)

    def capture_frame(self):
        # 从相机取一帧图片
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
            mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)

            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            return frame

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print(f"CameraGetImageBuffer failed({e.error_code}): {e.message}")
            return None

    def __del__(self):
        # 关闭相机
        mvsdk.CameraUnInit(self.hCamera)
        # 释放帧缓存
        mvsdk.CameraAlignFree(self.pFrameBuffer)
        cv2.destroyAllWindows()


class orbicCam:
    """
    奥比中光相机类
    """
    def __init__(self):
        self.LEFT_CAM = None
        self.RIGHT_CAM = None

        # 初始化OpenNI2
        openni2.initialize()
        self.device_list = openni2.Device.enumerate_uris()

    def get_device_names(self):
        print("Connected devices:")
        for i, device_info in enumerate(self.device_list):
            print(f"Device {i}: {device_info}")
            # 示例赋值：假设设备信息可以通过某种规则区分左右相机
            if b'2bc5/060f' in device_info:  # 根据设备名的特定标识符判断是左相机
                self.LEFT_CAM = device_info
            elif b'2bc5/0403' in device_info:  # 根据设备名的特定标识符判断是右相机
                self.RIGHT_CAM = device_info

        # 成功获取设备名
        if self.LEFT_CAM is not None and self.RIGHT_CAM is not None:
            print(f"LEFT_CAM = {self.LEFT_CAM}")
            print(f"RIGHT_CAM = {self.RIGHT_CAM}")
            # openni2.unload()
            return self.LEFT_CAM, self.RIGHT_CAM
        else:
            # openni2.unload()
            raise Exception("No camera was found!")
        

class videoWritter:
    """
    视频写入类
    """

    def __init__(self):
        pass

    # 查找文件夹中下一个可用的视频序号
    def find_next_video_index(self,folder):
        index = 1
        while True:
            filename = os.path.join(folder, f"{index}.mp4")
            if not os.path.exists(filename):
                return index
            index += 1

    # 初始化视频写入器
    def initialize_video_writers(self,o_folder, m_folder, frame_width=640, frame_height=480, fps=30.0, codec='mp4v'):
        # 创建文件夹
        os.makedirs(o_folder, exist_ok=True)
        os.makedirs(m_folder, exist_ok=True)

        # 查找下一个可用的文件名
        o_next_index = self.find_next_video_index(o_folder)
        m_next_index = self.find_next_video_index(m_folder)

        o_video_filename = os.path.join(o_folder, f"{o_next_index}.mp4")
        m_video_filename = os.path.join(m_folder, f"{m_next_index}.mp4")

        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*codec)
        o_video_writer = cv2.VideoWriter(o_video_filename, fourcc, fps, (frame_width * 2, frame_height))
        m_video_writer = cv2.VideoWriter(m_video_filename, fourcc, fps, (frame_width * 2, frame_height))

        return o_video_writer, m_video_writer, o_video_filename, m_video_filename


# 函数定义

def time_monitor(func):
    """
    函数运行时间装饰器，获取各函数运行时间，调试监控用
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper


def read_serial_data(port, shared_data):
    """
    串口读取，与主程序异线程，防止缓冲区过载
    """
    while True:
        flag, yaw, endCode = port.portReading()
        shared_data.flag = flag
        shared_data.yaw = yaw
        shared_data.endCode = endCode
        # time.sleep(0.1)  # 降低读取频率，防止过快读取



def framecombine(left_frame,right_frame):

    # 将两个帧水平拼接在一起
    combined_frame = cv2.hconcat([left_frame, right_frame])

    return combined_frame


# @time_monitor
def depthFinder(cx, cy, width, height,depthArray): 
    """
    若识别中心点无深度信息，用此函数进行补充
    """
    depthData = 0  # 存储深度信息
    dx = width // 20  # x方向的遍历步长
    dy = height // 20  # y方向的遍历步长
    flag = 0
    

    if depthArray[cy,cx] != 0: # 深度数据列优先
        return depthArray[cy,cx]

    else:
        for y in range(cy, cy + height // 2, dy):  # 遍历中点的右下方，此区域可以调整，看具体情况
            for x in range(cx, cx + width // 2, dx):
                
                if x >= 640:
                    flag == 0
                    break
                if y >= 480:
                    flag == 0
                    break

                if (depthArray[y, x] / 1000.0) != 0:
                    depthData = depthArray[y, x] / 1000.0
                    flag = 1
                    break
            if flag == 1:
                break
        if flag == 0:
            return -1
        else:
            print('!!!!!!!!!!!!!!!!成功补充深度值!!!!!!!!!!!!!!!!!!!')
            return depthData

# @time_monitor
def recognition_analysis(container,left_dist_array,right_dist_array,showInfo=True):
    """
    处理识别结果，将识别结果按照颜色分类,并且获取深度信息。这里没有对球框进行操作，注意修改
    [x1,y1,x2,y2,color,center_x,center_y,depth,world_x,world_y,world_z,flag]
    """

    if container is not None and len(container):

        blue = container[container[:,4] == 0]
        red = container[container[:,4] == 1]
        purple = container[container[:,4] == 2]
        basket = container[container[:,4] == 3]

        
        new_blue = np.hstack((blue,np.full((blue.shape[0],7),-1)))
        new_red = np.hstack((red,np.full((red.shape[0],7),-1)))
        new_purple = np.hstack((purple,np.full((purple.shape[0],7),-1)))


        # 深度信息

        for i in range(len(new_blue)):
            # 计算画面中心
            xc = (new_blue[i][0] + new_blue[i][2]) // 2
            yc = (new_blue[i][1] + new_blue[i][3]) // 2

            # 将中心点坐标添加到数组中
            new_blue[i][5] = xc
            new_blue[i][6] = yc

            # 添加深度信息
            # 如果中心点在左画面中
            if xc < 640:
                new_blue[i][7] = depthFinder(xc,yc,new_blue[i][2]-new_blue[i][0],new_blue[i][3]-new_blue[i][1],left_dist_array)
                new_blue[i][11] = 0  # 左侧标志位
                
            # 如果中心点在右画面中
            else:
                new_blue[i][7] = depthFinder(xc - 640,yc,new_blue[i][2]-new_blue[i][0],new_blue[i][3]-new_blue[i][1],right_dist_array)
                new_blue[i][11] = 1  # 左侧标志位

            # 如果成功获取深度，则进行坐标系转换
            if new_blue[i][7] > 0:
                    new_blue[i][8],new_blue[i][9],new_blue[i][10] = world_coordinate(new_blue[i])

        for i in range(len(new_red)):
            # 计算画面中心
            xc = (new_red[i][0] + new_red[i][2]) // 2
            yc = (new_red[i][1] + new_red[i][3]) // 2

            # 将中心点坐标添加到数组中
            new_red[i][5] = xc
            new_red[i][6] = yc

            # 添加深度信息
            # 如果中心点在左画面中
            if xc < 640:
                new_red[i][7] = depthFinder(xc,yc,new_red[i][2]-new_red[i][0],new_red[i][3]-new_red[i][1],left_dist_array)
                new_red[i][11] = 0
            
            # 如果中心点在右画面中
            else:
                new_red[i][7] = depthFinder(xc - 640,yc,new_red[i][2]-new_red[i][0],new_red[i][3]-new_red[i][1],right_dist_array)
                new_red[i][11] = 1

            # 如果成功获取深度，则进行坐标系转换
            if new_red[i][7] > 0:
                    new_red[i][8],new_red[i][9],new_red[i][10] = world_coordinate(new_red[i])   

        for i in range(len(new_purple)):        
            # 计算画面中心
            xc = (new_purple[i][0] + new_purple[i][2]) // 2
            yc = (new_purple[i][1] + new_purple[i][3]) // 2

            # 将中心点坐标添加到数组中
            new_purple[i][5] = xc
            new_purple[i][6] = yc

            # 添加深度信息
            # 如果中心点在左画面中
            if xc < 640:
                new_purple[i][7] = depthFinder(xc,yc,new_purple[i][2]-new_purple[i][0],new_purple[i][3]-new_purple[i][1],left_dist_array)
                new_purple[i][11] = 0
            
            # 如果中心点在右画面中
            else:
                new_purple[i][7] = depthFinder(xc - 640,yc,new_purple[i][2]-new_purple[i][0],new_purple[i][3]-new_purple[i][1],right_dist_array)
                new_purple[i][11] = 1

            # 如果成功获取深度，则进行坐标系转换
            if new_purple[i][7] > 0:
                    new_purple[i][8],new_purple[i][9],new_purple[i][10] = world_coordinate(new_purple[i])

        # 筛除无深度信息的识别框，这里可以修改深度范围
        new_blue = new_blue[new_blue[:,7] > 10]
        new_red = new_red[new_red[:,7] > 10 ]
        new_purple = new_purple[new_purple[:,7] > 10]

        if showInfo:
            pass
            # print('blue:',new_blue)
            # print('red:',new_red)
            # print('purple:',new_purple)
            # print('basket:',basket)


        return True,new_blue,new_red,new_purple,basket

    else:
        return False,None,None,None,None
    

# 相机内参
#LEFT_CAMERA_MARTIX = [[511.7172970323651, 0.0, 323.6153693308604], 
#                      [0.0, 511.91731705307615, 242.16782430395918], 
#                      [0.0, 0.0, 1.0]]

LEFT_CAMERA_MARTIX =[ [580.4919299941866, 0.0, 332.0647904755885],
                      [0.0, 579.8510936617516, 249.47655284829642], 
                      [0.0, 0.0, 1.0]]



RIGHT_CAMERA_MATRIX = [[593.1817683329759, 0.0, 350.073179717656], 
                       [0.0, 592.0340497331903, 247.2002113235381], 
                       [0.0, 0.0, 1.0]]

# 相机外参

# RIGHT_RVECS = [ [ 8.75432973e-01, -4.22689646e-01, -2.34415455e-01],
#                [-4.08308104e-01, -9.06273023e-01,  1.09333019e-01],
#                [-2.58855470e-01, -1.13202832e-05, -9.65916087e-01 ]]

# RIGHT_TVECS = [[401.36],
#               [-181.49],
#               [538.81]]


# LEFT_RVECS = [[ 8.75432973e-01,  4.22689646e-01, -2.34415455e-01],
#               [ 4.08308104e-01, -9.06273023e-01, -1.09333019e-01],
#               [-2.58855470e-01,  1.13202832e-05, -9.65916087e-01 ]]


# LEFT_TVECS = [[401.36],
#               [181.49],
#               [538.81]]

RIGHT_RVECS = [[ 0.88233171 ,-0.40668918 ,-0.23687166],
               [-0.39274256 ,-0.91358445,  0.10543606],
               [-0.25596905 ,-0.00133954 ,-0.96669106]]

RIGHT_TVECS = [[412.03],
              [-155.97],
              [535.94]]

LEFT_RVECS = [[ 8.82437955e-01 , 4.06686774e-01 ,-2.36454369e-01],
              [ 3.93639194e-01 ,-9.13194087e-01 ,-1.05477906e-01],
              [ 2.59572660e-01 , 3.21426454e-04 , 9.65723891e-01]]


LEFT_TVECS = [[401.43],
              [180.64],
              [538.79]]


def world_coordinate(recognition_array):
    """
    将像素坐标转换到光心世界坐标，计算出平移距离与角度,目前考虑为左右平移加前后平移，后续也许可以改为斜线或者更好的方案
    注意，相机坐标系的z轴正方向为光轴指向前方，x轴正方向为画面右侧，y轴正方向为下侧
    机器人坐标系的x轴正方向为机器人前进方向，y轴正方向为机器人左侧，z轴正方向为机器人上方
    """
    # 左相机
    if recognition_array[11] == 0:

        #相机坐标系中的三维坐标
        x = (recognition_array[5] - LEFT_CAMERA_MARTIX[0][2]) * recognition_array[7] / LEFT_CAMERA_MARTIX[0][0]
        y = (recognition_array[6] - LEFT_CAMERA_MARTIX[1][2]) * recognition_array[7] / LEFT_CAMERA_MARTIX[1][1]
        z = recognition_array[7]

        Pcam = [[z], # 对应机器人坐标系的x
                [x], # 对应机器人坐标系的y
                [y]] # 对应机器人坐标系的z
        
        # 相机坐标系到机器人坐标系的转换
        Probot = np.dot(LEFT_RVECS, Pcam) + LEFT_TVECS

        # print('11111111111111Pcam:',Pcam)
        # print('11111111111111Probot:',Probot)
        Probot[1]=Probot[1]
        return Probot[0],Probot[1],Probot[2]

    # 右相机  注意  所有右相机操作不要忘了减去640
    if recognition_array[11] == 1:

        #相机坐标系中的三维坐标
        x = ((recognition_array[5] -640) - RIGHT_CAMERA_MATRIX[0][2]) * recognition_array[7] / RIGHT_CAMERA_MATRIX[0][0]
        y = (recognition_array[6] - RIGHT_CAMERA_MATRIX[1][2]) * recognition_array[7] / RIGHT_CAMERA_MATRIX[1][1]
        z = recognition_array[7]

        Pcam = [[z], # 对应机器人坐标系的x
                [x], # 对应机器人坐标系的y
                [y]] # 对应机器人坐标系的z
        
        # 相机坐标系到机器人坐标系的转换
        Probot = np.dot(RIGHT_RVECS, Pcam) + RIGHT_TVECS

        # print('22222222222222Pcam:',Pcam)
        # print('22222222222222Probot:',Probot)

        return Probot[0],Probot[1],Probot[2]


def average_print(a,b):
    """
    打印平均值
    """
    print(f"水平移动={a}mm, 前进b={b}mm")
    return a, b


# # 谷仓参数，左边界、右边界、一层、二层、三层
# BARN = [[83, 187, 371, 315, 252],    # 1号谷仓
#         [356, 480, 381, 316, 251],   # 2号谷仓
#         [679, 772, 333, 272, 227],   # 3号谷仓
#         [920, 1012, 335, 291, 231],   # 4号谷仓
#         [1141, 1225, 327, 279, 232]] # 5号谷仓
#一区蓝
# 谷仓参数，左边界、右边界、一层、二层、三层
BARN = [[93, 186, 361, 293, 225], # 1号谷仓
        [351, 455, 373, 301, 229],  # 2号谷仓
        [676, 768, 329, 271, 213],  # 3号谷仓
        [906, 987, 323, 268, 213],  # 4号谷仓
        [1110, 1184, 322, 270, 217]] # 5号谷仓
##line_frame.py手动划分
# 修正参数
PARA =  15

def boardAnalysis(container,board,side):
    """
    分析棋盘布局
    """
    # 初始化所有变量
    barn1 = []
    barn2 = []
    barn3 = []
    barn4 = []
    barn5 = []

    # 处理传入数组
    ori_blue = container[container[:,4] == 0]
    ori_red = container[container[:,4] == 1]
    basket = container[container[:,4] == 3]

    blue = np.hstack((ori_blue,np.full((ori_blue.shape[0],2),-1)))
    red = np.hstack((ori_red,np.full((ori_red.shape[0],2),-1)))

    # 计算中心点
    if len(blue) > 0:
        blue[:,5] = (blue[:,0] + blue[:,2]) // 2
        blue[:,6] = (blue[:,1] + blue[:,3]) // 2
    if len(red) > 0:
        red[:,5] = (red[:,0] + red[:,2]) // 2
        red[:,6] = (red[:,1] + red[:,3]) // 2

    # 按照x坐标排序
    blue = blue[np.argsort(blue[:,5])]
    print('blue:',blue)
    red = red[np.argsort(red[:,5])]
    print('red:',red)

    array = np.vstack((blue,red))
 

    # 谷仓分析
    # 1号谷仓
    barn1 = barnAnalysis(array,BARN[0])
    if barn1[0] != 5:
        board[2][0] = barn1[0]
        if barn1[1] != 5:
            board[1][0] = barn1[1]
            if barn1[2] != 5:
                board[0][0] = barn1[2]

    # 2号谷仓
    barn2 = barnAnalysis(array,BARN[1])
    if barn2[0] != 5:
        board[2][1] = barn2[0]
        if barn2[1] != 5:
            board[1][1] = barn2[1]
            if barn2[2] != 5:
                board[0][1] = barn2[2]
    
    # 3号谷仓
    barn3 = barnAnalysis(array,BARN[2])
    if barn3[0] != 5:
        board[2][2] = barn3[0]
        if barn3[1] != 5:
            board[1][2] = barn3[1]
            if barn3[2] != 5:
                board[0][2] = barn3[2]

    # 4号谷仓
    barn4 = barnAnalysis(array,BARN[3])
    if barn4[0] != 5:
        board[2][3] = barn4[0]
        if barn4[1] != 5:
            board[1][3] = barn4[1]
            if barn4[2] != 5:
                board[0][3] = barn4[2]
    
    # 5号谷仓
    barn5 = barnAnalysis(array,BARN[4])
    print('barn5:',barn5)
    if barn5[0] != 5:
        board[2][4] = barn5[0]
        if barn5[1] != 5:
            board[1][4] = barn5[1]
            if barn5[2] != 5:
                board[0][4] = barn5[2]

    print('ori_board:',board,type(board))

    # 蓝方
    if side == 0:
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == 5:
                    board[i][j] = 0
                else:
                    if board[i][j] == 0:
                        board[i][j] = 1 
                    else:
                        board[i][j] = -1
    # 红方
    if side == 1:
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == 5:
                    board[i][j] = 0
                else:
                    if board[i][j] == 0:
                        board[i][j] = -1 
                    else:
                        board[i][j] = 1


    print('trans_board:',board)
    
    return board

def barnAnalysis(ball_array,barn_index):
    """
    具体分析每个谷仓
    """

    # print("bar_index:",barn_index[2])

    barn = [5,5,5]

    # 筛选出该谷仓范围的球
    condition = (ball_array[:, 5] >= (barn_index[0] - PARA)) & (ball_array[:, 5] <= (barn_index[1] + PARA))
    # print('condition:',(ball_array[:, 5] >= (barn_index[0] + PARA)),ball_array[:, 5],barn_index[0] + PARA)
    barn_balls = ball_array[condition]
    # print('barn_balls:',barn_balls)


    # 第一层
    filtered_rows = barn_balls[(barn_balls[:, 6] >= (barn_index[2] - PARA)) & (barn_balls[:, 6] <= (barn_index[2]) + PARA)]
    # print('filtered_rows:',filtered_rows)
    # print('filtered_rows:',(barn_balls[:, 6] >= (barn_index[2] - PARA)) , (barn_balls[:, 6] <= (barn_index[2]) + PARA))
    if len(filtered_rows) > 0:   
        barn[0] = filtered_rows[0][4] # 填写颜色信息

        # 第一层成功填充才进行第二层
        # 第二层
        filtered_rows = []
        filtered_rows = barn_balls[(barn_balls[:, 6] >= (barn_index[3] - PARA)) & (barn_balls[:, 6] <= (barn_index[3]) + PARA)]
        if len(filtered_rows) > 0:
            barn[1] = filtered_rows[0][4]

            # 第二层成功填充才进行第三层
            # 第三层
            filtered_rows = []
            filtered_rows = barn_balls[(barn_balls[:, 6] >= (barn_index[4] - PARA)) & (barn_balls[:, 6] <= (barn_index[4]) + PARA)]
            if len(filtered_rows) > 0:
                barn[2] = filtered_rows[0][4]

    return barn

