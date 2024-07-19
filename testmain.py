from Utils.cam_for_depth.Astra import depthCam
from Utils.serialTools.usbToTTL import TTL
from Utils.yolo_openvino.vinodetect import vinodetect
from Utils.policy.main import load_policy, run_policy, State

import cv2
import numpy as np
import time
import threading


#####################################类定义######################################

# 串口数据共享类，以对全局共享串口信息
class SharedData:
    def __init__(self):
        self.flag = 0
        self.yaw = 0
        self.endCode = 0
        self.redArray = []
        self.blueArray = []
        self.dist_vis = None
        self.distArray = None

# 创建线程读取类，以出去buffer里无用且堆积的frame
class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()



###################################################函数定义#######################

# 函数运行时间装饰器，便于获取每个函数的运行时间
def time_monitor(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper


# 串口读取函数,与主程序并行，防止缓冲区过载
def read_serial_data(port, shared_data):
    while True:
        flag, yaw, endCode = port.portReading()
        shared_data.flag = flag
        shared_data.yaw = yaw
        shared_data.endCode = endCode
        # time.sleep(0.1)  # 降低读取频率，防止过快读取


#  数组拼接函数，用于处理yolo识别结果
# @njit
@time_monitor
def array_joint(detect, side, astra, distArray, yaw,option):

    global ball_num

    if option == 'no_joint': # 寻球模式使用，返回带距离和yaw的球信息

        if side == 0:
            if len(detect.red):

                # 整合原始识别信息为n维4列的数组
                array = np.round(np.vstack(detect.red))

                # 计算x中心
                result1 = array[:, 0] + array[:, 2] / 2
                # 计算y中心
                result2 = array[:, 1] + array[:, 3] / 2

                # 将中心信息加入n维数组，得到一个n维6列的数组
                array = np.column_stack((array, result1, result2))
                array = array.astype(int)

                # 遍历数组
                for row in array:

                    # 当目标球略过画面中心时，获取深度值
                    if 319 <= row[4] <= 321:  # 该参数可调，画面中心

                        # 获取深度信息
                        distance = astra.depthFinder(row[4], row[5], row[2], row[3], distArray)
                        
                        # 此时数组为空
                        if ball_num == 0:
 
                            # 创建新数组
                            arrWithDistYaw = np.zeros(len(row) + 2, dtype=row.dtype)
                            # 将原始数组的内容复制到新数组中
                            arrWithDistYaw[:len(row)] = row
                            # 在新数组的末尾添加距离
                            arrWithDistYaw[-2] = distance
                            # print(distance)
                            # 在新数组的末尾添加此时的yaw值
                            arrWithDistYaw[-1] = yaw

                            redArray.append(arrWithDistYaw)

                            ball_num += 1 # 表面数组已存入一个球的信息

                        if ball_num >0 and ((abs(distance - redArray[ball_num-1][6])>10) or (yaw - redArray[ball_num-1][7]>10)):  # 避免同一位置的球反复加入序列中

                             # 创建新数组
                            arrWithDistYaw = np.zeros(len(row) + 2, dtype=row.dtype)
                            # 将原始数组的内容复制到新数组中
                            arrWithDistYaw[:len(row)] = row
                            # 在新数组的末尾添加距离
                            arrWithDistYaw[-2] = distance
                            # print(distance)
                            # 在新数组的末尾添加此时的yaw值
                            arrWithDistYaw[-1] = yaw

                            redArray.append(arrWithDistYaw)

                            ball_num += 1 # 表面数组已存入一个球的信息

            print(f'寻球数组-红色：{redArray}')

                        # print('寻球数组-红色')  # 调试信息#####################################
                        # print(redArray)
                        # 获得n维8列数组

         

    if option == 'joint':  # 决策模式使用，返回球的距离和颜色信息

        global_red = [] # 存放本次识别的红球信息
        global_blue = [] # 存放本次识别的篮球信息

        if len(detect.red):

            # 整合原始识别信息为n维4列的数组
            array = np.round(np.vstack(detect.red))

            # 计算x中心
            result1 = array[:, 0] + array[:, 2] / 2
            # 计算y中心
            result2 = array[:, 1] + array[:, 3] / 2

            # 将中心信息加入n维数组，得到一个n维6列的数组
            array = np.column_stack((array, result1, result2))
            array = array.astype(int)

            # 遍历数组
            for row in array:

                # 当目标球略过画面中心时，获取深度值
                if 280 <= row[4] <= 340:  # 该参数可调，画面中心

                    # 获取深度信息
                    distance = astra.depthFinder(row[4], row[5], row[2], row[3], distArray)

                    # 创建新数组
                    arrWithDistSide = np.zeros(len(row) + 2, dtype=row.dtype)
                    # 将原始数组的内容复制到新数组中
                    arrWithDistSide[:len(row)] = row
                    # 在新数组的末尾添加距离
                    arrWithDistSide[-2] = distance
                    # print(distance)
                    # 在新数组的末尾添加此时的side标签
                    arrWithDistSide[-1] = 0
                    # 获得n维8列数组
                    global_red.append(arrWithDistSide)
                    # print(global_red)

        if len(detect.blue):

            # 整合原始识别信息为n维4列的数组
            array = np.round(np.vstack(detect.blue))

            # 计算x中心
            result1 = array[:, 0] + array[:, 2] / 2
            # 计算y中心
            result2 = array[:, 1] + array[:, 3] / 2

            # 将中心信息加入n维数组，得到一个n维6列的数组
            array = np.column_stack((array, result1, result2))
            array = array.astype(int)

            # 遍历数组
            for row in array:

                # 当目标球略过画面中心时，获取深度值
                if 280 <= row[4] <= 340:  # 该参数可调，画面中心

                    # 获取深度信息
                    distance = astra.depthFinder(row[4], row[5], row[2], row[3], distArray)

                    # 创建新数组
                    arrWithDistSide = np.zeros(len(row) + 2, dtype=row.dtype)
                    # 将原始数组的内容复制到新数组中
                    arrWithDistSide[:len(row)] = row
                    # 在新数组的末尾添加距离
                    arrWithDistSide[-2] = distance
                    # print(distance)
                    # 在新数组的末尾添加此时的side标签
                    arrWithDistSide[-1] = 1
                    # 获得n维8列数组
                    global_blue.append(arrWithDistSide)
                    # print(global_blue)

        if len(global_red) and len(global_blue):
            return np.vstack([global_red, global_blue]) # 如果识别到了两种球，就拼接他们一并输出
        elif len(global_red):
            return np.vstack(global_red) # 只识别到了红球
        elif len(global_blue):
            return np.vstack(global_blue) # 只识别到了篮球

@time_monitor
def ball_locate(global_array, barn_index,side): # 棋牌排列功能

    global barn # 谷仓信息
    global ball_roi_para # 球roi阈值
    global board # 棋盘

    filtered_rows = [] # 存放过滤后的球信息

    # 列表推导式生成一个布尔索引数组，选择距离大于2000的球，排除掉无用信息
    selected_rows = [row for row in global_array if 1800 <= row[6] <= 3000]
    # print('1111',selected_rows)

    if len(selected_rows):

        # 将布尔索引数组转换为NumPy数组
        valid_array = np.array(selected_rows)

        # 根据y坐标重新排列
        sorted_indices = np.argsort(valid_array[:, 5])
        valid_array = valid_array[sorted_indices]

        # print('1111111111111111',valid_array)
        print('\n有效谷仓状态')
        for row in valid_array:
            print(row)

        # 在有效数组中提取各层的球信息

        # layer1
        filtered_rows = valid_array[(valid_array[:, 5] >= (barn[barn_index][2] - ball_roi_para))
                                    & (valid_array[:, 5] <= (barn[barn_index][2] + ball_roi_para))] # 满足第一层可信范围的球信息
        # 如果有多个行满足以上条件，则保留第五列元素最接近320的行
        if len(filtered_rows) > 1:
            board[2][barn_index] = 1 if filtered_rows[np.argmin(np.abs(filtered_rows[:, 4] - 320))][7]  == side else -1
            print(f'第一层已存放{board[2][barn_index]}')
        elif len(filtered_rows):
            board[2][barn_index] = 1 if filtered_rows[0][7] == side else -1
            print(f'第一层已存放{board[2][barn_index]}')

        # layer2
        if len(filtered_rows):
            filtered_rows = []
            filtered_rows = valid_array[(valid_array[:, 5] >= (barn[barn_index][3] - ball_roi_para))
                                        & (valid_array[:, 5] <= (barn[barn_index][3] + ball_roi_para))] # 满足第二层可信范围的球信息
            if len(filtered_rows) > 1:
                board[1][barn_index] = 1 if filtered_rows[np.argmin(np.abs(filtered_rows[:, 4] - 320))][7]  == side else -1
                print(f'第二层已存放{board[1][barn_index]}')
            elif len(filtered_rows):
                board[1][barn_index] = 1 if filtered_rows[0][7] == side else -1
                print(f'第二层已存放{board[1][barn_index]}')

        # layer3
        if len(filtered_rows):
            filtered_rows = []
            filtered_rows = valid_array[(valid_array[:, 5] >= (barn[barn_index][4] - ball_roi_para))
                                        & (valid_array[:, 5] <= (barn[barn_index][4] + ball_roi_para))] # 满足第三层可信范围的球信息
            if len(filtered_rows) > 1:
                board[0][barn_index] = 1 if filtered_rows[np.argmin(np.abs(filtered_rows[:, 4] - 320))][7]  == side else -1
                print(f'第三层已存放{board[0][barn_index]}')
            elif len(filtered_rows):
                board[0][barn_index] = 1 if filtered_rows[0][7] == side else -1
                print(f'第三层已存放{board[0][barn_index]}')


############################################全局变量和默认参数的配置

flag = 0   # 控制标志位，后续和电控联调
side = 0   # 选边，赛前更改  0 红  1蓝
status = 0  # 用于判断是否完成一轮棋盘识别
ball_num = 0 # 记录满足画面中心的球数量，方便剔除多余的球

# red 0
# blue 1
# purple 2
# basket 3

# vino初始化选项
MODEL = "/home/spr/RC2024/Utils/yolo_openvino/v5n_openvino_model/v5n.xml" # 模型地址
DEVICE = "CPU" # 设备

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_WB,0)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,-1) # 禁止使用，会导致曝光出错，自动曝光参数为3
cam_cleaner = CameraBufferCleanerThread(cap) # 开启cam线程

astra = depthCam(b'2bc5/0403@3/8')
detect = vinodetect(model_path=MODEL, device_name=DEVICE)
port = TTL()
shared_data = SharedData()


# 载入策略模型，约5-10秒
load_policy()
print('策略模型载入成功！')

# 存放球、框的像素、距离、yaw信息
redArray = []
blueArray = []
purpleArray = []
basketArray = []
array = []
DC_array = []

# 画面信息
center_x = 240
center_y = 320
roi_width = 100
roi_height = 150
roi_x1 = 190
roi_x2 = 290
roi_y1 = 250
roi_y2 = 400

# 初始化棋盘
board = [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]


# 谷仓大致信息，需要调参 ###################################

# dist,yaw,1,2,3
barn = [[2558, 2144, 203, 154, 106],  # barn1
        [2240, 1992, 217, 158, 100],  # barn2
        [2102, 1800, 219, 157, 100],  # barn3
        [2300, 1627, 216, 160, 100],  # barn4
        [2700, 1482, 204, 164, 110]]  # barn5

# 修正量#############################
barn_dist_para = 20
barn_yaw_para = 30
ball_roi_para = 25



# 调试用
# redArray = [[ 301,  348,   38,   34,  320,  365, 4641],
#             [ 301,  347,   38,   35,  320,  364, 4704],
#             [ 302,  332,   34,   31,  319,  347, 5052],
#             [ 303,  332,   34,   32,  320,  348, 5052],
#             [ 300,  353,   43,   51,  321,  378, 4518],
#             [ 299,  353,   44,   51,  321,  378, 4704]]


# 创建串口读取线程
serial_thread = threading.Thread(target=read_serial_data, args=(port, shared_data))
serial_thread.daemon = True
serial_thread.start()



###############################主程序###########################################
def mainfunc():
    while True:


        if cam_cleaner.last_frame is not None:
            frame = cam_cleaner.last_frame
        else:
            print('CAM ERROR!')
            break

        # # 读取画面
        # _, frame = cap.read()
        # _, frame = cap.read()

        monitor_frame = frame.copy()


        # 读取串口
        flag = shared_data.flag
        yaw = shared_data.yaw
        endCode = shared_data.endCode
        # print('串口数据',flag,yaw)

        flag = 'SC'  # 调试用#########################
        # yaw = 2900 # debug###################

        if endCode == 255:

            start = time.time()

            print(f'当前功能：{flag},当前角度：{yaw}')

            # 等待模式
            if flag == 'WT' and endCode == 255:

                # 关闭所有画面
                cv2.destroyAllWindows()

                # 清空所有数据变量
                redArray = []
                blueArray = []
                purpleArray = []
                basketArray = []
                array = []
                ball_num = 0

                # 继续循环
                continue

            # 寻球模式
            if flag == 'SC' and endCode == 255:


                # 推理
                detect.main(frame)

                # 获取深度信息与可视化
                dist_vis, distArray = astra.depthCapture()

                array_joint(detect, side, astra, distArray, yaw,option='no_joint')


                cv2.imshow("dist", dist_vis)
                cv2.waitKey(1)

            # 取球模式
            if flag == 'GT' and endCode == 255:

                # 关闭所有画面
                cv2.destroyAllWindows()

                ball_num = 0

                # 红方
                if side == 0:

                    if len(redArray):
                        # 将n维7列数组重新拼接整理形状
                        redArray = np.vstack(redArray)

                        # 对传入数据根据距离重新排序
                        sorted_indices = np.argsort(redArray[:, 6])
                        redArray = redArray[sorted_indices]

                        print('整合数组-红色')  # 调试信息
                        for row in redArray:
                            print(row)

                        # 遍历数组
                        for row in redArray:
                            if row[6] > 50:  # 距离阈值，防止误识别，根据实际情况修改##############

                                # 取得最近值
                                print(f'最近红色球距离:{row[6]},已发送距离')
                                port.int_to_bytes(row[6])
                                break

                    else:  # 如果本次识别没有识别到任何一个球，那么返回报错信息，电控相应处理##############

                        print('未识别到红色球,已发送报错信号!')
                        port.int_to_bytes(0000)

                # 蓝方
                if side == 1:

                    blueArray = np.vstack(blueArray)

                    # 对传入数据根据距离重新排序
                    sorted_indices = np.argsort(blueArray[:, 6])
                    blueArray = blueArray[sorted_indices]
                    print(blueArray)

                    for row in blueArray:
                        if row[6] > 0:
                            print(row[6])
                            break

            # 决策模式
            if flag == 'DC' and endCode == 255:

                if yaw >= 2200 :
                    print('初始化棋盘！')
                    board = [[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]]
                if yaw <= 1400 :
                    print('策略查询中!')
                    barn_ai,value = run_policy(board)
                    print(f'AI选择左{barn_ai}号谷仓\n程序将待机10秒')
                    print('value of all barn\n')
                    for value, col in value:
                        print(f' {value:.2f}   {col+1}')
                    time.sleep(10)


                # 左一谷仓
                if (barn[0][1] - barn_yaw_para) <= yaw <= (barn[0][1] + barn_yaw_para):

                    detect.main(frame)
                    dist_vis, distArray = astra.depthCapture()
                    DC_array = array_joint(detect, None, astra, distArray, None,option='joint')
                    if DC_array is not None and len(DC_array):

                        print(f'左一仓初始状态：{DC_array}')

                        ball_locate(DC_array, barn_index=0,side=side)
                    else:
                        print("左一仓 no target!")

                # 左二谷仓
                if (barn[1][1] - barn_yaw_para) <= yaw <= (barn[1][1] + barn_yaw_para):

                    detect.main(frame)
                    dist_vis, distArray = astra.depthCapture()
                    DC_array = array_joint(detect, None, astra, distArray, None,option='joint')
                    if DC_array is not None and len(DC_array):

                        print(f'左二仓初始状态：{DC_array}')

                        ball_locate(DC_array, barn_index=1,side=side)
                    else:
                        print("左二仓 no target!")

                # 左三谷仓
                if (barn[2][1] - barn_yaw_para) <= yaw <= (barn[2][1] + barn_yaw_para):

                    detect.main(frame)
                    dist_vis, distArray = astra.depthCapture()
                    DC_array = array_joint(detect, None, astra, distArray, None,option='joint')
                    if DC_array is not None and len(DC_array):

                        print(f'左三仓初始状态：{DC_array}')

                        ball_locate(DC_array, barn_index=2,side=side)
                    else:
                        print("左三仓 no target!")

                # 左四谷仓
                if (barn[3][1] - barn_yaw_para) <= yaw <= (barn[3][1] + barn_yaw_para):

                    detect.main(frame)
                    dist_vis, distArray = astra.depthCapture()
                    DC_array = array_joint(detect, None, astra, distArray, None,option='joint')
                    if DC_array is not None and len(DC_array):

                        print(f'左四仓初始状态：{DC_array}')

                        ball_locate(DC_array, barn_index=3,side=side)
                    else:
                        print("左四仓 no target!")

                # 左五谷仓
                if (barn[4][1] - barn_yaw_para) <= yaw <= (barn[4][1] + barn_yaw_para):

                    detect.main(frame)
                    dist_vis, distArray = astra.depthCapture()
                    DC_array = array_joint(detect, None, astra, distArray, None,option='joint')
                    if DC_array is not None and len(DC_array):

                        print(f'左五仓初始状态：{DC_array}')

                        ball_locate(DC_array, barn_index=4,side=side)
                    else:
                        print("no target!")

                print(f'本轮棋盘布局：')
                for row in board:
                    print(row)

            end = time.time()

            monitor_frame = cv2.putText(monitor_frame , str(int(1/(end - start))),(50,80),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 0, 255),5)

            # 定义十字准星的长度
            crosshair_size = 20
            # 在中心画一个横线
            cv2.line(monitor_frame, (320 - crosshair_size, 240), (320 + crosshair_size, 240), (0, 255, 255), 5)
            # 在中心画一个竖线
            cv2.line(monitor_frame, (320, 240 - crosshair_size), (320, 240 + crosshair_size), (0, 255, 255), 5)
            cv2.imshow('monitor', monitor_frame)
            cv2.waitKey(1)
            print("##########################################################################################")

        else:
            time.sleep(0.001)


if __name__ == "__main__" :

    mainfunc()