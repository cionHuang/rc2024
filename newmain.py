# 导入所需的各程序类
from Utils.cam_for_depth.Astra import depthCam
from Utils.serialTools.usbToTTL import TTL
from Utils.serialTools.rs485 import RS485
from Utils.policy.main import load_policy, load_states, run_policy, State
from Utils.tools import (SharedData, mindvisionCamera, MultiDataFrameFilter, orbicCam, videoWritter,
                         time_monitor, read_serial_data, recognition_analysis, average_print, boardAnalysis)
from Utils.new_vino.new_vinodetect import vinodetect

# 一些必要的库
import cv2
import os
import numpy as np
import time
import threading
import traceback

# 定义全局变量
# 模型相关
MODEL = '/home/spr-rc/rc2024/Models/0709v5n/best_openvino_model'
DEVICE = 'GPU'
THRESHOLD = 0.65


# 相机相关
LEFT_CAM = None
RIGHT_CAM = None

# 选边
SIDE = 0 # 蓝色
#SIDE = 1  # 红色

# 处理标志位
analysis_status = False

# 是否显示详细信息
SHOWINFO = True

# port thread reading data
latest_data = None

def rs485_reader(port):
    global latest_data
    while  True:
        latest_data = port.receive()

# 主程序


def mainfunc():

    # 系统初始化
    # 声明全局变量
    global analysis_status
    global MODEL
    global DEVICE
    global THRESHOLD
    global SHOWINFO
    global SIDE
    global latest_data


    # 决策模型加载
    load_states('./Utils/policy/models/all_states.pickle')  # 大约需要10s
    load_policy('./Utils/policy/models/policy_best.bin')

    # 识别模型加载
    detect = vinodetect(MODEL, DEVICE, THRESHOLD, SHOWINFO)

    # 相机初始化
    # 获取左右相机编号
    orbic = orbicCam()
    LEFT_CAM, RIGHT_CAM = orbic.get_device_names()

    right_cam = cv2.VideoCapture(0)  # 左相机彩色流
    left_cam = cv2.VideoCapture(2)  # 右相机彩色流
    left_cam.set(cv2.CAP_PROP_AUTO_WB, 0)  # 关闭自动白平衡
    right_cam.set(cv2.CAP_PROP_AUTO_WB, 0)
    # left_cleaner = CameraBufferCleanerThread(left_cam) # 创建清理线程
    # right_cleaner = CameraBufferCleanerThread(right_cam)

    right_depth = depthCam(LEFT_CAM)  # 左相机深度流
    left_depth = depthCam(RIGHT_CAM)  # 右相机深度流

    # 迈德威视相机初始化
    back_cam_left = mindvisionCamera(0,5)
    back_cam_right = mindvisionCamera(1,1)

    # 录像工具初始化
    video_writer = videoWritter()
    # 初始化视频写入器
    o_video_writer, m_video_writer, o_video_filename, m_video_filename = video_writer.initialize_video_writers('ovideos', 'mvideos')

    # 串口初始化
    # port = TTL()
    port = RS485()

    serial_thread = threading.Thread(target=rs485_reader,args=(port,),daemon=True)
    serial_thread.start()

    # # 串口数据共享
    # shared_data = SharedData()

    # # 创建串口读取线程
    # serial_thread = threading.Thread(target=read_serial_data, args=(port, shared_data))
    # serial_thread.daemon = True
    # serial_thread.start()

    # 创建数据滤波器
    # 滤波5帧,阈值50mm
    data_filter = MultiDataFrameFilter(5, threshold_a=50, threshold_b=50)

    # 时间管理
    start_time = time.time()

    port.ST_messagae()
    print('Initialization is done!')
    # 循环开始
    while True:

        # if (left_cleaner.last_frame is not None) and True:
        #     left_frame = left_cleaner.last_frame
        #     # right_frame = right_cleaner.last_frame
        # else:
        #     raise Exception('No frame!')

        #串口读取

        flag = latest_data


        if flag is None:
            continue

       
        print(f'flag:{flag}')##

        # flag = shared_data.flag
        # yaw = shared_data.yaw
        # endCode = shared_data.endCode

        #flag = 'SC'  # 调试用k

        # print(f'flag:{flag},yaw:{yaw},endCode:{endCode}')

        # 录制前后相机画面
        # 获取前相机左右画面
        l, o_left_frame = left_cam.read()
        r, o_right_frame = right_cam.read()
        # 获取后相机左右画面
        m_left_frame = back_cam_left.capture_frame()
        m_right_frame = back_cam_right.capture_frame()
        # 录制
        if o_left_frame is not None and o_right_frame is not None and m_left_frame is not None and m_right_frame is not None:
            o_frame = cv2.hconcat([o_left_frame, o_right_frame])
            m_frame = cv2.hconcat([cv2.rotate(m_left_frame, cv2.ROTATE_180), cv2.rotate(m_right_frame, cv2.ROTATE_180)])
            o_video_writer.write(o_frame)
            print(f"Recording video o:{o_video_filename} ...")
            m_video_writer.write(m_frame)
            print(f"Recording video m:{m_video_filename} ...")

        # 当录像30s后，保存一次
        if (time.time() - start_time)>=30:
            # 释放视频写入器并销毁所有窗口
            o_video_writer.release()
            print(f"Video saved as o: {o_video_filename}")
            m_video_writer.release()
            print(f"Video saved as m: {m_video_filename}")
        
            start_time  = time.time()
            o_video_writer, m_video_writer, o_video_filename, m_video_filename = video_writer.initialize_video_writers('ovideos', 'mvideos')

        # 重新建立连接
        if flag == 'RE':
            port.ST_messagae()
            continue

        # 等待模式
        if flag == 'WT':
            # 关闭所有窗口
            cv2.destroyAllWindows()
            # 发送心跳
            port.WT_messagae()
            # 继续循环
            continue

        if flag == 'SC':

            cv2.imshow('o_frame', o_frame)

            # 推理
            container = detect.infer_frame(o_frame)

            if container is not None and len(container):

                # 获取深度信息
                left_dist_vis, left_dist_array = left_depth.depthCapture()
                right_dist_vis, right_dist_array = right_depth.depthCapture()
                cv2.imshow('depth', cv2.hconcat([left_dist_vis, right_dist_vis]))

                # 若果成功获取深度信息，则进行识别分析
                if left_dist_array is not None and right_dist_array is not None:

                    # 识别分析
                    analysis_status, blue, red, purple, basket = recognition_analysis(
                        container, left_dist_array, right_dist_array, SHOWINFO)

                    # 蓝方
                    if (SIDE == 0) and (len(blue) > 0):
                        # 按照前进平移绝对值之和最小排列
                        sorted_blue = blue[np.argsort(np.abs(blue[:, 8]) + np.abs(blue[:, 9]))]
                        # 滤波
                        data_filter.add_data_frame([sorted_blue[0][8], sorted_blue[0][9]])
                        result = data_filter.get_filtered_data()

                        # 滤波结果
                        if result is not None:
                            print(f'\n\n\n\n\nFiltered result:{result}\n\n\n\n\n')
                            # 串口发送
                            port.SC_messagae(result[0], result[1])

                    # 红方
                    if (SIDE == 1) and (len(red) > 0):
                        # 按照前进平移绝对值之和最小排列
                        sorted_red = red[np.argsort(np.abs(red[:, 8]) + np.abs(red[:, 9]))]
                        # 滤波
                        data_filter.add_data_frame([sorted_red[0][8], sorted_red[0][9]])
                        result = data_filter.get_filtered_data()

                        # 滤波结果
                        if result is not None:
                            print(f'\n\n\n\n\nFiltered result:{result}\n\n\n\n\n')
                            # 串口发送
                            port.SC_messagae(result[0], result[1])

                else:
                    port.SC_messagae(20000.0, 20000.0)
                    continue
            else:
                port.SC_messagae(20000.0, 20000.0)
                continue

        # 决策模式
        if flag == 'DC':

            # 初始化本帧棋盘信息
            board = [[5, 5, 5, 5, 5],
                     [5, 5, 5, 5, 5],
                     [5, 5, 5, 5, 5]]

            cv2.imshow('m_frame', m_frame)

            # 推理
            container = detect.infer_frame(m_frame)

            # # 该帧识别有效
            if container is not None and len(container):

                # 棋盘分析
                board = boardAnalysis(container, board, SIDE)

                # 决策分析
                barn_ai, value = run_policy(board)
                print(f'AI选择左{barn_ai}号谷仓\n')
                print('value of all barn\n')
                for value, col in value:
                    print(f' {value:.2f}   {col+1}')

                port.DC_messagae(barn_ai)

                # print('board:',board)
            else:
                port.DC_messagae(0)
                continue

        print('##########################################################')

        
# 程序入口
if __name__ == '__main__':
    mainfunc()
