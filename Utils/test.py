# # #coding=utf-8
# # import cv2
# # import numpy as np
# # import mvsdk

# # class CameraCapture:
# #     def __init__(self):
# #         # 枚举相机
# #         self.DevList = mvsdk.CameraEnumerateDevice()
# #         self.nDev = len(self.DevList)
# #         if self.nDev < 1:
# #             raise Exception("No camera was found!")

# #         for i, DevInfo in enumerate(self.DevList):
# #             print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
# #         self.i = 0 if self.nDev == 1 else int(input("Select camera: "))
# #         self.DevInfo = self.DevList[self.i]
# #         print(self.DevInfo)

# #         # 打开相机
# #         self.hCamera = 0
# #         try:
# #             self.hCamera = mvsdk.CameraInit(self.DevInfo, -1, -1)
# #         except mvsdk.CameraException as e:
# #             raise Exception("CameraInit Failed({}): {}".format(e.error_code, e.message))

# #         # 获取相机特性描述
# #         self.cap = mvsdk.CameraGetCapability(self.hCamera)

# #         # 判断是黑白相机还是彩色相机
# #         self.monoCamera = (self.cap.sIspCapacity.bMonoSensor != 0)

# #         # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
# #         if self.monoCamera:
# #             mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
# #         else:
# #             mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

# #         # 相机模式切换成连续采集
# #         mvsdk.CameraSetTriggerMode(self.hCamera, 0)

# #         # 手动曝光，曝光时间30ms
# #         mvsdk.CameraSetAeState(self.hCamera, 0)
# #         mvsdk.CameraSetExposureTime(self.hCamera, 30 * 1000)

# #         # 让SDK内部取图线程开始工作
# #         mvsdk.CameraPlay(self.hCamera)

# #         # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
# #         self.FrameBufferSize = self.cap.sResolutionRange.iWidthMax * self.cap.sResolutionRange.iHeightMax * (1 if self.monoCamera else 3)

# #         # 分配RGB buffer，用来存放ISP输出的图像
# #         self.pFrameBuffer = mvsdk.CameraAlignMalloc(self.FrameBufferSize, 16)

# #     def capture_frame(self):
# #         # 从相机取一帧图片
# #         try:
# #             pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
# #             mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
# #             mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)

# #             # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
# #             # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
# #             frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
# #             frame = np.frombuffer(frame_data, dtype=np.uint8)
# #             frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

# #             frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
# #             return frame

# #         except mvsdk.CameraException as e:
# #             if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
# #                 print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))
# #             return None

# #     def __del__(self):
# #         # 关闭相机
# #         mvsdk.CameraUnInit(self.hCamera)
# #         # 释放帧缓存
# #         mvsdk.CameraAlignFree(self.pFrameBuffer)
# #         cv2.destroyAllWindows()

# # def main():
# #     try:
# #         camera_capture = CameraCapture()
# #         while True:
# #             frame = camera_capture.capture_frame()
# #             if frame is not None:
# #                 cv2.imshow("Press q to end", frame)
# #             if (cv2.waitKey(1) & 0xFF) == ord('q'):
# #                 break
# #     except Exception as e:
# #         print(e)
# #     finally:
# #         cv2.destroyAllWindows()

# # if __name__ == "__main__":
# #     main()

# from openni import openni2

# class CameraInitializer:
#     def __init__(self):
#         self.LEFT_CAM = None
#         self.RIGHT_CAM = None

#         # 初始化OpenNI2
#         openni2.initialize()
#         self.device_list = openni2.Device.enumerate_uris()

#     def get_device_names(self):
#         print("Connected devices:")
#         for i, device_info in enumerate(self.device_list):
#             print(f"Device {i}: {device_info}")
#             # 示例赋值：假设设备信息可以通过某种规则区分左右相机
#             if b'2bc5/060f' in device_info:  # 根据设备名的特定标识符判断是左相机
#                 self.LEFT_CAM = device_info
#             elif b'2bc5/0403' in device_info:  # 根据设备名的特定标识符判断是右相机
#                 self.RIGHT_CAM = device_info

#         if self.LEFT_CAM is None or self.RIGHT_CAM is None:
#             openni2.unload()
#             raise Exception("No left or right camera found!")
#         else:
#             print(f"LEFT_CAM = {self.LEFT_CAM}")
#             print(f"RIGHT_CAM = {self.RIGHT_CAM}")
#             openni2.unload()
#             return self.LEFT_CAM, self.RIGHT_CAM


# def main():
#     # 创建 CameraInitializer 实例
#     camera_initializer = CameraInitializer()
    
#     # 获取相机设备名
#     camera_initializer.get_device_names()

#     # 在主程序中获取相机信息
#     left_cam = camera_initializer.LEFT_CAM
#     right_cam = camera_initializer.RIGHT_CAM


# if __name__ == "__main__":
#     main()



# #coding=utf-8
# import cv2
# import numpy as np
# import mvsdk
# import platform
# import threading
# from queue import Queue

# def camera_loop(camera_index, frame_queue):
#     # 枚举相机
#     DevList = mvsdk.CameraEnumerateDevice()
#     nDev = len(DevList)
#     if nDev < 1:
#         print("No camera was found!")
#         return

#     DevInfo = DevList[camera_index]
#     print(DevInfo)

#     # 打开相机
#     hCamera = 0
#     try:
#         hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
#     except mvsdk.CameraException as e:
#         print("CameraInit Failed({}): {}".format(e.error_code, e.message))
#         return

#     # 获取相机特性描述
#     cap = mvsdk.CameraGetCapability(hCamera)

#     # 判断是黑白相机还是彩色相机
#     monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

#     # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
#     if monoCamera:
#         mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
#     else:
#         mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

#     # 相机模式切换成连续采集
#     mvsdk.CameraSetTriggerMode(hCamera, 0)

#     # 手动曝光，曝光时间30ms
#     mvsdk.CameraSetAeState(hCamera, 0)
#     mvsdk.CameraSetExposureTime(hCamera, 30 * 1000)

#     # 让SDK内部取图线程开始工作
#     mvsdk.CameraPlay(hCamera)

#     # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
#     FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

#     # 分配RGB buffer，用来存放ISP输出的图像
#     pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

#     while True:
#         # 从相机取一帧图片
#         try:
#             pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
#             mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
#             mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

#             # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
#             # linux下直接输出正的，不需要上下翻转
#             if platform.system() == "Windows":
#                 mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

#             # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
#             # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
#             frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
#             frame = np.frombuffer(frame_data, dtype=np.uint8)
#             frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

#             frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            
#             # 将帧放入队列
#             frame_queue.put((camera_index, frame))

#         except mvsdk.CameraException as e:
#             if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
#                 print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))

#     # 关闭相机
#     mvsdk.CameraUnInit(hCamera)

#     # 释放帧缓存
#     mvsdk.CameraAlignFree(pFrameBuffer)

# def display_frames(frame_queue):
#     windows = {}
#     while True:
#         if not frame_queue.empty():
#             camera_index, frame = frame_queue.get()
#             window_name = f"Camera {camera_index + 1}"
#             if window_name not in windows:
#                 windows[window_name] = True
#             cv2.imshow(window_name, frame)
        
#         if (cv2.waitKey(1) & 0xFF) == ord('q'):
#             break

# def main():
#     frame_queue = Queue()
#     threads = []
#     for i in range(2):  # 假设有两个摄像头
#         t = threading.Thread(target=camera_loop, args=(i, frame_queue))
#         threads.append(t)
#         t.start()

#     display_thread = threading.Thread(target=display_frames, args=(frame_queue,))
#     display_thread.start()

#     for t in threads:
#         t.join()
    
#     display_thread.join()
#     cv2.destroyAllWindows()

# main()


#coding=utf-8
# import cv2
# import numpy as np
# import mvsdk

# class mindvisionCamera:
#     """
#     迈德威视相机类
#     """
#     def __init__(self, camera_index):
#         # 枚举相机
#         self.DevList = mvsdk.CameraEnumerateDevice()
#         self.nDev = len(self.DevList)
#         if self.nDev < 1:
#             raise Exception("No camera was found!")

#         if camera_index >= self.nDev:
#             raise Exception(f"Camera index {camera_index} is out of range. Only {self.nDev} cameras found.")

#         self.DevInfo = self.DevList[camera_index]
#         print(f"Opening camera {camera_index}: {self.DevInfo.GetFriendlyName()} {self.DevInfo.GetPortType()}")

#         # 打开相机
#         self.hCamera = 0
#         try:
#             self.hCamera = mvsdk.CameraInit(self.DevInfo, -1, -1)
#         except mvsdk.CameraException as e:
#             raise Exception(f"CameraInit Failed({e.error_code}): {e.message}")

#         # 获取相机特性描述
#         self.cap = mvsdk.CameraGetCapability(self.hCamera)

#         # 判断是黑白相机还是彩色相机
#         self.monoCamera = (self.cap.sIspCapacity.bMonoSensor != 0)

#         # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
#         if self.monoCamera:
#             mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
#         else:
#             mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

#         # 相机模式切换成连续采集
#         mvsdk.CameraSetTriggerMode(self.hCamera, 0)

#         # 手动曝光，曝光时间30ms
#         mvsdk.CameraSetAeState(self.hCamera, 0)
#         mvsdk.CameraSetExposureTime(self.hCamera, 20 * 1000)

#         # 让SDK内部取图线程开始工作
#         mvsdk.CameraPlay(self.hCamera)

#         # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
#         self.FrameBufferSize = self.cap.sResolutionRange.iWidthMax * self.cap.sResolutionRange.iHeightMax * (1 if self.monoCamera else 3)

#         # 分配RGB buffer，用来存放ISP输出的图像
#         self.pFrameBuffer = mvsdk.CameraAlignMalloc(self.FrameBufferSize, 16)

#     def capture_frame(self):
#         # 从相机取一帧图片
#         try:
#             pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
#             mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
#             mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)

#             # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
#             # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
#             frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
#             frame = np.frombuffer(frame_data, dtype=np.uint8)
#             frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

#             frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
#             return frame

#         except mvsdk.CameraException as e:
#             if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
#                 print(f"CameraGetImageBuffer failed({e.error_code}): {e.message}")
#             return None

#     def __del__(self):
#         # 关闭相机
#         mvsdk.CameraUnInit(self.hCamera)
#         # 释放帧缓存
#         mvsdk.CameraAlignFree(self.pFrameBuffer)
#         cv2.destroyAllWindows()

# def main():
#     left_camera = mindvisionCamera(1)
#     right_camera = mindvisionCamera(0)

#     while True:
#         left_frame = left_camera.capture_frame()
#         right_frame = right_camera.capture_frame()

#         frame = cv2.hconcat([cv2.rotate(left_frame,cv2.ROTATE_180), cv2.rotate(right_frame,cv2.ROTATE_180)])
#         cv2.imshow('left and right', frame)
#         cv2.waitKey(1)

# main()

import cv2

cap0 = cv2.VideoCapture(2)
print('here')
cap1 = cv2.VideoCapture(0)
print('here')

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    cv2.imshow('left', frame0)
    cv2.imshow('right', frame1)

    frame = cv2.hconcat([frame0, frame1])
    cv2.imshow('left and right', frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap0.release()

# import cv2

# cap = cv2.VideoCapture('/dev/video2')
# while True:
#     ret, frame = cap.read()
#     if ret:
#         cv2.imshow('frame', frame)
#         if (cv2.waitKey(1) & 0xFF) == ord('q'):
#             break
#     else:
#         break

