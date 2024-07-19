# import cv2
# import numpy as np

# # 定义窗口名字
# window_name = 'Video Stream'

# # 定义滚动条的回调函数
# def on_trackbar(position):
#     global line_pos
#     line_pos = position
#     pass  # 回调函数暂时为空，稍后实现

# # 创建一个空的画布作为滚动条的背景
# scrollbar_canvas = np.zeros((50, 620, 3), dtype=np.uint8)

# # 在滚动条的背景上绘制两条竖直线
# cv2.line(scrollbar_canvas, (100, 0), (100, 50), (255, 255, 255), 2)
# cv2.line(scrollbar_canvas, (500, 0), (500, 50), (255, 255, 255), 2)

# # 创建一个窗口，并在窗口中显示滚动条的背景
# cv2.namedWindow(window_name)
# cv2.createTrackbar('Line 1', window_name, 100, 500, on_trackbar)
# cv2.createTrackbar('Line 2', window_name, 500, 500, on_trackbar)
# cv2.createTrackbar('Horizontal Line', window_name, 200, 480, on_trackbar)
# cv2.imshow(window_name, scrollbar_canvas)

# # 读取视频流
# cap = cv2.VideoCapture(4)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 获取滚动条的位置
#     line1_pos = cv2.getTrackbarPos('Line 1', window_name)
#     line2_pos = cv2.getTrackbarPos('Line 2', window_name)
#     horizontal_line_pos = cv2.getTrackbarPos('Horizontal Line', window_name)

#     # 在视频帧上绘制两条竖直线
#     cv2.line(frame, (line1_pos, 0), (line1_pos, frame.shape[0]), (0, 255, 0), 2)
#     cv2.line(frame, (line2_pos, 0), (line2_pos, frame.shape[0]), (0, 255, 0), 2)

#     # 在视频帧上绘制水平线
#     cv2.line(frame, (0, horizontal_line_pos), (frame.shape[1], horizontal_line_pos), (255, 0, 0), 2)

#     # 定义十字准星的长度
#     crosshair_size = 20
#     # 在中心画一个横线
#     cv2.line(frame, (320 - crosshair_size, 240), (320 + crosshair_size, 240), (255, 255, 255), 5)
#     # 在中心画一个竖线
#     cv2.line(frame, (320, 240 - crosshair_size), (320, 240 + crosshair_size), (255, 255, 255), 5)
#     # 在窗口中显示视频帧
#     cv2.imshow(window_name, frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import mvsdk

class mindvisionCamera:
    """
    迈德威视相机类
    """
    def __init__(self, camera_index):
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
        mvsdk.CameraSetExposureTime(self.hCamera, 20 * 1000)

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

def update_vertical1(val):
    global vertical1_pos
    vertical1_pos = val

def update_vertical2(val):
    global vertical2_pos
    vertical2_pos = val

def update_horizontal1(val):
    global horizontal1_pos
    horizontal1_pos = val

def update_horizontal2(val):
    global horizontal2_pos
    horizontal2_pos = val

def main():
    global vertical1_pos, vertical2_pos, horizontal1_pos, horizontal2_pos
    vertical1_pos = 320
    vertical2_pos = 320
    horizontal1_pos = 240
    horizontal2_pos = 240

    right_camera = mindvisionCamera(1)
    left_camera = mindvisionCamera(0)

    cv2.namedWindow('left and right')
    cv2.createTrackbar('Vertical Line 1', 'left and right', vertical1_pos, 1280, update_vertical1)
    cv2.createTrackbar('Vertical Line 2', 'left and right', vertical2_pos, 1280, update_vertical2)
    cv2.createTrackbar('Horizontal Line 1', 'left and right', horizontal1_pos, 480, update_horizontal1)
    cv2.createTrackbar('Horizontal Line 2', 'left and right', horizontal2_pos, 480, update_horizontal2)

    while True:
        left_frame = left_camera.capture_frame()
        right_frame = right_camera.capture_frame()

        if left_frame is not None and right_frame is not None:
            frame = cv2.hconcat([cv2.rotate(left_frame, cv2.ROTATE_180), cv2.rotate(right_frame, cv2.ROTATE_180)])

            # 绘制竖直线
            cv2.line(frame, (vertical1_pos, 0), (vertical1_pos, 480), (0, 255, 0), 2)
            cv2.line(frame, (vertical2_pos, 0), (vertical2_pos, 480), (0, 255, 0), 2)

            # 绘制水平线
            cv2.line(frame, (0, horizontal1_pos), (1280, horizontal1_pos), (0, 255, 0), 2)
            cv2.line(frame, (0, horizontal2_pos), (1280, horizontal2_pos), (0, 255, 0), 2)

            cv2.imshow('left and right', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

main()
