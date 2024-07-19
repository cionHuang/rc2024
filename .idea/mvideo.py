import cv2
import numpy as np
import mvsdk
import os

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
##工具
def find_next_video_index(folder):
    #找到文件夹中下一个可用的视频序号
    index = 1
    while True:
        filename = os.path.join(folder, f"{index}.mp4")
        if not os.path.exists(filename):
            return index
        index += 1
def main():
    folder = 'mvideos'
    # 确保文件夹存在，如果不存在则创建
    os.makedirs(folder, exist_ok=True)
    # 查找文件夹中已有的最大序号
    next_index = find_next_video_index(folder)
    video_filename = os.path.join(folder, f"{next_index}.mp4")

    left_camera = mindvisionCamera(1)
    right_camera = mindvisionCamera(0)
    # 设置视频编码器和帧率  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码  
    fps = 30.0  # 帧率    
    frame_width = 640  # 宽度  
    frame_height = 480  # 高度  
    # 创建 VideoWriter 对象  
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width * 2, frame_height))  # 因为两个帧水平连接，所以宽度加倍  
    while True:

        left_frame = left_camera.capture_frame()
        right_frame = right_camera.capture_frame()

        if left_frame is not None and right_frame is not None:
            frame = cv2.hconcat([cv2.rotate(left_frame, cv2.ROTATE_180), cv2.rotate(right_frame, cv2.ROTATE_180)])
            print(f"Recording video {next_index}.mp4 ...")
            # 写入帧到视频文件
            video_writer.write(frame)

            # 显示录制的视频
            cv2.imshow('Recording', frame)

            # 按下 'q' 键退出录制
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:  
            # 如果任一相机没有捕获到帧，可以跳过这一帧或采取其他措施  
            print(f"no frame")
            pass  
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {video_filename}")


if __name__ == "__main__":
    main()
