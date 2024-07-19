from openni import openni2
import numpy as np  
import cv2
import math

from scipy.interpolate import interp2d

# 定义深度相机类，聚合所有程序中所使用的功能
class depthCam:
   
    # 相机内参
    cameraMatrix = np.array([
        [602.60620769, 0., 325.53789961],  
        [0., 602.664394, 248.45389531],
        [0., 0., 1.]
    ])

    # 存放深度数据的初始化空数组
    depth_array=np.empty((0,0))

    # 深度相机初始化
    def __init__(self,filename):
        openni2.initialize()
        dev = openni2.Device.open_file(filename)
        # 设置相机模式并且开启彩色视频流
        self.depth_stream = dev.create_depth_stream()
        self.depth_stream.set_video_mode(
            openni2.VideoMode(pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM,
                                resolutionX=640,
                                resolutionY=480,
                                fps=30))
        self.depth_stream.start()

        # self.color_stream = dev.create_color_stream()
        # self.color_stream.start()

        # 深度图与彩色图对其
        if dev.is_image_registration_mode_supported(
                openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR):
            dev.set_image_registration_mode(
                openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    
    # 深度信息解析
    def depthCapture(self):
            depth_frame = self.depth_stream.read_frame()
            depth_data = depth_frame.get_buffer_as_uint16()
            self.depth_array = np.ndarray((depth_frame.height, depth_frame.width),
                                     dtype=np.uint16,
                                     buffer=depth_data)

            self.depth_array = cv2.flip(self.depth_array, 1)  # 所有深度信息的数组
            # Convert depth frame to CV_8U format for visualization

            # print(self.depth_array.shape)


            max_depth = self.depth_stream.get_max_pixel_value()  #最大深度值，单位mm 可注释
            # self.depth_visual_bef = cv2.convertScaleAbs(self.depth_array,
            #                                    alpha=255 / max_depth)  # 可注释

            # print(max_depth)
            self.depth_visual = cv2.convertScaleAbs(self.depth_array,
                                               alpha=255 / max_depth)  # 可注释
            return self.depth_visual, self.depth_array # 可视化的深度信息灰度图
    
    # 获取彩色流画面
    # def colorCapture(self):
        
    #     cframe = self.color_stream.read_frame()
    #     cframe_data = np.array(cframe.get_buffer_as_triplet()).reshape([480, 640, 3])
    #     R = cframe_data[:, :, 0]
    #     G = cframe_data[:, :, 1]
    #     B = cframe_data[:, :, 2]
    #     cframe_data = np.transpose(np.array([B, G, R]), [1, 2, 0])

    #     return cframe_data



    
    # 若识别中心点无深度信息，用此函数进行补充
    def depthFinder(self,cx, cy, width, height,depthArray):  # 如果中心点没有深度信息，就向中心点的下面遍历寻找有值的深度，目前来看一般是上方缺失深度信息

        depthData = 0  # 存储深度信息
        dx = width // 20  # x方向的遍历步长
        dy = height // 20  # y方向的遍历步长
        # x = cx
        # y = cy
        flag = 0
        

        if depthArray[cy,cx] != 0: # 深度数据列优先！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            return depthArray[cy,cx]

        else:
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
        
    # 像素坐标转换为世界坐标，以便进行偏航角计算
    def coordinate(self,xy, d):  
        # 世界坐标
        x = (xy[0] - self.cameraMatrix[0, 2]) * d / self.cameraMatrix[0, 0]
        y = (xy[1] - self.cameraMatrix[1, 2]) * d / self.cameraMatrix[1, 1]
        z = d

        yaw = math.degrees(math.atan(x / z))

        # print([x,y,z],'yaw=',yaw)

        return yaw
    


# test = depthCam(b'2bc5/0403@3/8')
# test2 = depthCam(b'2bc5/060f@3/7')

# while True:

#     frame,_ = test.depthCapture()
#     frame2,_ = test2.depthCapture()
#     cv2.imshow('test',frame)
#     cv2.imshow('test2',frame2)
#     cv2.waitKey(1)