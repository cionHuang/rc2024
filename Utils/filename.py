# from openni import openni2
# import cv2
# import numpy as np
# if __name__ == "__main__":
#     openni2.initialize()
# dev = openni2.Device.open_any()
# print(dev.get_device_info())
# color_stream = dev.create_color_stream()
# color_stream.start()

# print('start0',color_stream.start())
# cv2.namedWindow('color')

# while True:
#     colorTemplateframe = color_stream.read_frame()

#     print(colorTemplateframe)

#     # colorTemplateframe是VideoFrame类型，所以需要转换为np.ndarry类型。
#     cframe_data = np.array(colorTemplateframe.get_buffer_as_uint8()).reshape([480,640,3])
#     # 通道转换：RGB转为BGR
#     cframe = cv2.cvtColor(cframe_data,cv2.COLOR_RGB2BGR)
#     cv2.imshow("color",cframe)

#     key = cv2.waitKey(1)
#     if int(key) == ord('q'):
#         break
# color_stream.stop()
# dev.close()


from openni import openni2
# 初始化OpenNI2
openni2.initialize()

# 获取所有连接设备的信息
device_list = openni2.Device.enumerate_uris()

# 打印每个设备的信息
print("Connected devices:")
for i, device_info in enumerate(device_list):
    print(f"Device {i}: {device_info}")

# # 假设我们要打开第一个设备
# device_uri = device_list[0].uri

# # 打开指定的设备
# device = openni2.Device.open(device_uri)

# # 获取设备的详细信息
# device_info = device.get_device_info()
# print(f"Opened device: {device_info.name} ({device_info.uri})")

# # 使用设备，例如打开深度流
# depth_stream = device.create_depth_stream()
# depth_stream.start()

# # 停止流并关闭设备
# depth_stream.stop()
# device.close()

# # 关闭OpenNI2
# openni2.unload()
