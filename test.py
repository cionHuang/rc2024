from Utils.tools import mindvisionCamera
import cv2

# 调取两个摄像头的画面并拼接显示
def main():
    right_cam = mindvisionCamera(1,5)
    left_cam = mindvisionCamera(0,1)
    while True:
        left_img = left_cam.capture_frame()
        right_img = right_cam.capture_frame()
        # 画面拼接
        frame = cv2.hconcat([cv2.rotate(left_img,cv2.ROTATE_180),cv2.rotate(right_img,cv2.ROTATE_180)])
        cv2.imshow('frame',frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()

# import serial
# import time

# def send_and_wait_for_ack(port, baudrate, data, ack=b'ACK', timeout=1):
#     try:
#         # 打开串口
#         ser = serial.Serial(port, baudrate, timeout=timeout)

#         # 发送数据前设置RS485为发送模式
#         ser.setRTS(True)
#         ser.write(data)
#         ser.flush()  # 确保数据已完全发送
#         time.sleep(0.1)  # 根据硬件情况调整延时
#         ser.setRTS(False)

#         # 等待确认信息
#         ack_received = ser.read(len(ack))

#         if ack_received == ack:
#             print("Data sent and ACK received successfully.")
#         else:
#             print("Data sent but ACK not received.")
        
#         ser.close()
#     except serial.SerialException as e:
#         print(f"Serial exception: {e}")
#     except Exception as e:
#         print(f"Unexpected error: {e}")

# if __name__ == "__main__":
#     port = '/dev/ttyUSB0'  # 根据实际情况修改端口
#     baudrate = 9600
#     data = b'Hello RS485\n'
#     ack = b'ACK'  # 假设设备会发送'ACK'作为确认信息

#     send_and_wait_for_ack(port, baudrate, data, ack)
