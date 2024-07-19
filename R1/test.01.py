import serial  
import crcmod  
import time  
import cv2  
import numpy as np 
import math


def zhongxin(img0):

    img = cv2.GaussianBlur(img0, (3,3),0)#高斯模糊
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = (105, 90, 40)
    upper_white= (139, 255, 255)
    white_mask2 = cv2.inRange(hsv_image, lower_white, upper_white)
    ret,thresh=cv2.threshold(white_mask2,110,255,0)
    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    a = -1
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])  # 计算面积
        if area > 10000:
            a = i  # 用面积筛选轮廓
            break  # 找到符合条件的轮廓后立即退出循环  
    if a != -1:
        cnt = contours[a]
        area = cv2.contourArea(contours[a])#计算面积
        m=cv2.moments(cnt)
        x,y,w,h  = cv2.boundingRect(cnt)
        center = (int(x),int(y))
        cv2.rectangle(img0, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cx=int(m['m10']/m['m00'])
        cy=int(m['m01']/m['m00'])#计算中心
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon,True)
        cv2.drawContours(img0,[approx],-1,(0,255,0),3)
        return a,cx,cy,w



def duizheng(x2, y2, w): 
    # 相机内参矩阵和畸变系数  
    camera_matrix = np.array([[542.51079468, 0, 305.3305253], [0, 542.04531159, 230.61392353], [0, 0, 1]], dtype="double") 
    dist_coeffs = np.array([[0.13869922, -0.14081154, -0.0087059, -0.00547145, -0.55545688]], dtype="double")  
    # 假设的3D世界坐标点和对应的2D图像坐标点（用于PnP）  
    camera_points_3d = np.array([  
        (0, 2.1, 0),  
        (0, -2.1, 0),  
        (2.1, 0, 0),  
        (-2.1, 0, 0)], dtype="double")  
    image_points_2d = np.array([  
        (x2, y2 + w/2),  
        (x2, y2 - w/2),  
        (x2 + w/2, y2),  
        (x2 - w/2, y2)], dtype="double")  
    
    # 执行PnP算法  
    ret, rvec, tvec = cv2.solvePnP(camera_points_3d, image_points_2d, camera_matrix, dist_coeffs)  
    R, _ = cv2.Rodrigues(rvec)  # 将旋转向量转为旋转矩阵
    # 相机的世界坐标
    zuobiao= -np.dot(R.T, tvec)
    # 输出结果
    """"
    print("x距离cm:",zuobiao[0] )
    print("y距离cm:",zuobiao[1] )
    print("z距离cm:",zuobiao[2] )
    """
    return zuobiao  



# 定义CRC8计算函数  
def crc8(data):  
    crc8 = crcmod.predefined.Crc('crc-8')  
    crc8_value = crc8.new(bytes(data[:-1])).hexdigest()  # 计算前两个字节的CRC，并转换为十六进制字符串  
    return int(crc8_value, 16)  



# 定义串口接收函数  
def receive():
  ser = serial.Serial(port="/dev/ttyUSB0", baudrate=115200, timeout=1)  
  buffer = []
  # 读取串口输入信息并输出。
  while True:
      com_input = ser.read(1)
      if com_input:   # 如果读取结果非空，则输出
          buffer.append(ord(com_input))  
          if len(buffer) >= 3 and buffer[0] == 0xFF and buffer[2] ==crc8(buffer):  
                  # 检测到起始符且CRC校验正确，返回数据  
              #print ("接收", buffer[1]) 
              return buffer[1]
          elif len(buffer) >= 3 and buffer[0] == 0xFF and buffer[2] != crc8(buffer): 
              print("buffer[2] != crc8(buffer)") 
              # 检测到起始符但CRC校验失败，清除buffer继续等待  
              buffer.clear()  
              continue  
          elif buffer[0] != 0xFF:  
              # 如果不是起始符，清除buffer继续等待  
              buffer.clear() 


# 定义发送函数              
def send(a) : 
  # 初始化串口通信（请根据您的串口配置调整这些参数）  
  ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)  # 例如，使用COM1端口，波特率为9600  
  coordinate_int = int(a * 1000)  # 乘以1000转换为毫米（不是厘米），保留三位小数  
  # 初始化buffer列表  
  buffer = [0xFF, 0, 0,0]  
  # 将整数拆分为高八位和低八位  
  buffer[1] = (coordinate_int >> 8) & 0xFF  
  buffer[2] = coordinate_int & 0xFF  
  # 计算CRC8校验  
  crc= crc8(buffer) # 选择CRC-8算法   
  buffer[3]=crc # 将十六进制字符串转换为整数并添加到buffer中  
  print(buffer)
  # 通过串口发送数据  
  ser.write(bytes(buffer))   
  return buffer 

ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)
stop=1 
while True:
    data = receive()  
    if data==0x00:  
        print("start...") 
        camera = cv2.VideoCapture(0)  # 0代表摄像头设备索引，如果有多个摄像头可以尝试不同的索引号
        if not camera.isOpened():
            print("摄像头未能打开")
        while True:
            ret, frame = camera.read()  # 逐帧捕获
            if not ret:
                print("未能接收帧")
                break
            result = zhongxin(frame)  
            if result is not None:  
                a, x2, y2, radius2 = result  
                if a != -1:
                    zuobiao=duizheng(x2, y2, radius2)    
                    """"
                    zuobiao_strx = f"x:{zuobiao[0]}" 
                    zuobiao_stry = f"y:{zuobiao[1]}" 
                    zuobiao_strz = f"z:{zuobiao[2]}" 
                    cv2.putText(frame, zuobiao_strx, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)   
                    cv2.putText(frame, zuobiao_stry, (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)  
                    cv2.putText(frame, zuobiao_strz, (10, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2) 
                    """ 
                    send(zuobiao[0])
                    print(zuobiao[0])

            else:  
                print("未识别到中心")  
            # 处理这种情况，比如设置默认值或退出函数
            cv2.imshow("Camera", frame)  # 显示图像 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                stop=0 # 按q键退出
                break 
            data = receive()
            if data==0x01:
                camera.release()  # 释放摄像头
                cv2.destroyAllWindows()  # 销毁所有窗口
                break


    print("end----------")
    ser.close()
    if stop== 0: 
        break 