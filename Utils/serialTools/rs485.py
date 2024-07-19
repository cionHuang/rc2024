# 完成通信定义类
import serial
import struct
import binascii
import time
import crcmod

# 设置串口参数
SERIAL_PORT = '/dev/ttyUSB0'  # 根据实际情况设置
BAUD_RATE = 115200
TIMEOUT = 1  # 设置超时时间，单位为秒

crc32_mpeg2 = crcmod.mkCrcFun(0x104C11DB7, initCrc=0xFFFFFFFF, xorOut=0x00000000, rev=False)

class RS485:
    """
    RS485通信类
    """

    def __init__(self):
        """
        初始化
        """
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        print(f"打开串口 {SERIAL_PORT}, 波特率 {BAUD_RATE}")
        # 建立缓存区
        self.buffer = bytearray()

    def WT_messagae(self):
        """
        发送数据WT心跳同步信息
        """
        heards = b'\xFF' # 帧头
        chars = b'WT' # 标志位
        data = b'\x00' * 9 # 数据位
        crc = crc32_mpeg2(heards + chars + data)  # CRC校验
        package = heards + chars + data + struct.pack('I', crc) # 打包数据
        self.ser.write(package)
        print(f"发送给下位机的数据: {package}")
        self.ser.flush()

    def SC_messagae(self, x_pos, y_pos):
        """
        发送寻球信息
        """
        heards = b'\xFF'
        chars = b'SC'
        x_data = struct.pack('f', x_pos)
        y_data = struct.pack('f', y_pos)
        data = x_data + y_data
        barn = b'\x00'
        crc = crc32_mpeg2(heards + chars + data + barn)
        package = heards + chars + data + barn +struct.pack('I', crc)
        self.ser.write(package)
        print(f"发送给下位机的数据: {package}")
        self.ser.flush()

    def DC_messagae(self, barn_index):
        """
        发送谷仓决策信息
        """
        heards = b'\xFF'
        chars = b'DC'
        x_data = b'\x00' * 4
        y_data = b'\x00' * 4
        data = x_data + y_data
        barn = struct.pack('I', barn_index)
        crc = crc32_mpeg2(heards + chars + data + barn) 
        package = heards + chars + data + barn +struct.pack('I', crc)
        self.ser.write(package)
        print(f"发送给下位机的数据: {package}")
        self.ser.flush()

    def ST_messagae(self):
        """
        发送初始化信息
        """
        heards = b'\xFF'
        chars = b'ST'
        data = b'\x00' * 9
        crc = crc32_mpeg2(heards + chars + data) 
        package = heards + chars + data + struct.pack('I', crc)
        self.ser.write(package)
        print(f"发送给下位机的数据: {package}")
        self.ser.flush()

    def receive(self):
        """
        接收数据
        """


        # 建立新的数据帧
        frame = bytearray()

        if self.ser.in_waiting > 0 :
            
            self.buffer.extend(self.ser.read(self.ser.in_waiting))  # 将接收到的数据添加到缓冲区

            while len(self.buffer) >= 8:

                # 读取帧头
                header = self.buffer[:2]

                if header != b'\xAA\x55':

                    # 将数据帧的第一位移动到最后一位
                    first_byte = self.buffer.pop(0)
                    frame = self.buffer[:7] + bytearray([first_byte])
                    print(f'新数组是{repr(frame)}')
                else:
                    print('帧头正确')
                    frame = self.buffer[:8]
                    self.buffer = self.buffer[8:]
                    print('frame',frame)
                if frame[:2] == b'\xAA\x55':
                    # 读取数据
                    chars = frame[2:4]
                    if chars in [b'WT', b'SC',b'DC',b'RE']:
                        # 读取crc32校验位
                        crc_received = struct.unpack('<I',frame[4:])[0]
                        data_to_check = header + chars
                        crc_calculated = crc32_mpeg2(data_to_check)
                        # 校验成功
                        if crc_received == crc_calculated:      
                            # 返回标志位给主程序
                            print('校验成功')
                            return chars.decode('utf-8')
                        else:
                            # 显示接收到的原始16进制数据
                            self.buffer = self.buffer[8:] # 清除缓存
                            print(f'接收到的{hex(crc_received)}，计算的{hex(crc_calculated)}')
                            print("校验失败")
                            return None
                    else:
                        print("数据错误")
                        return None
                else:
                    continue
        else:
            return None



    def close(self):
        """
        关闭串口
        """
        self.ser.close()
        print("关闭串口")

#if __name__ == "__main__":
   # port = RS485()
  #  while True:
 #       port.receive()