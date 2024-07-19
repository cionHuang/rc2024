import serial

# 定义串口类，聚合所有程序中会使用的功能
class TTL:

    # 初始化串口信息
    def __init__(self):
        self.ser=serial.Serial('/dev/ttyUSB0', 115200)

    # 为串口返回列表数组
    def digitArray(self,num):  
        array = [0, 0, 0, 0,
                0]  # 1位：符号、校验位 ； 2位：yaw的千位 ：3为：yaw百位 类推     1负2正3错误4成功
        digitArray = [int(digit) for digit in str(abs(num))]
        lenth = len(digitArray)
        if lenth > 4:
            array = [3, 0, 0, 0, 0]  # 若数组长度大于4,说明yaw结算出错，返回首位为3的数组
            return array
        else:
            if num < 0:
                array[0] = 1  # 若小于0,返回的数组首位为1
                for i in range(lenth):
                    array[4 - i] = digitArray[lenth - 1 - i]
                return array
            else:
                array[0] = 2  # 若大于0,返回的数组首位为2
                for i in range(lenth):
                    array[4 - i] = digitArray[lenth - 1 - i]
                return array

    # 串口读取程序
    def portReading(self):
        data = self.ser.read(5)
        # print(len(data))
        if data:
            # 将两个字节的数据转换为十进制数
            end = data[4]
            if end == 255:
                ascii = data[:2].decode('ascii')
                value = int.from_bytes(data[2:4], byteorder='little')
                return ascii , value ,end
            else:
                raise Exception('End code error!')
        else:
            raise Exception('No data!')
        
    # 结束串口通信
    def killport(self):
        self.ser.close()

    # 格式转换,发送数据
    def int_to_bytes(self,value):
        # 将十进制数转换为十六进制字符串，并去掉开头的 '0x'，然后填充为4位，即两个字节
        hex_string = format(value, '04x')
        # 将十六进制字符串分为两个字节并返回
        bytes_written = self.ser.write(bytes.fromhex(hex_string))

        if bytes_written == len(bytes.fromhex(hex_string)):
            print(f"Serial Write successful:{value}")
        else:
            raise Exception('Serial Write error!')
        
    # 新数据收发
    def message(self,flag,*data):
        # 取球模式
        if flag == 'SC':
            str_data = "and".join(map(str, data))
            message = f"SC{str_data}SC"
        
        # 决策模式
        if flag == 'DC':
            message = f"DC{data}DC"

        print('message',message)

        byte_message = message.encode('utf-8')

        print('encode',byte_message)

        # 通过串口发送字节数据
        bytes_written = self.ser.write(byte_message)
        
        # 检查写入字节数是否正确
        if bytes_written == len(byte_message):
            print(f"Serial Write successful: {message}")
        else:
            raise Exception('Serial Write error!')