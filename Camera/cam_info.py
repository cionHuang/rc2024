import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# 遍历所有可能的参数并获取它们的值
params = [
    cv2.CAP_PROP_POS_MSEC,            # 当前帧的播放时间（以毫秒为单位）
    cv2.CAP_PROP_POS_FRAMES,          # 当前帧的索引
    cv2.CAP_PROP_POS_AVI_RATIO,       # 视频文件的相对位置（0：开始，1：结束）
    cv2.CAP_PROP_FRAME_WIDTH,         # 帧的宽度
    cv2.CAP_PROP_FRAME_HEIGHT,        # 帧的高度
    cv2.CAP_PROP_FPS,                 # 帧率
    cv2.CAP_PROP_FOURCC,              # 视频编解码器的四字符代码
    cv2.CAP_PROP_FRAME_COUNT,         # 视频中的总帧数
    cv2.CAP_PROP_FORMAT,              # 图像格式
    cv2.CAP_PROP_MODE,                # 捕获模式
    cv2.CAP_PROP_BRIGHTNESS,          # 亮度
    cv2.CAP_PROP_CONTRAST,            # 对比度
    cv2.CAP_PROP_SATURATION,          # 饱和度
    cv2.CAP_PROP_HUE,                 # 色调
    cv2.CAP_PROP_GAIN,                # 增益
    cv2.CAP_PROP_EXPOSURE,            # 曝光
    cv2.CAP_PROP_AUTO_EXPOSURE,       # 自动曝光
    cv2.CAP_PROP_AUTOFOCUS,           # 自动对焦
    cv2.CAP_PROP_AUTO_WB              # 自动白平衡
]

for param in params:
    value = cap.get(param)
    print(f"Parameter {param}: {value}")

# 释放资源
cap.release()
cv2.destroyAllWindows()





