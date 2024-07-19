import cv2

# 创建一个回调函数，用于调整圆的位置
def update_position(x):
    global y1, y2, y3
    y1 = cv2.getTrackbarPos('Circle 1', 'Video Stream')
    y2 = cv2.getTrackbarPos('Circle 2', 'Video Stream')
    y3 = cv2.getTrackbarPos('Circle 3', 'Video Stream')

# 视频流处理
cap = cv2.VideoCapture(0)
cv2.namedWindow('Video Stream')

# 初始圆的位置
y1 = 200
y2 = 240
y3 = 280

# 创建滚条来调整圆的位置
cv2.createTrackbar('Circle 1', 'Video Stream', y1, 480, update_position)
cv2.createTrackbar('Circle 2', 'Video Stream', y2, 480, update_position)
cv2.createTrackbar('Circle 3', 'Video Stream', y3, 480, update_position)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 在视频流中心画三个圆
    cv2.circle(frame, (320, y1), 10, (0, 255, 0), -1)
    cv2.circle(frame, (320, y2), 10, (0, 255, 0), -1)
    cv2.circle(frame, (320, y3), 10, (0, 255, 0), -1)

    # 显示圆心的行像素坐标
    cv2.putText(frame, f'Y1: {y1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Y2: {y2}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Y3: {y3}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Video Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
