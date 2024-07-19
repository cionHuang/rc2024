import cv2
import os


def capture_video():
    """
    从摄像头捕获视频流,按一次'q'键保存一帧,按esc退出

    Args:
        无

    Returns:
        无

    """
    # 创建保存帧的文件夹
    save_folder = r"data"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 打开视频流，这里以摄像头为例，如果要打开视频文件，将0替换为文件路径
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Couldn't open the video stream.")
        exit()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read a frame from the video stream.")
            break

        # 显示当前帧
        cv2.imshow('Video Stream', frame)

        # 检查是否按下了'q'键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 保存当前帧到文件夹
            frame_filename = os.path.join(save_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
            frame_count += 1
        elif key == 27:  # ESC键
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_video()
