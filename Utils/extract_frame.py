import cv2
import os

input_folder = "recordings"
output_folder = "screenshots"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_interval = 10  # 每隔多少帧截图一次

def capture_frames(video_path, output_folder):
    video_capture = cv2.VideoCapture(video_path)
    success, image = video_capture.read()
    count = 0

    while success:
        if count % frame_interval == 0:
            output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_{count:05}.png")
            cv2.imwrite(output_path, image)
        success, image = video_capture.read()
        count += 1

    video_capture.release()

# 遍历录像文件夹下的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith(".avi"):
        video_path = os.path.join(input_folder, filename)
        capture_frames(video_path, output_folder)

print("Screenshots captured and saved successfully.")
