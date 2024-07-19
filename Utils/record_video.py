import cv2
import os
import time

output_folder = "recordings"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

recording = False
video_writer = None
record_start_time = None
current_video_number = 1

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_AUTO_WB,0)

while True:
    ret, frame = camera.read()
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') and not recording:
        recording = True
        record_start_time = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_folder, f"recording_{current_video_number:03}.avi")
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (640, 480))
        print(f"Recording started. Press 'w' to stop recording.")
    elif key == ord('w') and recording:
        recording = False
        video_writer.release()
        print(f"Recording stopped. Video saved as recording_{current_video_number:03}.avi.")
        current_video_number += 1

    if recording:
        video_writer.write(frame)

    if key == ord('c'):  # Press 'c' to close the program
        if recording:
            video_writer.release()
        break

camera.release()
cv2.destroyAllWindows()
