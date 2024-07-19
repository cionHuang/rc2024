import serial
import cv2
import numpy as np
import crcmod


def find_center(image):
    """查找图像中心"""
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    lower_white = (105, 100, 100)
    upper_white = (139, 255, 255)
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    _, thresh = cv2.threshold(white_mask, 110, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            image = cv2.circle(image, center, radius, (25, 0, 255), 2)
            m = cv2.moments(contour)
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
            return cx, cy, radius

    return None


def transform_coordinates(x, y, radius):
    """转换坐标"""
    camera_matrix = np.array([[542.51079468, 0, 305.3305253],
                              [0, 542.04531159, 230.61392353],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.array([[0.13869922, -0.14081154, -0.0087059, -0.00547145, -0.55545688]], dtype="double")
    camera_points_3d = np.array([(0, 7.5, 0), (0, -7.5, 0), (7.5, 0, 0), (-7.5, 0, 0)], dtype="double")
    image_points_2d = np.array([(x, y + radius), (x, y - radius), (x + radius, y), (x - radius, y)], dtype="double")
    _, rvec, tvec = cv2.solvePnP(camera_points_3d, image_points_2d, camera_matrix, dist_coeffs)
    R, _ = cv2.Rodrigues(rvec)
    coordinates = -np.dot(R.T, tvec)
    return coordinates[0]


def calculate_crc8(data):
    """计算CRC8校验"""
    crc8 = crcmod.predefined.Crc('crc-8')
    crc8.update(bytes(data[:-1]))
    return crc8.crcValue


def receive_data():
    """接收数据"""
    ser = serial.Serial(port="/dev/ttyUSB0", baudrate=115200, timeout=1)
    buffer = []

    while True:
        com_input = ser.read(1)
        if com_input:
            buffer.append(ord(com_input))
            if len(buffer) >= 3 and buffer[0] == 0xFF and buffer[2] == calculate_crc8(buffer):
                return buffer[1]
            elif len(buffer) >= 3 and buffer[0] == 0xFF and buffer[2] != calculate_crc8(buffer):
                print("CRC Error")
                buffer.clear()
                continue
            elif buffer[0] != 0xFF:
                buffer.clear()


def send_data(coordinates):
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    coordinate_int = int(coordinates[0] * 1000)  # Convert to millimeters with three decimal places
    buffer = [0xFF, 0, 0, 0]
    buffer[1] = (coordinate_int >> 8) & 0xFF
    buffer[2] = coordinate_int & 0xFF
    buffer[3] = calculate_crc8(buffer)
    print(buffer)
    ser.write(bytes(buffer))
    ser.close()


def main():
    stop = 1
    while True:
        data = receive_data()
        if data == 0x00:
            print("start...")
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("Camera not found")
            while True:
                ret, frame = camera.read()
                if not ret:
                    print("Frame not received")
                    break
                result = find_center(frame)
                if result:
                    cx, cy, radius = result
                    coordinates = transform_coordinates(cx, cy, radius)
                    send_data(coordinates)
                    print(coordinates)
                else:
                    print("Center not found")
                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop = 0
                    break
                data = receive_data()
                if data == 0x01:
                    camera.release()
                    cv2.destroyAllWindows()
                    break

        print("End----------")
        if stop == 0:
            break


if __name__ == "__main__":
    main()
