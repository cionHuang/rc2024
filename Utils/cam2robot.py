# import numpy as np
# import cv2

# # 相机内参矩阵（假设已知）
# mtx = np.array([[593.1817683329759, 0.0, 350.073179717656],
#                 [0.0, 592.0340497331903, 247.2002113235381],
#                 [0.0, 0.0, 1.0]])

# # 相机畸变系数（假设已知）
# dist = np.array([-0.08128269142535678, 0.7080413947129398, 0.0013881643278435016, 0.012735363429373923, -1.3566172110731252])

# # 棋盘格的尺寸和方块边长
# chessboard_size = (10, 7)
# square_size = 29  # 单位：毫米

# # 准备棋盘格的3D点坐标（单位：毫米）
# objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
# objp *= square_size

# # 读取标定图像并检测角点
# img = cv2.imread('data/frame_0.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

# if ret:
#     corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
#     print("Detected corners:", corners2)
#     ret, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
#     R_cam2target, _ = cv2.Rodrigues(rvec)
#     t_cam2target = tvec

#     # 标定板在机器人坐标系中的位置（以左上角为基准点，单位：毫米）
#     # X_base = 1000  # 0.5米 = 500毫米
#     # Y_base = -339  # 0.2米 = 200毫米
#     # Z_base = 480  # 0.1米 = 100毫米

#     X_base = 339
#     Y_base = -480
#     Z_base = 1000

#     # 假设标定板竖直放置在机器人前方
#     # 标定板的平面与机器人的XY平面垂直，法向量沿Z轴正方向
#     # 旋转矩阵：90度绕X轴旋转
#     R_target2robot = np.array([[0, -1, 0],
#                                [1, 0, 0],
#                                [0, 0, 1]])
#     t_target2robot = np.array([X_base, Y_base, Z_base])

#     # 计算相机到机器人坐标系的旋转和平移
#     R_cam2robot = np.dot(R_target2robot, R_cam2target.T)
#     t_cam2robot = t_target2robot.reshape(3, 1) - np.dot(R_cam2robot, t_cam2target)

#     print("Rotation matrix from camera to robot coordinate system:", R_cam2robot)
#     print("Translation vector from camera to robot coordinate system:", t_cam2robot)

#     # 显示检测到的角点
#     img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
#     cv2.imshow('Detected Corners', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# else:
# #     print("Failed to find chessboard corners.")


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 机器人坐标系原点
# robot_origin = np.array([0, 0, 0])

# # 相机坐标系的原点在机器人坐标系中的坐标
# camera_origin = np.array([243.99, 177.87, 690.98])  # 替换为实际值

# # 相机坐标系的轴方向向量
# # camera_x_axis = np.array([-0.4081, 0.9064, 0.1094])  # 替换为实际值
# # camera_y_axis = np.array([0.2593, 0, 0.9658])  # 替换为实际值
# # camera_z_axis = np.array([0.8755, 0.4255, -0.2346])  # 替换为实际值

# camera_x_axis = np.array([0.8755, 0.4255, -0.2346])  # 替换为实际值
# camera_y_axis = np.array([-0.4081, 0.9064, 0.1094])  # 替换为实际值
# camera_z_axis = np.array([0.2593, 0, 0.9658])  # 替换为实际值

# R = np.column_stack((camera_x_axis, camera_y_axis, camera_z_axis))
# # 输出旋转矩阵
# print("Rotation matrix from camera coordinate system to robot coordinate system:")
# print(R)

# # 创建一个新的3D图形
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制机器人坐标系
# ax.quiver(robot_origin[0], robot_origin[1], robot_origin[2], 1, 0, 0, color='r', length=100.0, label='Robot X')
# ax.quiver(robot_origin[0], robot_origin[1], robot_origin[2], 0, 1, 0, color='g', length=100.0, label='Robot Y')
# ax.quiver(robot_origin[0], robot_origin[1], robot_origin[2], 0, 0, 1, color='b', length=100.0, label='Robot Z')

# # 绘制相机坐标系的轴
# ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2], 
#           camera_x_axis[0], camera_x_axis[1], camera_x_axis[2], 
#           color='r', length=100.0, linestyle='dashed', label='Camera X')
# ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2], 
#           camera_y_axis[0], camera_y_axis[1], camera_y_axis[2], 
#           color='g', length=100.0, linestyle='dashed', label='Camera Y')
# ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2], 
#           camera_z_axis[0], camera_z_axis[1], camera_z_axis[2], 
#           color='b', length=100.0, linestyle='dashed', label='Camera Z')

# # 设置图例
# # ax.legend()

# # 设置坐标轴标签
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

# # 设置图形标题
# ax.set_title('Camera and Robot Coordinate Systems')

# # 显示图形
# plt.show()






# import numpy as np

# # 假设相机坐标系中的点
# P_cam = np.array([[1.0, 0.0, 0.0],
#                   [0.0, 1.0, 0.0],
#                   [0.0, 0.0, 1.0]])

# # 假设机器人坐标系中的点
# P_robot = np.array([[0.8755, 0.4255, -0.2346],
#                     [-0.4081, 0.9064, 0.1094],
#                     [0.2593, 0, 0.9658]])



# # 构建协方差矩阵
# H = P_robot @ P_cam.T

# # SVD分解
# U, S, Vt = np.linalg.svd(H)

# # 计算旋转矩阵
# R = Vt.T @ U.T

# print("Rotation Matrix R:")
# print(R)


import numpy as np

# 假设我们有三个已知的三行一列的向量 y1, y2, y3 和 x1, x2, x3
# 左相机
# y1 = np.array([[1.0], [0.0], [0.0]]) # 机器人坐标系的 x 轴
# y2 = np.array([[0.0], [-1.0], [0.0]]) # 机器人坐标系的 y 轴
# y3 = np.array([[0.0], [0.0], [-1.0]]) # 机器人坐标系的 z 轴

# x1 = np.array([[0.8755], [0.4255], [-0.2346]]) # 相机坐标系的 z 轴对应的机器人坐标系的 x 轴，且方向均指向前方
# x2 = np.array([[-0.4081], [0.9064], [0.1094]]) # 相机坐标系的 x 轴对应的机器人坐标系的 y 轴，且方向相反
# x3 = np.array([[0.2593], [0], [0.9658]]) # 相机坐标系的 y 轴对应的机器人坐标系的 z 轴，且方向相反

# # # 右相机
# y1 = np.array([[1.0], [0.0], [0.0]]) # 机器人坐标系的 x 轴
# y2 = np.array([[0.0], [1.0], [0.0]]) # 机器人坐标系的 y 轴
# y3 = np.array([[0.0], [0.0], [1.0]]) # 机器人坐标系的 z 轴

# # 原始法向量
# roi_x = np.array([[2.65],[-1.22],[-0.7]]) # 实际对应相机光轴z，同向，不变
# roi_y = np.array([[-10.41],[-24.21],[2.79]]) # 实际对应相机x轴，反向，反向延长
# roi_z = np.array([[-3.89],[0],[-14.49]]) # 实际对应相机y轴，反向，反向延长

# # 化单位向量
# x1 = roi_x / np.linalg.norm(roi_x)
# x2 = roi_y / np.linalg.norm(roi_y)
# x3 = roi_z / np.linalg.norm(roi_z)
#左相机
y1 = np.array([[1.0], [0.0], [0.0]]) # 机器人坐标系的 x 轴
y2 = np.array([[0.0], [-1.0], [0.0]]) # 机器人坐标系的 y 轴
y3 = np.array([[0.0], [0.0], [-1.0]]) # 机器人坐标系的 z 轴

# 原始法向量
roi_x = np.array([[2.64],[1.22],[-0.71]]) # 实际对应相机光轴z，同向，不变
roi_y = np.array([[10.41],[-24.21],[-2.79]]) # 实际对应相机x轴，反向，反向延长
roi_z = np.array([[-3.88],[0],[-14.48]]) # 实际对应相机y轴，反向，反向延长

# 化单位向量
x1 = roi_x / np.linalg.norm(roi_x)
x2 = roi_y / np.linalg.norm(roi_y)
x3 = roi_z / np.linalg.norm(roi_z)

# x1 = np.array([[0.8755], [-0.4255], [-0.2346]]) # 相机坐标系的 z 轴对应的机器人坐标系的 x 轴，且方向均指向前方
# x2 = np.array([[-0.4081], [-0.9064], [0.1094]]) # 相机坐标系的 x 轴对应的机器人坐标系的 y 轴，且方向相同指向右方
# x3 = np.array([[-0.2593], [0], [-0.9658]]) # 相机坐标系的 y 轴对应的机器人坐标系的 z 轴，且方向相反

# 将 y1, y2, y3 组合成矩阵 Y
Y = np.hstack([y1, y2, y3])

# 将 x1, x2, x3 组合成矩阵 X
X = np.hstack([x1, x2, x3])

# 求解 A
A = np.dot(Y, np.linalg.inv(X))

print("矩阵 A 为：")
print(A)


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 机器人坐标系原点
# robot_origin = np.array([0, 0, 0])

# # 相机坐标系的原点在机器人坐标系中的坐标
# camera_origin = np.array([243.99, 177.87, 690.98])

# # 相机坐标系的轴方向向量
# camera_x_axis = np.array([0.8755, 0.4255, -0.2346])  # 替换为实际值
# camera_y_axis = np.array([-0.4081, 0.9064, 0.1094])  # 替换为实际值
# camera_z_axis = np.array([0.2593, 0, 0.9658])

# # 旋转矩阵R（相机坐标系到机器人坐标系）
# R = [[ 8.74238574e-01,  4.21949300e-01, -2.34717397e-01],
#        [-4.10402155e-01,  9.05185981e-01,  1.10185627e-01],
#        [ 2.58846930e-01, -3.93876231e-05,  9.65915294e-01]]

# # 创建一个新的3D图形
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制机器人坐标系
# ax.quiver(robot_origin[0], robot_origin[1], robot_origin[2], 
#           1, 0, 0, color='r', arrow_length_ratio=0.1, label='Robot X')
# ax.quiver(robot_origin[0], robot_origin[1], robot_origin[2], 
#           0, 1, 0, color='g', arrow_length_ratio=0.1, label='Robot Y')
# ax.quiver(robot_origin[0], robot_origin[1], robot_origin[2], 
#           0, 0, 1, color='b', arrow_length_ratio=0.1, label='Robot Z')

# # 应用旋转矩阵到相机坐标系的轴
# rotated_camera_x_axis = R @ camera_x_axis
# rotated_camera_y_axis = R @ camera_y_axis
# rotated_camera_z_axis = R @ camera_z_axis

# # 绘制旋转后的相机坐标系的轴
# ax.quiver(robot_origin[0], robot_origin[1], robot_origin[2],
#           rotated_camera_x_axis[0], rotated_camera_x_axis[1], rotated_camera_x_axis[2],
#           color='r', linestyle='dashed', arrow_length_ratio=0.1, label='Rotated Camera X')
# ax.quiver(robot_origin[0], robot_origin[1], robot_origin[2],
#           rotated_camera_y_axis[0], rotated_camera_y_axis[1], rotated_camera_y_axis[2],
#           color='g', linestyle='dashed', arrow_length_ratio=0.1, label='Rotated Camera Y')
# ax.quiver(robot_origin[0], robot_origin[1], robot_origin[2],
#           rotated_camera_z_axis[0], rotated_camera_z_axis[1], rotated_camera_z_axis[2],
#           color='b', linestyle='dashed', arrow_length_ratio=0.1, label='Rotated Camera Z')

# 设置图例
# ax.legend()

# 设置坐标轴标签
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

# 设置图形标题
# ax.set_title('Camera and Robot Coordinate Systems')

# 显示图形
# plt.show()