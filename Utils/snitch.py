import cv2
import numpy as np

# 假设我们已经从标定中得到两个相机的旋转矩阵和平移向量
R2 =  np.array([[-0.5189922 ],
       [ 0.63304619],
       [-2.92957193]])

T2 = np.array([[ 12.95884268],
       [ 42.17133181],
       [999.91332496]])

R1 = np.array([[ 0.71583408],
       [-0.10020293],
       [-2.87420256]])
T1 = np.array([[-166.48853419],
       [  31.14688546],
       [1070.08853424]])
# 计算相对旋转矩阵和平移向量
R12 = np.dot(R1.T, R2)
T12 = np.dot(R1.T, (T2 - T1))

# 加载两幅图像
img1 = cv2.imread('left.jpg')
img2 = cv2.imread('right.jpg')

# 将相对旋转矩阵和平移向量转换为3x4变换矩阵
RT = np.hstack((R12, T12))
RT = np.vstack((RT, [0, 0, 1]))


# 使用变换矩阵对图像进行变换
h, w = img2.shape[:2]
img1_warped = cv2.warpPerspective(img1, RT, (w, h))

# 图像拼接
result = np.maximum(img1_warped, img2)

# 显示结果
cv2.imshow('Stitched Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
