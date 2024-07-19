import numpy as np

# 定义相机坐标系的轴方向向量
camera_x_axis = np.array([0.8755, 0.4255, -0.2346])
camera_y_axis = np.array([-0.4081, 0.9064, 0.1094])
camera_z_axis = np.array([-0.2593, 0, -0.9658])

# 计算向量之间的点积
dot_product_xy = np.dot(camera_x_axis, camera_y_axis)
dot_product_xz = np.dot(camera_x_axis, camera_z_axis)
dot_product_yz = np.dot(camera_y_axis, camera_z_axis)

# 输出点积结果
print(f"Dot product of camera_x_axis and camera_y_axis: {dot_product_xy}")
print(f"Dot product of camera_x_axis and camera_z_axis: {dot_product_xz}")
print(f"Dot product of camera_y_axis and camera_z_axis: {dot_product_yz}")

# 检查向量是否正交
def is_orthogonal(dot_product, tolerance=0.01):
    return abs(dot_product) < tolerance

print(f"camera_x_axis and camera_y_axis are orthogonal: {is_orthogonal(dot_product_xy)}")
print(f"camera_x_axis and camera_z_axis are orthogonal: {is_orthogonal(dot_product_xz)}")
print(f"camera_y_axis and camera_z_axis are orthogonal: {is_orthogonal(dot_product_yz)}")
