# encoding:utf-8

import argparse
import json
import pycocotools.mask as mask_util
import cv2
import numpy as np
from imantics import Mask
import os
from tqdm import tqdm


def mask2polygons(mask):
    output = []
    # 得到掩码对应的全部像素点
    polygons_list = (Mask(mask).polygons()).points

    # 对像素点进行均匀采样生成多边形边界框
    for polygons in polygons_list:
        saved_length = 15 if len(polygons) > 300 else 10 if len(polygons) > 200 else 5 \
            if len(polygons) > 100 else 2 if len(polygons) > 50 else 1

        polygons = np.concatenate((polygons[::saved_length], polygons[-1:]))
        output.append(polygons.tolist())
    return output[0]


def easydata2labelme(name, img_path, json_path, out_dir):
    """
    :param img_path: 待转换的图片路径
    :param json_path: Easydata导出的json文件路径
    :param out_dir: 转换后的json文件路径
    :return:
    """
    if not os.path.exists(img_path):
        print(img_path + " is not exists!")
        return
    if not os.path.exists(json_path):
        print(json_path + " is not exists!")
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(json_path, 'r', encoding='utf8')as fp:
        results = json.load(fp)

    ori_img = cv2.imread(img_path).astype(np.float32)
    height, width = ori_img.shape[:2]
    data = {}

    labels = {
        'red': '1',
        'blue': '2',
        'purple': '3',
    }
    # 版本号对应的是环境中安装的labelme的版本
    data["version"] = "5.4.1"
    data["flags"] = {}
    data["shapes"] = []

    for item in results['labels']:
        # Draw bbox
        label = labels[item['name']]  # 所属类别
        points = []
        shape_type = item['shape']
        if shape_type == "brush":
            # Draw mask
            rle_obj = {"counts": item['mask'],
                       "size": [height, width]}
            mask = mask_util.decode(rle_obj)
            points = mask2polygons(mask)
            shape_type = "polygon"
        elif shape_type == "polygon":
            ori_points = item["meta"]["points"]  # 列表嵌套字典
            points = []
            for idx in ori_points:
                ls = []
                x = idx["x"]
                y = idx["y"]
                ls.append(x)
                ls.append(y)
                points.append(ls)
        elif shape_type == "circle":
            ori_points = item["meta"]
            points = []
            radius = ori_points["radius"]
            center_x = ori_points["center"]["x"]
            center_y = ori_points["center"]["y"]
            points.append([center_x, center_y])
            points.append([center_x + radius, center_y])

        shapes = {}
        shapes["label"] = label
        shapes["points"] = points
        shapes["group_id"] = None
        shapes["shape_type"] = shape_type
        shapes["flags"] = {}

        data["shapes"].append(shapes)

    data["imagePath"] = name
    data["imageData"] = None
    data["imageHeight"] = height
    data["imageWidth"] = width

    json_name = json_path.split('/')[-1]
    out_path = os.path.join(out_dir, json_name)
    with open(out_path, 'w') as f:
        json.dump(data, f)


def main():
    easydata_dir = '/Users/harbourhaobo/Desktop/rc2024/data/data_set/ball_240207/images'
    out_dir = '/Users/harbourhaobo/Desktop/rc2024/data/data_set/ball_240207/labelme_json'

    # for循环拼接路径
    for path in tqdm(os.listdir(easydata_dir)):
        if path.split('.')[-1] == 'json':
            continue
        img_path = os.path.join(easydata_dir, path)
        json_path = easydata_dir + '/' + path.split('.')[0] + '.json'
        easydata2labelme(path, img_path, json_path, out_dir)


if __name__ == '__main__':
    main()
