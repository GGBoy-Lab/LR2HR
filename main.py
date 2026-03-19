import sys
import os

import cv2
import json

import retinex

data_path = 'input'
img_list = os.listdir(data_path)
if len(img_list) == 0:
    print('Data directory is empty.')
    exit()

with open('config.json', 'r') as f:
    config = json.load(f)

# 创建输出目录
output_dirs = {
    'original': 'output/original',
    'msrcr': 'output/msrcr',
    'amsrcr': 'output/amsrcr',
    'msrcp': 'output/msrcp'
}

for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

for img_name in img_list:
    if img_name == '.gitkeep':
        continue

    img = cv2.imread(os.path.join(data_path, img_name))

    # 获取不带扩展名的文件名
    name_without_ext = os.path.splitext(img_name)[0]

    img_msrcr = retinex.MSRCR(
        img,
        config['sigma_list'],
        config['G'],
        config['b'],
        config['alpha'],
        config['beta'],
        config['low_clip'],
        config['high_clip']
    )

    img_amsrcr = retinex.automatedMSRCR(
        img,
        config['sigma_list']
    )

    img_msrcp = retinex.MSRCP(
        img,
        config['sigma_list'],
        config['low_clip'],
        config['high_clip']
    )

    #
    original_path = os.path.join(output_dirs['original'], f'{name_without_ext}.jpg')
    msrcr_path = os.path.join(output_dirs['msrcr'], f'{name_without_ext}.jpg')
    amsrcr_path = os.path.join(output_dirs['amsrcr'], f'{name_without_ext}.jpg')
    msrcp_path = os.path.join(output_dirs['msrcp'], f'{name_without_ext}.jpg')

    cv2.imwrite(original_path, img)
    cv2.imwrite(msrcr_path, img_msrcr)
    cv2.imwrite(amsrcr_path, img_amsrcr)
    cv2.imwrite(msrcp_path, img_msrcp)

    shape = img.shape
    cv2.imshow('Image', img)
    cv2.imshow('retinex', img_msrcr)
    cv2.imshow('Automated retinex', img_amsrcr)
    cv2.imshow('MSRCP', img_msrcp)
    cv2.waitKey()
