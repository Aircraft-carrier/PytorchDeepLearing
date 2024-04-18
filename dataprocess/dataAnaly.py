from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np

import sys
sys.path.append('E:\\challenge\\PytorchDeepLearing-main\\')
from dataprocess.utils import file_name_path

image_pre = ".nii.gz"
mask_pre = ".nii.gz"

def getImageSizeandSpacing(aorticvalve_path):
    """
    get image and spacing
    :return:
    """
    file_path_list = file_name_path(aorticvalve_path, False, True)
    size = []
    spacing = []
    for subsetindex in range(len(file_path_list)):
        if mask_pre in file_path_list[subsetindex]:
            mask_name = file_path_list[subsetindex]
            mask_gt_file = aorticvalve_path + "/" + mask_name
            src = sitk.ReadImage(mask_gt_file, sitk.sitkUInt8)
            imageSize = src.GetSize()
            imageSpacing = src.GetSpacing()
            size.append(np.array(imageSize))
            spacing.append(np.array(imageSpacing))
            print("image size,image spacing:", (imageSize, imageSpacing))
    print("mean size,mean spacing:", (np.mean(np.array(size), axis=0), np.mean(np.array(spacing), axis=0)))


if __name__ == "__main__":
    aorticvalve_path = r"E:\challenge\raw_data"
    # aorticvalve_path = r"E:\challenge\data\MM-WHS2017\processtage\validation\Image"
    getImageSizeandSpacing(aorticvalve_path)


    '''
    >>> python dataprocess/dataAnaly.py
    files: ['patient1_C0.nii.gz', 'patient2_T2.nii.gz', 'patient3_T2_manual.nii.gz']
    image size,image spacing: ((256, 256, 12), (1.25, 1.25, 11.999996185302734))
    image size,image spacing: ((288, 288, 3), (1.3194444179534912, 1.3194444179534912, 23.000001907348633))
    image size,image spacing: ((256, 256, 6), (1.3671875, 1.3671875, 12.000001907348633))
    mean size,mean spacing: (array([266.66666667, 266.66666667,   7.        ]), array([ 1.31221064,  1.31221064, 15.66666667]))
'''

# 这段代码定义的函数 getImageSizeandSpacing 的目的是从给定的文件夹路径（aorticvalve_path）中读取所有的 .nii.gz 格式的文件（通常是医学图像），并计算这些图像的尺寸（size）和间距（spacing）。
# 对于每个找到的 .nii.gz 文件，如果文件名中包含 mask_pre（在这里被设置为 .nii.gz），那么它将被认为是一个掩码（mask）文件，并计算其尺寸和间距。
# 函数的输入参数 aorticvalve_path 是一个字符串，它代表了一个文件夹的路径，该文件夹应该包含 .nii.gz 格式的图像文件。这些文件可以是原始的医学图像数据或者是分割后的掩码图像。
# 具体来说，函数的工作流程如下：
# 使用 file_name_path 函数（这个函数应该是在 'E:\\challenge\\PytorchDeepLearing-main\\' 路径下的 dataprocess.utils 模块中定义的）获取 aorticvalve_path 路径下所有的文件列表。
# 遍历这个文件列表，检查每个文件名是否包含 mask_pre（在这里是 .nii.gz）。
# 如果文件名包含 mask_pre，则读取这个文件（使用 SimpleITK 库），获取其尺寸和间距，并将这些信息添加到 size 和 spacing 列表中。
# 打印每个读取的文件的尺寸和间距。
# 计算并打印所有读取文件的平均尺寸和平均间距。
# 在 __main__ 部分，代码使用两个可能的 aorticvalve_path 路径之一来调用 getImageSizeandSpacing 函数。这两个路径分别指向不同的数据集，但都是指向包含 .nii.gz 文件的文件夹。
# 因此，这段代码要求的输入是一个包含 .nii.gz 格式文件的文件夹路径。
