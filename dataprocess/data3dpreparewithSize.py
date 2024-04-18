from __future__ import print_function, division

import os
import SimpleITK as sitk
import numpy as np

import sys
sys.path.append('E:\\challenge\\PytorchDeepLearing-main\\')


from dataprocess.utils import file_name_path, resize_image_itkwithsize, normalize

image_dir = "Image"
mask_dir = "Mask"


def preparesampling3dtraindata(datapath, trainImage, trainMask, shape=(96, 96, 96)):
    newSize = shape
    dataImagepath = datapath + "/" + image_dir
    dataMaskpath = datapath + "/" + mask_dir
    all_files = file_name_path(dataImagepath, False, True)
    for subsetindex in range(len(all_files)):
        print(subsetindex)  # !!
        print(all_files)    # !!
        mask_name = all_files[subsetindex]
        mask_gt_file = dataMaskpath + "/" + mask_name
        masksegsitk = sitk.ReadImage(mask_gt_file)
        image_name = all_files[subsetindex]
        image_gt_file = dataImagepath + "/" + image_name
        imagesitk = sitk.ReadImage(image_gt_file)

        _, resizeimage = resize_image_itkwithsize(imagesitk, newSize, imagesitk.GetSize(),
                                                  sitk.sitkLinear)
        _, resizemask = resize_image_itkwithsize(masksegsitk, newSize, masksegsitk.GetSize(),
                                                 sitk.sitkNearestNeighbor)
        # sitk.WriteImage(resizeimage, 'resizeimage.nii.gz')
        # sitk.WriteImage(resizemask, 'resizemask.nii.gz')
        resizemaskarray = sitk.GetArrayFromImage(resizemask)
        resizeimagearray = sitk.GetArrayFromImage(resizeimage)
        resizeimagearray = normalize(resizeimagearray)
        # step 3 get subimages and submasks
        if not os.path.exists(trainImage):
            os.makedirs(trainImage)
        if not os.path.exists(trainMask):
            os.makedirs(trainMask)
        filepath1 = trainImage + "\\" + str(subsetindex) + ".npy"
        filepath = trainMask + "\\" + str(subsetindex) + ".npy"
        np.save(filepath1, resizeimagearray)
        np.save(filepath, resizemaskarray)

# 这段代码的功能是准备用于训练的三维数据。具体实现步骤如下：

# 1. 定义了一个函数`preparesampling3dtraindata`，接受输入参数`datapath`（数据路径）、`trainImage`（训练图像路径）、`trainMask`（训练掩膜路径）以及可选参数`shape`（数据尺寸，默认为(96, 96, 96)）。
# 2. 将输入的数据路径和文件夹名称拼接成完整的图像和掩膜路径。
# 3. 获取数据路径下的所有文件名。
# 4. 遍历每个文件，读取对应的图像和掩膜。
# 5. 调用`resize_image_itkwithsize`函数，将图像和掩膜调整为指定尺寸`newSize`。
# 6. 将调整后的图像和掩膜转换为数组。
# 7. 对图像数组进行归一化处理。
# 8. 创建训练图像和训练掩膜的文件夹（如果不存在）。
# 9. 将处理后的图像和掩膜数组保存为.npy文件，文件名为对应的索引。

# 总体来说，这段代码的功能是将原始数据调整为指定尺寸后保存为.npy文件，用于后续的训练过程。




def preparetraindata():
    """
    :return:
    """
    src_train_path = r"E:\challenge\data\MM-WHS2017\processtage\train"
    source_process_path = r"E:\challenge\data\MM-WHS2017\trainstage\train"
    outputimagepath = source_process_path + "/" + image_dir
    outputlabelpath = source_process_path + "/" + mask_dir
    preparesampling3dtraindata(src_train_path, outputimagepath, outputlabelpath, (112, 112, 128))


def preparevalidationdata():
    """
    :return:
    """
    # src_train_path = r"D:\challenge\data\KiPA2022\processstage\validation"
    # source_process_path = r"D:\challenge\data\KiPA2022\trainstage\validation"
    src_train_path = r"E:\challenge\data\MM-WHS2017\processtage\validation"
    source_process_path = r"E:\challenge\data\MM-WHS2017\trainstage\validation"
    outputimagepath = source_process_path + "/" + image_dir
    outputlabelpath = source_process_path + "/" + mask_dir
    preparesampling3dtraindata(src_train_path, outputimagepath, outputlabelpath, (112, 112, 128))


if __name__ == "__main__":
    preparetraindata()
    preparevalidationdata()


# `SimpleITK` 是一个用于医学图像处理的强大库，它是 `ITK (Insight Segmentation and Registration Toolkit)` 的 Python 封装。`SimpleITK` 提供了一种简单而高效的方式来处理医学图像数据，具有以下主要作用：

# 1. 读取和保存医学图像数据：`SimpleITK` 提供了读取和保存常见医学图像格式（如DICOM、NIfTI、NRRD等）的功能，方便用户处理医学图像数据。

# 2. 图像处理和分析：`SimpleITK` 提供了丰富的图像处理和分析功能，包括图像平滑、滤波、配准、分割、特征提取等，帮助用户进行医学图像分析和研究。

# 3. 图像可视化：`SimpleITK` 支持将医学图像数据可视化，帮助用户直观地观察和分析图像数据。

# 4. 算法实现：`SimpleITK` 封装了许多常用的医学图像处理算法，用户可以直接调用这些算法进行图像处理，而无需从头实现。

# 综合来看，`SimpleITK` 是一个功能强大且易于使用的库，特别适用于医学图像处理领域。
