import sys
sys.path.append('E:\\challenge\\PytorchDeepLearing-main\\')

from dataprocess.Augmentation.ImageAugmentation import DataAug3D
if __name__ == '__main__':
    aug = DataAug3D(rotation=10, width_shift=0.01, height_shift=0.01, depth_shift=0, zoom_range=0,
                    vertical_flip=True, horizontal_flip=True)
    aug.DataAugmentation('data/traindata.csv', 10, aug_path='E:/challenge/data/MM-WHS2017/trainstage/augtrain/')


'''数据集增强'''
# 这段代码定义了一个名为DataAugmentation的方法，用于数据增强操作。具体功能包括：

# 1. 从给定的文件路径（filepathX）读取CSV文件中的数据。
# 2. 对读取的数据进行处理，提取出所有行的数据。
# 3. 遍历数据中的每一行，将每行数据作为图像路径，调用私有方法__ImageMaskTranform对图像进行处理。
# 4. __ImageMaskTranform方法用于对图像进行转换操作，其中包括对图像进行蒙版处理。

# 如果给定的aug_path参数不为空，则将aug_path赋值给类属性self.aug_path。
