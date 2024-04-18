import numpy as np
from torch.utils.data import Dataset
import torch
from typing import Tuple
# from typing import (
#     Generic,
#     Iterable,
#     Iterator,
#     List,
#     Optional,
#     Sequence,
#     Tuple,
#     TypeVar,
# )


# T_co = TypeVar('T_co', covariant=True)
# T = TypeVar('T')

# class Dataset(Generic[T_co]):

#     def __getitem__(self, index) -> T_co:
#         raise NotImplementedError

#     def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
#         return ConcatDataset([self, other])
# 泛型类定义

# define datasetModelSegwithnpy class wiht npy
class datasetModelSegwithnpy(Dataset):
    def __init__(self, images, labels, targetsize=(16, 64, 128, 128)):
        super(datasetModelSegwithnpy).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize
        
        self.labels1 = [0, 38, 52, 82, 88, 164, 205, 244]
        self.labels2 = [0, 205, 420, 500, 550, 600, 820, 850]
        self.idx_map = {number: index for index, number in enumerate(self.labels1)}
        self.idx_map_ = {number: index for index, number in enumerate(self.labels2)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        image = np.load(imagepath)
        D, H, W = np.shape(image)[0], np.shape(image)[1], np.shape(image)[2]
        image = np.reshape(image, (D, H, W, 1))
        image = np.transpose(image, (3, 0, 1, 2))

        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2] and image.shape[3] == self.targetsize[3]

        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor

        labelpath = self.labels[index]
        label = np.load(labelpath)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(label)[0], np.shape(label)[1], np.shape(label)[2]
        label = np.reshape(label, (D, H, W))
        label = label.astype(np.int64)
        if label.max() > 7:
            label_tensor = torch.as_tensor(label, dtype=torch.long)
            label_tensor = self._3D_data_relabel(label_tensor)
            label_np = label_tensor.numpy()
            np.save(labelpath, label_np)
        else:
            # print("np.unique :" , np.unique(label))
            label_tensor = torch.as_tensor(label, dtype=torch.long)
        return {'image': images_tensor, 'label': label_tensor}

    def _3D_data_relabel(self, original_tensor):
        # Check if the unique values in original_tensor are contained in labels
        if original_tensor.unique().tolist() == self.labels1:
            # Use torch's indexing functionality to replace labels
            replaced_tensor = torch.tensor([self.idx_map[num.item()] for num in original_tensor.view(-1)]).view(original_tensor.shape).to(torch.long)
            return replaced_tensor
        elif original_tensor.unique().tolist() == self.labels2:
            replaced_tensor = torch.tensor([self.idx_map_[num.item()] for num in original_tensor.view(-1)]).view(original_tensor.shape).to(torch.long)
            return replaced_tensor
        else:
            print("error: The unique values in the original_tensor do not match the expected labels.")
            # If an error occurs, you can return None or raise an exception
            return original_tensor  # or raise Exception("error message")



# define datasetModelSegwithopencv class with npy
class datasetModelSegwithopencv(Dataset):
    def __init__(self, images, labels, targetsize=(1, 512, 512)):
        super(datasetModelSegwithopencv).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        # load image
        image = cv2.imread(imagepath, 0)
        # resize image to fixed size
        image = cv2.resize(image, (self.targetsize[1], self.targetsize[2]))
        # normalization image to zscore
        image = (image - image.mean()) / image.std()
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(image)[0], np.shape(image)[1]
        image = np.reshape(image, (H, W, 1))
        image = np.transpose(image, (2, 0, 1))
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2]
        # convert numpy to tensor
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor
        # torch.set_printoptions(8)
        labelpath = self.labels[index]
        label = cv2.imread(labelpath, 0)
        label = cv2.resize(label, (self.targetsize[1], self.targetsize[2]))
        # transpose (H,W,C) order to (C,H,W) order
        label = np.reshape(label, (H, W))
        label_tensor = torch.as_tensor(label).long()
        return {'image': images_tensor, 'label': label_tensor}


# define datasetModelRegressionwithopencv class with npy
class datasetModelRegressionwithopencv(Dataset):
    def __init__(self, images, labels, targetsize=(1, 512, 512)):
        super(datasetModelRegressionwithopencv).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        # load image
        image = cv2.imread(imagepath, 0)
        # resize image to fixed size
        image = cv2.resize(image, (self.targetsize[1], self.targetsize[2]), interpolation=cv2.INTER_LINEAR)
        # normalization image to zscore
        mean = image.mean()
        std = image.std()
        eps = 1e-5
        image = (image - mean) / (std + eps)
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(image)[0], np.shape(image)[1]
        image = np.reshape(image, (H, W, 1))
        image = np.transpose(image, (2, 0, 1))
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2]
        # convert numpy to tensor
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor
        # torch.set_printoptions(8)
        labelpath = self.labels[index]
        label = cv2.imread(labelpath, 0)
        label = cv2.resize(label, (self.targetsize[1], self.targetsize[2]), interpolation=cv2.INTER_LINEAR)
        # transpose (H,W,C) order to (C,H,W) order
        label = np.reshape(label, (H, W))
        label = (label - mean) / (std + eps)
        label_tensor = torch.as_tensor(label).float()

        mean_tensor = torch.as_tensor(mean).float()
        std_tensor = torch.as_tensor(std + eps).float()
        return {'image': images_tensor, 'label': label_tensor, 'mean': mean_tensor, 'std': std_tensor}
