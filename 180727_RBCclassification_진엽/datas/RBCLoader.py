import torch
from torch.utils import data
import random
import numpy as np
from collections import Counter
import os
from imblearn.over_sampling import SMOTE
import copyreg

from scipy import io

# ignore skimage zoom warning
import warnings
warnings.filterwarnings("ignore", ".*output shape of zoom.*")

class RBCDataset(data.Dataset):

    def __init__(self, f_path, sampler=None, infer=False, transform=None, torch_type="float", augmentation_rate=0.3):

        data_pre = io.loadmat(f_path+'/data.mat')
        data_pre = data_pre[list(data_pre.keys())[3]]
        labels_pre = io.loadmat(f_path + '/labels.mat')
        labels_pre = labels_pre[list(labels_pre.keys())[3]]

        data_pre = np.swapaxes(data_pre, 0, 1)
        labels_pre = np.swapaxes(labels_pre, 0, 1)
        labels_pre_idx = []
        for iter1 in range(labels_pre.shape[0]):
            labels_pre_idx = np.concatenate(
                (labels_pre_idx, np.where(labels_pre[iter1, :] > 0)[0]), axis=0)

        # execute SMOTE if transform=='smote'
        if transform is 'smote':
            labels_pre_idx = labels_pre_idx.astype(int)
            sme = SMOTE(ratio='all', random_state=42)
            data, labels_idx = sme.fit_sample(data_pre, labels_pre_idx)
            print('Original dataset shape {}'.format(Counter(labels_pre_idx)))
            print('Resampled dataset shape {}'.format(Counter(labels_idx)))
            self.data = data.astype(float)
            self.labels = labels_idx.astype(int)
        else :
            self.data = data_pre
            self.labels = labels_pre_idx

        if len(self.data) == 0:
            raise ValueError("Check data path : %s"%(img_root))

        self.transform = [] if transform is None else transform
        self.torch_type = torch.float if torch_type == "float" else torch.half

        self.channel = 1

    def __getitem__(self, idx):
        if self.channel == 1:
            return self._1D_data(idx)
        else:
            raise ValueError("error message")

    def __len__(self):
        return len(self.data)

    def _1D_data(self, idx):
        # 2D ( 1 x H x W )
        input_np = self.data[idx,]
        target_np = np.array(self.labels[idx,])


        input_  = self._np2tensor(input_np,torch.float)
        target_ = self._np2tensor(target_np,torch.long)

        return input_, target_

    def _np2tensor(self, np, torch_type):
        tmp = torch.from_numpy(np)
        return tmp.to(dtype=torch_type)


def RBCLoader(f_path, batch_size,
                  transform=None, sampler='',
                  channel=1, torch_type="float", cpus=1, infer=False,
                  shuffle=True, drop_last=True
              ):
    """Return a DataLoader applying transforms, sampler
    
    Parameters:
        image_path : full data path
        batch_size : batch_size
        transform :
        sampler(str) : "weight" or the 
        channel : 
        torch_type : 
        cpus : 
        infer : 
        shuffle : 
        drop_last : 
    
    Returns:
        data.DataLoader
    """

    dataset = RBCDataset(f_path, infer=infer, transform=transform, torch_type=torch_type)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=cpus, drop_last=drop_last)


def reconstruct_torch_dtype(torch_dtype: str):
    # a dtype string is "torch.some_dtype"
    dtype = torch_dtype.split('.')[1]
    return getattr(torch, dtype)


def pickle_torch_dtype(torch_dtype: torch.dtype):
    return reconstruct_torch_dtype, (str(torch_dtype),)


if __name__ == "__main__":
    # Test Data Loader
    f_path  = 'C:/Users/SJY/PycharmProjects/180727_RBCclassification_진엽/data/Data_Normalized/train'

    testLoader=RBCLoader(f_path,10,transform='smote')
    int(testLoader.dataset.__getitem__(0)[0].size(0))

    # hack to make torch.dtype pickleable
    copyreg.pickle(torch.dtype, pickle_torch_dtype)
    for i,(input_, target_)in enumerate(testLoader):
        print(input_.shape, target_.shape)

