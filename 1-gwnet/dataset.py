import numpy as np
from torchvision import transforms 
from gwpy import timeseries as tser
from torch.utils.data import Dataset

def pre_proces(data, cut_off_t=0.05, timming = False):
    """Apply whitening process
    
    :param data: ndarray (3xN)
        read data in the form of numpy.
    :param cut_off_t: ndarray
        cut the data prediod off at the beginning and 
        ending of the signal.
    """
    prp_data_list = []
    if timming:
        start =time.time()
    for i in range(3):
        # convert to a TimeSeries object
        wave = tser.TimeSeries(data[i,:], epoch=0, sample_rate=2048)
        # data whitening
        wave = wave.whiten(0.125, 0.125/2, fduration=0.125/2, method = 'median', window='hann')
        # time cut-off
        wave = wave.crop(*wave.span.contract(cut_off_t))
        prp_data_list.append(wave.value.tolist())
    pre_data = np.array(prp_data_list)
    if timming:
        print(time.time() - start)
    return pre_data


def get_transforms(data):
    
    if data == 'train':
        return transforms.Compose([
            transforms.ToTensor(),
        ])

    elif data == 'valid':
        return A.Compose([
            transforms.ToTensor(),
        ])

class GwaveDataset(Dataset):
    def __init__(self, df, cfg, transform=get_transforms('train')):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df[cfg.target_col].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        data = np.load(file_path)
        prp_data = pre_proces(data, timming= False)
        
        if self.transform:
            prp_data = self.transform(prp_data)
            prp_data = prp_data.type(torch.float32).squeeze()
        label = torch.tensor(self.labels[idx]).float()
        return prp_data, label