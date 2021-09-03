import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from nnAudio import Spectrogram

class SpectrogramModel(nn.Module):
    """Learning to represent a 2-D strain to a 3D spectrogram
    
    """
    def __init__(self):
        super(SpectrogramModel,self).__init__()
        self.cqt = Spectrogram.CQT2010v2(sr=2048, fmin=20, fmax=1028, hop_length=16,
                                        pad_mode='constant',
                                        n_bins = 320,
                                        bins_per_octave=64,
                                        norm=True, basis_norm=1, window='hann')
        
    def forward(self, x):
        batch_size, num_chanels = x.shape[0], x.shape[1]
        x = x.reshape(batch_size*num_chanels,-1)
        x = self.cqt(x)
        x = x.reshape(batch_size, num_chanels, x.shape[1], x.shape[2])
        return x
        
        
class G2NetModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.spectro = SpectrogramModel()
        self.model = timm.create_model(self.cfg.model_name, pretrained=pretrained, in_chans=3)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, self.cfg.target_size)
        
    def forward(self, x):
        x = self.spectro(x)
        x = self.model(x)
        return x