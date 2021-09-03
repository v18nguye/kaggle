import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import GwaveDataset, get_transforms
from nets import G2NetModel

sys.path.append('../utils/')
from cfg import CFG
from seeds import seed_torch
from logging import init_logger
from tools import train_fn, valid_fn
from metrics import get_auc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##############################################
#   SETTING
##############################################

OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

cfg = CFG()
seed_torch(cfg.seed)
LOGGER = init_logger()

##############################################
#   DATASET
##############################################
DATA_PATH = '..'

train = pd.read_csv(DATA_PATH+'/input/g2net-gravitational-wave-detection/training_labels.csv')
test = pd.read_csv(DATA_PATH+'/input/g2net-gravitational-wave-detection/sample_submission.csv')

def get_train_file_path(image_id):
    return DATA_PATH+"/input/g2net-gravitational-wave-detection/train/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)

def get_test_file_path(image_id):
    return DATA_PATH+"/input/g2net-gravitational-wave-detection/test/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)

train['file_path'] = train['id'].apply(get_train_file_path)
test['file_path'] = test['id'].apply(get_test_file_path)

train_detected_waves = train[train['target'] == 1].reset_index(drop=True)
train_none_waves = train[train['target'] == 0].reset_index(drop=True)
train_part = pd.concat([train_detected_waves, train_none_waves]).reset_index(drop=True)
print(len(train_detected_waves), len(train_none_waves), len(train_part))

## Cross validation

Fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=cfg.seed)
for n, (train_index, val_index) in enumerate(Fold.split(train_part, train_part[cfg.target_col])):
    train_part.loc[val_index, 'fold2'] = int(n)
    
train_part['fold'] = [0]*len(train_part)
display(train_part.groupby(['fold', 'target']).size())
display(train_part.groupby(['fold2', 'target']).size())

##############################################
#   TRAIN LOOPS
##############################################
def train_loop(folds, fold, cfg):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] == fold].index
    val_idx = folds[folds['fold2'] == 1].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds[cfg.target_col].values

    train_dataset = GwaveDataset(train_folds, transform=get_transforms(data='train'))
    valid_dataset = GwaveDataset(valid_folds, transform=get_transforms(data='train'))

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size, 
                              shuffle=True, 
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=cfg.batch_size * 2, 
                              shuffle=False, 
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        if cfg.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = G2NetModel(cfg, pretrained=True)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss()

    best_score = 0.
    best_loss = np.inf
    
    for epoch in range(cfg.epochs):
        
        start_time = time.time()
        
        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg)
        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device, cfg)
        
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        # scoring
        score = get_auc(valid_labels, preds)
        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')

        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        OUTPUT_DIR+f'{cfg.model_name}_fold{fold}_best_score.pth')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        OUTPUT_DIR+f'{cfg.model_name}_fold{fold}_best_loss.pth')
    
    valid_folds['preds'] = torch.load(OUTPUT_DIR+f'{cfg.model_name}_fold{fold}_best_score.pth', 
                                      map_location=torch.device('cpu'))['preds']

    return valid_folds

def main():

    """
    Prepare: 1.train 
    """

    def get_result(result_df):
        preds = result_df['preds'].values
        labels = result_df[cfg.target_col].values
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}')
    
    if CFG.train:
        # train 
        oof_df = pd.DataFrame()
        for fold in range(cfg.n_fold):
            if fold in cfg.trn_fold:
                _oof_df = train_loop(train_part, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)


if __name__ == '__main__':
    main()