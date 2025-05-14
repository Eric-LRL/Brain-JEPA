import os
import glob
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from logging import getLogger
import pickle
from tqdm import tqdm

logger = getLogger()

class fMRIDataset(Dataset):
    def __init__(
        self, 
        split='train', 
        n_cortical_rois=400, 
        n_subcortical_rois=50,
        seq_length=490, 
        root_dir='/fill/this/with/root_path',
        params_file="normalization_params_train.npz",
        downsample=False,
        sampling_rate=3,
        num_frames=160,
        use_standatdization=False,
    ):
        self.use_standatdization = use_standatdization
        self.n_cortical_rois = n_cortical_rois
        self.n_subcortical_rois = n_subcortical_rois
        self.n_rois = n_cortical_rois + n_subcortical_rois
        self.seq_length = seq_length
        self.root_dir = root_dir
        self.ts_dir = os.path.join(self.root_dir, 'time_series')
        
        self.downsample = downsample
        self.sampling_rate = sampling_rate
        self.num_frames = num_frames
        
        save_param_root_path = self.root_dir  
        os.makedirs(save_param_root_path, exist_ok=True)
        self.params_file = os.path.join(save_param_root_path, params_file)
        
        # Load pretrain IDs
        with open('/fill/this/with/path/to/id_file.pkl', 'rb') as f:
            train_val_test_ids = pickle.load(f)

        self.ids = train_val_test_ids[f'{split}_ids']

        self.cortical_file = f'fMRI.Schaefer17n{n_cortical_rois}p.csv.gz'
        self.subcortical_file = f'fMRI.Tian_Subcortex_{self._map_subcortical(n_subcortical_rois)}_3T.csv.gz'
        
        # Load all time series data to compute normalization parameters
        self.normalization_params = self._load_or_compute_normalization_params()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        ts_cortical = self._load_ts(id, self.cortical_file)
        ts_subcortical = self._load_ts(id, self.subcortical_file)
        ts_array = np.concatenate((ts_subcortical, ts_cortical), axis=0).astype(np.float32)
        
        # Apply robust scaling
        if not self.use_standatdization:
            median, iqr = self.normalization_params['medians'], self.normalization_params['iqrs']
            ts_array = (ts_array - median[:, None]) / iqr[:, None]
        
        if self.downsample:
            clip_size = self.sampling_rate * self.num_frames
            start_idx, end_idx = self._get_start_end_idx(self.seq_length, clip_size)
            ts_array = self._temporal_sampling(
                        torch.from_numpy(ts_array), start_idx, end_idx, self.num_frames
                    )
            ts = torch.unsqueeze(ts_array, 0).to(torch.float32)
        else:
            ts = torch.unsqueeze(torch.from_numpy(ts_array), 0).to(torch.float32)
            
        if self.use_standatdization:
            mean = ts.mean()
            std = ts.std()
            ts = (ts - mean) / std

        return {'fmri': ts}
    
    def _get_start_end_idx(self, fmri_size, clip_size):
        "Reference: https://github.com/facebookresearch/mae_st"
        """
        Sample a clip of size clip_size from a video of size video_size and
        return the indices of the first and last frame of the clip. If clip_idx is
        -1, the clip is randomly sampled, otherwise uniformly split the video to
        num_clips clips, and select the start and end index of clip_idx-th video
        clip.
        Args:
            video_size (int): number of overall frames.
            clip_size (int): size of the clip to sample from the frames.
            clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
                clip_idx is larger than -1, uniformly split the video to num_clips
                clips, and select the start and end index of the clip_idx-th video
                clip.
            num_clips (int): overall number of clips to uniformly sample from the
                given video for testing.
        Returns:
            start_idx (int): the start frame index.
            end_idx (int): the end frame index.
        """
        delta = max(fmri_size - clip_size, 0)
        start_idx = random.uniform(0, delta)
        end_idx = start_idx + clip_size - 1
        return start_idx, end_idx
    
    def _temporal_sampling(self, frames, start_idx, end_idx, num_samples):
        "Reference: https://github.com/facebookresearch/mae_st"
        """
        Given the start and end frame index, sample num_samples frames between
        the start and end with equal interval.
        Args:
            frames (tensor): a tensor of video frames, dimension is
                `num video frames` x `channel` x `height` x `width`.
            start_idx (int): the index of the start frame.
            end_idx (int): the index of the end frame.
            num_samples (int): number of frames to sample.
        Returns:
            frames (tersor): a tensor of temporal sampled video frames, dimension is
                `num clip frames` x `channel` x `height` x `width`.
        """
        index = torch.linspace(start_idx, end_idx, num_samples)
        index = torch.clamp(index, 0, frames.shape[1] - 1).long()
        new_frames = torch.index_select(frames, 1, index)
        return new_frames

    def _map_subcortical(self, n_subcortical_rois):
        mapping = {16: 'S1', 32: 'S2', 50: 'S3', 54: 'S4'}
        return mapping[n_subcortical_rois]
    
    def _load_or_compute_normalization_params(self):
        if os.path.exists(self.params_file):
            params_df = np.load(self.params_file)
            medians = params_df['medians']
            iqrs = params_df['iqrs']
            print("Normalization parameters loaded from file.")
            return {'medians': medians, 'iqrs': iqrs}
        
        else:
            # Compute and save the robust scaling statistical parameters if not already done
            return self._compute_normalization_params()

    def _compute_normalization_params(self):
        all_data = []
        all_data_mean = []
        for id in tqdm(self.ids):
            ts_cortical = self._load_ts(id, self.cortical_file)
            ts_subcortical = self._load_ts(id, self.subcortical_file)
            data = np.concatenate((ts_subcortical, ts_cortical), axis=0)
            all_data.append(np.concatenate((ts_subcortical, ts_cortical), axis=0))
            
            temp_data = np.mean(data, axis=1)
            all_data_mean.append(temp_data)
            
        all_data = np.stack(all_data)
        all_data_mean = np.stack(all_data_mean)

        medians = np.median(all_data_mean, axis=0)
        iqrs = np.percentile(all_data_mean, 75, axis=0) - np.percentile(all_data_mean, 25, axis=0)
        np.savez(self.params_file, medians=medians, iqrs=iqrs)
        
        return {'medians': medians, 'iqrs': iqrs}

    def save_normalization_params(self, medians, iqrs, filename="normalization_params_train.csv"):
        df = pd.DataFrame({
            'roi_index': range(len(medians)),
            'median': medians,
            'iqr': iqrs
        })
        df.to_csv(self.params_file, index=False)
        print(f"Normalization parameters saved to {self.params_file}")

    def _load_csv(self, id, file):
        file_path = os.path.join(self.ts_dir, id, file)
        return pd.read_csv(file_path)

    def _load_ts(self, id, file):
        df = self._load_csv(id, file)
        return df.iloc[:, 1:self.seq_length+1].values
    
    def _load_roi_names(self):
        df_cortical = self._load_csv(self.ids[0], self.cortical_file)
        df_subcortical = self._load_csv(self.ids[0], self.subcortical_file)
        cortical_names = df_cortical['label_name'].values
        subcortical_names = df_subcortical['label_name'].values
        return np.concatenate((subcortical_names, cortical_names), axis=0)


def make_ukbiobank1k(
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    drop_last=True,
    downsample=False,
    use_standatdization=False,
    portion=1,
):
    dataset = fMRIDataset(
        downsample=downsample,
        use_standatdization=use_standatdization,
        portion=portion
    )
    logger.info('fMRI dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('fMRI unsupervised data loader created')

    return dataset, data_loader, dist_sampler


if __name__ == '__main__':
    print(f'loading data')
    
    dataset, unsupervised_loader, unsupervised_sampler = make_ukbiobank1k(
            batch_size=1,
            collator=None,
            pin_mem=True,
            num_workers=0,
            world_size=1,
            rank=0,
            drop_last=False,
            portion=1)
    dataset = fMRIDataset(downsample=True)
    
    print(f'len pretrained dataset: {len(dataset)}')
    print('ok with scaling')
