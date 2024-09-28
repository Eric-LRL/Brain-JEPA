# --------------------------------------------------------
# References:
# I-JEPA: https://github.com/facebookresearch/ijepa
# --------------------------------------------------------

import math

from multiprocessing import Value

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator_fmri(object):

    def __init__(
        self,
        input_size=(450, 160),
        patch_size=16,
        enc_mask_scale=(0.84, 0.84),
        pred_mask_R_scale=(0.45, 0.6),
        pred_mask_T_scale=(0.2, 0.4),
        pred_mask_T_roi_scale=(0.2, 0.6),
        pred_mask_R_roi_scale=(0.15, 0.3),
        min_keep=4,
        allow_overlap=False
    ):
        super(MaskCollator_fmri, self).__init__()
        assert len(input_size) == 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0], input_size[1] // patch_size
        
        self.enc_mask_scale = enc_mask_scale
        
        self.pred_mask_R_scale = pred_mask_R_scale
        self.pred_mask_T_scale = pred_mask_T_scale
        self.pred_mask_T_roi_scale = pred_mask_T_roi_scale
        self.pred_mask_R_roi_scale = pred_mask_R_roi_scale
        
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size_e_proi(self, generator, roi_scale, ts_scale):
        # -- Sample roi scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = roi_scale
        mask_roi_scale = min_s + _rand * (max_s - min_s)
        max_keep_roi = int(self.height * mask_roi_scale)
        # -- Sample ts scale
        _rand = torch.rand(1, generator=generator).item()
        min_ts, max_ts = ts_scale
        mask_ts_scale = min_ts + _rand * (max_ts - min_ts)
        max_keep_ts = int(self.width * mask_ts_scale)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(max_keep_roi))
        w = int(round(max_keep_ts))
        if (h >= self.height) or (w >= self.width):
            raise Exception('error')

        return (h, w)
    
    def _sample_block_size_p_ts(self, generator, roi_scale, ts_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample roi scale
        min_s, max_s = roi_scale
        mask_roi_scale = min_s + _rand * (max_s - min_s)
        max_keep_roi_1 = int(len(self.e_roi_mask) * mask_roi_scale)
        max_keep_roi_2 = int(len(self.e_roi_nonmask) * mask_roi_scale)
        
        # -- Sample ts scale
        _rand = torch.rand(1, generator=generator).item()
        min_ts, max_ts = ts_scale
        mask_ts_scale = min_ts + _rand * (max_ts - min_ts)

        max_keep_ts = int(self.width * mask_ts_scale)
        # -- Compute block height and width
        h_1 = int(round(max_keep_roi_1))
        h_2 = int(round(max_keep_roi_2))
        w = int(round(max_keep_ts))
        if (h_1 >= self.height) or (h_2 >= self.height) or (w >= self.width):
            raise Exception('error')

        return (h_1, h_2, w)
    
    def get_remain_indices(self, roi_mask):
        m = torch.ones(self.height, dtype=torch.bool)

        m[roi_mask] = False
        remaining_indices = torch.nonzero(m).squeeze()
        
        return remaining_indices

    def _sample_block_mask_e(self, b_size, acceptable_regions=None):
        roi_num, ts_m = b_size
        
        mask = torch.zeros((self.height, self.width), dtype=torch.int32)
        roi_mask = torch.randperm(self.height)[:roi_num]

        self.e_roi_nonmask = self.get_remain_indices(roi_mask)

        ts_mask = torch.randint(0, self.width - ts_m, (1,))
        
        self.e_ts_mask_left = ts_mask
        self.e_ts_mask_right = ts_mask + ts_m

        mask[roi_mask, self.e_ts_mask_left:self.e_ts_mask_right] = 1

        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[roi_mask, self.e_ts_mask_left:self.e_ts_mask_right] = 0
        # --
        return mask, mask_complement
    
    def _sample_block_mask_p_roi(self, b_size, acceptable_regions=None):
        roi_num, ts_m = b_size
        
        mask = torch.zeros((self.height, self.width), dtype=torch.int32)
        roi_mask = torch.randperm(self.height)[:roi_num]
        
        self.p_roi_nonmask = self.get_remain_indices(roi_mask)
        if self.e_roi_nonmask.dim() == 0:
            self.e_roi_nonmask = self.p_roi_nonmask
        else:
            self.e_roi_nonmask = torch.cat((self.e_roi_nonmask, self.p_roi_nonmask)).unique()
        self.e_roi_mask = self.get_remain_indices(self.e_roi_nonmask)
        
        assert self.e_ts_mask_right-self.e_ts_mask_left > ts_m
        ts_mask = torch.randint(0, self.e_ts_mask_right.item() - self.e_ts_mask_left.item() - ts_m, (1,))
        
        mask[roi_mask, self.e_ts_mask_left.item()+ts_mask:self.e_ts_mask_left.item()+ts_mask+ts_m] = 1
        mask = torch.nonzero(mask.flatten())
        
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[roi_mask, :] = 0
        # --
        return mask, mask_complement
    
    
    def _sample_block_mask_p_ts(self, b_size, side='left', acceptable_regions=None):
        # roi_num_1 for beta, roi_num_2 for gamma
        roi_num_1, roi_num_2, ts_m = b_size
        
        mask = torch.zeros((self.height, self.width), dtype=torch.int32)
        roi_mask_1 = self.e_roi_mask[torch.randperm(self.e_roi_mask.size(0))][:roi_num_1]
        roi_mask_2 = self.e_roi_nonmask[torch.randperm(self.e_roi_nonmask.size(0))][:roi_num_2]
        
        roi_mask = torch.cat([roi_mask_2, roi_mask_1])

        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
            
        ts_mask = torch.randperm(3)[:ts_m]
        if side == 'right':
            ts_mask = 7 + ts_mask
        
        if ts_m > 0:
            for i in range(len(ts_mask)):
                mask[roi_mask, ts_mask[i]] = 1

        mask = torch.nonzero(mask.flatten())
        
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        if ts_m > 0:
            mask_complement[:, ts_mask] = 0
        # --
        return mask, mask_complement
    
    def constrain_e_mask(self, mask, acceptable_regions=None):
        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = len(acceptable_regions)
            for k in range(N):
                mask *= acceptable_regions[k]
        
        tries = 0
        valid_mask = False
        
        while not valid_mask:
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            else:
                raise Exception('error')
            
            valid_mask = len(mask) > self.min_keep

            if not valid_mask:
                logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
                raise Exception('error')    
                
        return mask

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block
        # 2. sample alpha block
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        # alpha
        p_size_roi = self._sample_block_size_e_proi(
            generator=g,
            roi_scale=self.pred_mask_R_scale,
            ts_scale=self.pred_mask_T_roi_scale)
        # observation
        e_size = self._sample_block_size_e_proi(
            generator=g,
            roi_scale=self.enc_mask_scale,
            ts_scale=self.enc_mask_scale)

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        
        for _ in range(B):

            masks_p, masks_C = [], []
            # sample observation block
            mask_e, _ = self._sample_block_mask_e(e_size)
            # sample alpha block
            mask, mask_C = self._sample_block_mask_p_roi(p_size_roi)
            masks_p.append(mask)
            # Overlapped sampling
            mask_e = self.constrain_e_mask(mask_e, acceptable_regions=[mask_C])
            
            # sample beta and gamma block
            p_size_ts = self._sample_block_size_p_ts(
                generator=g,
                roi_scale=self.pred_mask_R_roi_scale,
                ts_scale=self.pred_mask_T_scale)
            # sample from the right side of the observation block for beta and gamma block
            mask, mask_C = self._sample_block_mask_p_ts(p_size_ts, side='right')
            masks_p.append(mask)
            masks_C.append(mask_C)
            # sample from the left side of the observation block for beta and gamma block
            p_size_ts = self._sample_block_size_p_ts(
                generator=g,
                roi_scale=self.pred_mask_R_roi_scale,
                ts_scale=self.pred_mask_T_scale)
            
            mask, mask_C = self._sample_block_mask_p_ts(p_size_ts, side='left')
            masks_p.append(mask)
            masks_C.append(mask_C)
            
            # Overlapped sampling
            mask_e = self.constrain_e_mask(mask_e, acceptable_regions=masks_C)
            mask_e = torch.nonzero(mask_e.flatten())
            mask_e = mask_e.squeeze()
            min_keep_enc = min(min_keep_enc, len(mask_e))

            mask_p_final = torch.cat(masks_p)
            min_keep_pred = min(min_keep_pred, len(mask_p_final))
            
            collated_masks_pred.append([mask_p_final])
            collated_masks_enc.append([mask_e])
        
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred