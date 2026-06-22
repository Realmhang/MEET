import math
from typing import Callable, Union, Optional
from tqdm import tqdm

import numpy as np
import torch
import sympy

from utils import register as R

from .resample import IdentityMapping


@R.register('MixDatasetWrapper')
class MixDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, datasets, collate_fn: Callable=None, weights=None) -> None:
        super().__init__()
        self.datasets = [R.recur_construct(dataset) for dataset in datasets]
        self.cum_len = []
        self.total_len = 0
        for dataset in self.datasets:
            self.total_len += len(dataset)
            self.cum_len.append(self.total_len)
        self.collate_fn = self.datasets[0].collate_fn if collate_fn is None else collate_fn
        if weights is not None: assert len(weights) == len(datasets)
        self.weights = weights
        self.dynamic_idx = None
        self.update_epoch()

    def _get_dataset_and_idx(self, idx: int):
        assert idx < self.total_len
        last_cum_len = 0
        for i, cum_len in enumerate(self.cum_len):
            if idx < cum_len:
                return self.datasets[i], idx - last_cum_len
            last_cum_len = cum_len
        return None, None  # this is not possible

    def update_epoch(self):
        for dataset in self.datasets:
            if hasattr(dataset, 'update_epoch'):
                dataset.update_epoch()
        if self.weights is None:
            self.dynamic_idx = IdentityMapping(self.total_len)
        else:
            self.dynamic_idx = []
            start_idx = 0
            for i, (w, dataset) in enumerate(zip(self.weights, self.datasets)):
                add_len, end_idx = int(len(dataset) * w), self.cum_len[i]
                self.dynamic_idx.extend(np.random.choice(
                    list(range(start_idx, end_idx)),
                    size=add_len, replace=True # maybe weight > 1.0
                ))
                start_idx = end_idx

    def get_len(self, idx):
        idx = self.dynamic_idx[idx]
        dataset, idx = self._get_dataset_and_idx(idx)
        return dataset.get_len(idx)

    def __len__(self):
        return len(self.dynamic_idx)
    
    def __getitem__(self, idx):
        idx = self.dynamic_idx[idx]
        dataset, idx = self._get_dataset_and_idx(idx)
        return dataset[idx]


@R.register('ReplaceCacheWrapper')
class ReplaceCacheWrapper(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset: dict,
            cache_num: Union[int, float],
            replace_rate: float,
            # pre_shuffled_idx: Optional[str] = None,
            pre_shuffled_idx: bool = False,
        ) -> None:
        super().__init__()
        self.dataset = R.recur_construct(dataset)
        self.cache_num = cache_num
        if isinstance(self.cache_num, float):
            assert (0.0 < self.cache_num) and (self.cache_num <= 1.0), 'If cache_num is float, it should be between 0 and 1.'
            self.cache_num = math.ceil(len(self.dataset) * self.cache_num)
        elif isinstance(self.cache_num, int):
            assert (0 < self.cache_num) and (self.cache_num <= len(self.dataset)), f'cache_num should be in (0, total_length: {len(self.dataset)}]'
        assert (0 < cache_num) and (cache_num <= len(self.dataset)), f'cache_num (currently {cache_num}) should be in (0, {len(self.dataset)}]'
        self.replace_rate = replace_rate
        assert (0 < replace_rate) and (replace_rate <= 1.0)
        self._replace_num = math.ceil(self.cache_num * self.replace_rate)
        self.collate_fn = self.dataset.collate_fn

        # self.random_sample = pre_shuffled_idx is None
        self.random_sample = not pre_shuffled_idx
        if self.random_sample:
            self._cached_indexes = np.random.randint(0, len(self.dataset), size=self.cache_num).tolist()
        else:
            # self._pre_shuffled_idx = np.load(pre_shuffled_idx, mmap_mode='r')
            self._pre_shuffled_idx = np.arange(len(self.dataset))
            np.random.shuffle(self._pre_shuffled_idx)
            self._cached_indexes = self._pre_shuffled_idx[np.arange(self.cache_num) % len(self.dataset)].tolist()
            self._end_index = self.cache_num % len(self.dataset)
        self._looped_data_cnt = len(self._cached_indexes)
    
    def update_epoch(self):
        self._looped_data_cnt += self._replace_num
        # if already looped all the data once, call the update_epoch for the original dataset
        if self._looped_data_cnt > len(self.dataset):
            self._looped_data_cnt = 0
            if hasattr(self.dataset, 'update_epoch'):
                self.dataset.update_epoch()
            if not self.random_sample:
                np.random.shuffle(self._pre_shuffled_idx)
        del self._cached_indexes[:self._replace_num]
        if self.random_sample:
            self._cached_indexes.extend(np.random.randint(0, len(self.dataset), size=self._replace_num).tolist())
        else:
            self._cached_indexes.extend(self._pre_shuffled_idx[np.arange(self._end_index, self._end_index + self._replace_num) % len(self.dataset)].tolist())
            self._end_index = (self._end_index + self._replace_num) % len(self.dataset)

    
    def __len__(self):
        return len(self._cached_indexes)
    
    def __getitem__(self, idx):
        return self.dataset[self._cached_indexes[idx]]
        

class StandardIterator:
    def __init__(self, indexes):
        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, i):
        return self.indexes[i]
    
    def prefecth(self, i):
        return self.__getitem__(i)

    def done_batch(self):
        pass
    

class PackIterator(StandardIterator):
    def __init__(self, indexes, lengths):
        super().__init__(indexes)
        
        self.ordered_indexes = sorted(self.indexes, key=lambda i: lengths[i], reverse=True)
        self.idx_to_sorted = { i: sorted_i for sorted_i, i in enumerate(self.ordered_indexes) }

        # for recording (dynamically change during iteration)
        self.not_visited = { idx: True for idx in self.indexes }
        self.last_visited = None
        self.within_batch = False

        # TODO: prefetch, and local batch bias

    def done_batch(self):
        self.with_in_batch = False

    def __getitem__(self, i, prefetch=False):
        if self.within_batch:
            rank = self.idx_to_sorted[self.last_visited]
            offset = 1
            while True:
                left, right = rank - offset, rank + offset
                idx = None
                if left >=0 and self.ordered_indexes[left] in self.not_visited:
                    idx = self.ordered_indexes[left]
                elif right < len(self.ordered_indexes) and self.ordered_indexes[right] in self.not_visited:
                    idx = self.ordered_indexes[right]
                offset += 1
                if idx is not None: break
        else: # start a new batch
            assert len(self.not_visited)
            for idx in self.not_visited:
                break
            if not prefetch:
                self.within_batch = True
        if not prefetch:
            self.last_visited = idx
            self.not_visited.pop(idx)
        return idx
    
    def prefecth(self, i):
        return self.__getitem__(i, prefetch=True)


@R.register('DynamicBatchWrapper')
class DynamicBatchWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, complexity, ubound_per_batch, n_use_max_in_batch=False, max_batch_size=None, pack_similar_len=False, silent=False) -> None:
        super().__init__()
        self.dataset = R.recur_construct(dataset)
        self.indexes = [i for i in range(len(self.dataset))]
        self.complexity = complexity
        self.eval_func = sympy.lambdify('n', sympy.simplify(complexity))
        self.ubound_per_batch = ubound_per_batch
        self.n_use_max_in_dataset = n_use_max_in_batch
        self.max_batch_size = max_batch_size
        if self.max_batch_size is None: self.max_batch_size = 1e10 # a very large value
        self.pack_similar_len = pack_similar_len
        if self.pack_similar_len: # put items with similar lengths together
            assert n_use_max_in_batch, 'Pack_similar_len enabled, but not in the mode n_use_max_in_batch' # otherwise the packing algorithm is not necessary
        self.silent = silent
        self.total_size = None
        self.batch_indexes = []
        self._form_batch()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        else:
            raise AttributeError(f"'DynamicBatchWrapper'(or '{type(self.dataset)}') object has no attribute '{attr}'")

    def update_epoch(self):
        if hasattr(self.dataset, 'update_epoch'):
            self.dataset.update_epoch()
        self._form_batch()

    ########## overload with your criterion ##########
    def _form_batch(self):

        np.random.shuffle(self.indexes)
        last_batch_indexes = self.batch_indexes
        self.batch_indexes = []

        cur_complexity = 0
        batch = []

        # if self.pack_similar_len:
        #     iterator = PackIterator(self.indexes, [self.dataset.get_len(i) for i in self.indexes])
        # else:
        #     iterator = StandardIterator(self.indexes)
        if self.pack_similar_len:
            batch_max_n = 0
            iterator = PackIterator(self.indexes, [self.dataset.get_len(i) for i in self.indexes])
            pbar = range(len(iterator))
            if not self.silent: pbar = tqdm(pbar, ascii=True)
            for idx in pbar:
                i = iterator.prefecth(idx)
                n = self.dataset.get_len(i)
                if self.eval_func(n) > self.ubound_per_batch:
                    i = iterator[idx] # record visited
                    continue
                batch_max_n = max(batch_max_n, n)
                cur_complexity = self.eval_func(batch_max_n) * len(batch)
                if (cur_complexity > self.ubound_per_batch) or (len(batch) >= self.max_batch_size):
                    self.batch_indexes.append(batch)
                    iterator.done_batch()
                    i = iterator[idx] # get a new one for a new batch
                    n = self.dataset.get_len(i)
                    batch = []
                    batch_max_n = n # for next batch
                batch.append(i)
            self.batch_indexes.append(batch)    # last batch

        elif self.n_use_max_in_dataset:
            batch_max_n = 0
            pbar = self.indexes
            if not self.silent: pbar = tqdm(pbar, ascii=True)
            for i in pbar:
                n = self.dataset.get_len(i)
                if self.eval_func(n) > self.ubound_per_batch:
                    continue
                batch_max_n = max(batch_max_n, n)
                cur_complexity = self.eval_func(batch_max_n) * len(batch)
                if (cur_complexity > self.ubound_per_batch) or (len(batch) >= self.max_batch_size):
                    self.batch_indexes.append(batch)
                    batch = []
                    batch_max_n = n # for next batch
                batch.append(i)
            self.batch_indexes.append(batch)    # last batch
        else:
            pbar = self.indexes
            if not self.silent: pbar = tqdm(pbar, ascii=True)
            for i in pbar:
                item_len = self.eval_func(self.dataset.get_len(i))
                if item_len > self.ubound_per_batch:
                    continue
                cur_complexity += item_len
                if (cur_complexity > self.ubound_per_batch) or (len(batch) >= self.max_batch_size):
                    self.batch_indexes.append(batch)
                    batch = []
                    cur_complexity = item_len
                batch.append(i)
            self.batch_indexes.append(batch)    # last batch

        if self.total_size is None:
            self.total_size = len(self.batch_indexes)
        else:
            # control the lengths of the dataset, otherwise the dataloader will raise error
            if len(self.batch_indexes) < self.total_size:
                num_add = self.total_size - len(self.batch_indexes)
                self.batch_indexes = self.batch_indexes + last_batch_indexes[:num_add]
            else:
                self.batch_indexes = self.batch_indexes[:self.total_size]

    def __len__(self):
        return len(self.batch_indexes)
    
    def __getitem__(self, idx):
        return [self.dataset[i] for i in self.batch_indexes[idx]]
    
    def collate_fn(self, batched_batch):
        batch = []
        for minibatch in batched_batch:
            batch.extend(minibatch)
        return self.dataset.collate_fn(batch)