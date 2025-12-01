import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class EvalDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank):
        seps = np.linspace(0, len(dataset), num_replicas+1, dtype=int)
        beg, end = seps[:-1], seps[1:]
        beg, end = beg[rank], end[rank]
        self.indices = tuple(range(beg, end))
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self) -> int:
        return len(self.indices)


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, seed_for_all_rank=0, fill_last=False, shuffle=True, drop_last=False, start_ep=0, start_it=0):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.iters_per_ep = dataset_len // batch_size if drop_last else (dataset_len + batch_size - 1) // batch_size
        self.max_p = self.iters_per_ep * batch_size
        self.fill_last = fill_last
        self.shuffle = shuffle
        self.epoch = start_ep
        self.same_seed_for_all_ranks = seed_for_all_rank
        self.indices = self.gener_indices()
        self.start_ep, self.start_it = start_ep, start_it
    
    def gener_indices(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.same_seed_for_all_ranks)
            indices = torch.randperm(self.dataset_len, generator=g).numpy()
        else:
            indices = torch.arange(self.dataset_len).numpy()
        
        tails = self.batch_size - (self.dataset_len % self.batch_size)
        if tails != self.batch_size and self.fill_last:
            tails = indices[:tails]
            np.random.shuffle(indices)
            indices = np.concatenate((indices, tails))
        
        # built-in list/tuple is faster than np.ndarray (when collating the data via a for-loop)
        # noinspection PyTypeChecker
        return tuple(indices.tolist())
    
    def __iter__(self):
        self.epoch = self.start_ep
        while True:
            self.epoch += 1
            p = (self.start_it * self.batch_size) if self.epoch == self.start_ep else 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            if self.shuffle:
                self.indices = self.gener_indices()
    
    def __len__(self):
        return self.iters_per_ep


class DistInfiniteBatchSampler(InfiniteBatchSampler):
    def __init__(self, world_size, rank, dataset_len, glb_batch_size, same_seed_for_all_ranks=0, repeated_aug=0, fill_last=False, shuffle=True, start_ep=0, start_it=0):
        assert glb_batch_size % world_size == 0
        self.world_size, self.rank = world_size, rank
        self.dataset_len = dataset_len
        self.glb_batch_size = glb_batch_size
        self.batch_size = glb_batch_size // world_size
        
        self.iters_per_ep = (dataset_len + glb_batch_size - 1) // glb_batch_size
        self.fill_last = fill_last
        self.shuffle = shuffle
        self.repeated_aug = repeated_aug
        self.epoch = start_ep
        self.same_seed_for_all_ranks = same_seed_for_all_ranks
        self.indices = self.gener_indices()
        self.start_ep, self.start_it = start_ep, start_it
    
    def gener_indices(self):
        global_max_p = self.iters_per_ep * self.glb_batch_size  # global_max_p % world_size must be 0 cuz glb_batch_size % world_size == 0
        # print(f'global_max_p = iters_per_ep({self.iters_per_ep}) * glb_batch_size({self.glb_batch_size}) = {global_max_p}')
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.same_seed_for_all_ranks)
            global_indices = torch.randperm(self.dataset_len, generator=g)
            if self.repeated_aug > 1:
                global_indices = global_indices[:(self.dataset_len + self.repeated_aug - 1) // self.repeated_aug].repeat_interleave(self.repeated_aug, dim=0)[:global_max_p]
        else:
            global_indices = torch.arange(self.dataset_len)
        filling = global_max_p - global_indices.shape[0]
        if filling > 0 and self.fill_last:
            global_indices = torch.cat((global_indices, global_indices[:filling]))
        # global_indices = tuple(global_indices.numpy().tolist())
        
        seps = torch.linspace(0, global_indices.shape[0], self.world_size + 1, dtype=torch.int)
        local_indices = global_indices[seps[self.rank].item():seps[self.rank + 1].item()].tolist()
        self.max_p = len(local_indices)
        return local_indices

def hierarchical_multinomial(weights, num_samples, replacement=False, chunk_size=1_000_000):
    device = weights.device
    num_categories = weights.size(0)
    # Step 1: Number of chunks
    num_chunks = (num_categories + chunk_size - 1) // chunk_size
    # Step 2: Compute sum of weights per chunk
    chunk_sums = torch.empty(num_chunks, device=device)
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, num_categories)
        chunk_sums[i] = weights[start:end].sum()
    # Step 3: Sample which chunks to pick from
    chosen_chunks = torch.multinomial(chunk_sums, num_chunks, replacement=True)
    # Step 4: Sample inside each chosen chunk
    samples = []
    local_num_samples = num_samples // num_chunks
    for i, chunk_idx in enumerate(chosen_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, num_categories)
        local_weights = weights[start:end]
        local_samples = torch.multinomial(local_weights, local_num_samples, replacement=replacement)
        samples.append(local_samples)
    samples = torch.cat(samples)
    return samples

class DistInfiniteWeightedBatchSampler(InfiniteBatchSampler):
    def __init__(self, world_size, rank, dataset_len, num_samples, weights, glb_batch_size, same_seed_for_all_ranks=0, repeated_aug=0, fill_last=False, start_ep=0, start_it=0, replacement=False):
        assert glb_batch_size % world_size == 0
        self.world_size, self.rank = world_size, rank
        self.dataset_len = dataset_len
        self.num_samples = num_samples
        self.glb_batch_size = glb_batch_size
        self.batch_size = glb_batch_size // world_size
        self.weights = weights
        self.iters_per_ep = (num_samples + glb_batch_size - 1) // glb_batch_size
        self.fill_last = fill_last
        self.shuffle = False
        self.repeated_aug = repeated_aug
        self.epoch = start_ep
        self.same_seed_for_all_ranks = same_seed_for_all_ranks
        self.replacement = replacement # must be init before calling gener_indices()
        self.indices = self.gener_indices()
        self.start_ep, self.start_it = start_ep, start_it
    
    def gener_indices(self):
        if self.weights.size(0) > 2**23:
            print("hierarchical_multinomial sampling")
            global_indices = hierarchical_multinomial(self.weights, self.num_samples, self.replacement)
        else:
            global_indices = torch.multinomial(self.weights, self.num_samples, self.replacement)
        local_indices = global_indices[self.rank:global_indices.shape[0]:self.world_size].tolist()
        self.max_p = len(local_indices)
        print('length of local indices: ', len(local_indices))
        return local_indices
    
    def reset_weights(self, new_weights):
        self.weights = new_weights
        self.indices = self.gener_indices()


