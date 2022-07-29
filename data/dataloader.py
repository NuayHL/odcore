import os
from torch.utils.data import DataLoader, distributed
from .dataset import CocoDataset

def build_dataloader(anns_path, img_path, config_data, batch_size, rank, workers):
    dataset = CocoDataset(anns_path, img_path, config_data)
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset)
    workers = min(os.cpu_count() // int(os.getenv("WORLD_SIZE", 1)),
                  batch_size if batch_size > 1 else 0,
                  workers)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=workers,
                        collate_fn=dataset.OD_default_collater,
                        sampler=sampler)
    return loader
