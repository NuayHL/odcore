from torch.utils.data import DataLoader

from .dataset import CocoDataset

def build_dataloader(anns_path, img_path, config):
    dataset = CocoDataset(anns_path, img_path, config.data)
    loader = DataLoader(dataset,
                        batch_size=8,
                        num_workers=8,
                        collate_fn=dataset.OD_default_collater)
    return loader
