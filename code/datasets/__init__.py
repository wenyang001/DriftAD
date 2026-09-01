from header import *
from .samplers import DistributedBatchSampler
from .mvtec import *
from .visa import VisaDataset

def load_mvtec_dataset(args):
    data = MVtecDataset(os.environ.get('MVTEC_ROOT', '../data/mvtec'))

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=4,
        collate_fn=data.collate, 
        pin_memory=False
    )
    return data, iter_, sampler


def load_visa_dataset(args):
    data = VisaDataset(os.environ.get('VISA_ROOT', '../data/visa'))

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=4,
        collate_fn=data.collate, 
        pin_memory=False
    )
    return data, iter_, sampler
