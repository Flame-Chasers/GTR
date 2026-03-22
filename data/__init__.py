import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval, coco_karpathy_retrieval_eval
from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
from data.nlvr_dataset import nlvr_dataset
from data.nocaps_dataset import nocaps_eval
from data.pretrain_dataset import pretrain_dataset
from data.ps_caption_dataset import ps_caption_train_dataset, ps_caption_eval_dataset
from data.ps_dataset import ps_train_dataset, ps_eval_dataset
# from data.ps_dataset_mask import ImageTextMLMDataset2, ps_train_dataset, ps_eval_dataset, ImageTextMLMDataset
from data.ps_pretrain_dataset import ps_pretrain_dataset
from data.vqa_dataset import vqa_dataset
from data.ps_filter_dataset import ps_filter_dataset


def create_dataset(dataset, config, min_scale=0.5, get_utils=False):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    # transform_train = transforms.Compose([
    #         transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
    #         transforms.RandomHorizontalFlip(),
    #         RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
    #                                           'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # transform_test = transforms.Compose([
    #     transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
    #     transforms.ToTensor(),
    #     normalize,
    #     ])

    transform_train = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if get_utils:
        return transform_train
    
    if dataset == 'pretrain':
        dataset = pretrain_dataset(config['train_file'], config['laion_path'], transform_train)
        return dataset

    elif dataset == 'ps':
        train_dataset = ps_train_dataset(config['train_file'], transform_train, config['image_root'], config['max_words'])
        val_dataset = ps_eval_dataset(config['val_file'], transform_test, config['image_root'], config['max_words'])
        test_dataset = ps_eval_dataset(config['test_file'], transform_test, config['image_root'], config['max_words'])
        print('len: ', train_dataset.__len__())
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'mask':
        train_dataset = ImageTextMLMDataset(config['train_file'], transform_train, config['image_root'], config)
        val_dataset = ps_eval_dataset(config['val_file'], transform_test, config['image_root'], config['max_words'])
        test_dataset = ps_eval_dataset(config['test_file'], transform_test, config['image_root'], config['max_words'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'mask2':
        train_dataset = ImageTextMLMDataset2(config['train_file'], transform_train, config['image_root'], config)
        val_dataset = ps_eval_dataset(config['val_file'], transform_test, config['image_root'], config['max_words'])
        test_dataset = ps_eval_dataset(config['test_file'], transform_test, config['image_root'], config['max_words'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'ps_caption':
        train_dataset = ps_caption_train_dataset(config['train_file'], transform_train, config['train_image_root'],
                                                 config['max_words'], prompt=config['prompt'])
        val_dataset = ps_caption_eval_dataset(config['val_file'], transform_test, config['val_image_root'], config['max_words'])
        test_dataset = ps_caption_eval_dataset(config['test_file'], transform_test, config['test_image_root'], config['max_words'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'ps_filter':
        dataset = ps_filter_dataset(config['ann_file'], transform_test, config['image_root'], config['max_words'])
        return dataset

    elif dataset == 'ps_pretrain':
        train_dataset = ps_pretrain_dataset(config['train_file'], transform_train, config['train_image_root'], config['max_words'],
                                            config['ra_probability'], config['id_agnostic'], config['select_index_file'])
        val_dataset = ps_eval_dataset(config['val_file'], transform_test, config['val_image_root'], config['max_words'])
        test_dataset = ps_eval_dataset(config['test_file'], transform_test, config['test_image_root'], config['max_words'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'caption_coco':
        train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'], prompt=config['prompt'])
        val_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'nocaps':
        val_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        return val_dataset, test_dataset

    elif dataset == 'retrieval_coco':
        train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'])
        val_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'retrieval_flickr':
        train_dataset = flickr30k_train(transform_train, config['image_root'], config['ann_root'])
        val_dataset = flickr30k_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr30k_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'vqa':
        train_dataset = vqa_dataset(transform_train, config['ann_root'], config['vqa_root'], config['vg_root'],
                                    train_files=config['train_files'], split='train')
        test_dataset = vqa_dataset(transform_test, config['ann_root'], config['vqa_root'], config['vg_root'], split='test')
        return train_dataset, test_dataset

    elif dataset == 'nlvr':
        train_dataset = nlvr_dataset(transform_train, config['image_root'], config['ann_root'], 'train')
        val_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'], 'test')
        return train_dataset, val_dataset, test_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle, drop_last=True)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        loader = create_single_loader(dataset, sampler, bs, n_worker, is_train, collate_fn)
        loaders.append(loader)
    return loaders


def create_single_loader(dataset, sampler, bs, n_worker, is_train, collate_fn):
    if is_train:
        shuffle = (sampler is None)
        drop_last = True
    else:
        shuffle = False
        drop_last = False
    loader = DataLoader(
        dataset,
        batch_size=bs,
        num_workers=n_worker,
        pin_memory=True,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    return loader
