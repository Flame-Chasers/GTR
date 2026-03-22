'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import datetime
import os
import random
import time
from pathlib import Path

import numpy as np
import ruamel_yaml as yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import utils
from data import create_dataset, create_single_loader
from models.blip_ps import blip_ps


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Filtering:'

    print('Computing features for evaluation...')
    start_time = time.time()
    num_text = len(data_loader.dataset)

    itc_score = torch.zeros(num_text).to(device)
    itm_score = torch.zeros(num_text).to(device)
    for image, text, index in metric_logger.log_every(data_loader, 50, header):
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_atts = torch.ones(image_feat.size()[:-1], dtype=torch.long).to(device)

        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=config['max_words'], return_tensors="pt").to(device)

        output = model.text_encoder(text_input.input_ids,
                                    attention_mask=text_input.attention_mask,
                                    encoder_hidden_states=image_feat,
                                    encoder_attention_mask=image_atts,
                                    return_dict=True,
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        itm_score[index] = score

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(itc_score, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(itm_score, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return itc_score.cpu(), itm_score.cpu()


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    print()
    print(args)
    print(config)

    #### Dataset #### 
    print("Creating filter dataset")
    dataset = create_dataset('ps_filter', config)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    data_loader = create_single_loader(dataset, sampler, bs=config['batch_size'], n_worker=4, is_train=False, collate_fn=None)

    #### Model #### 
    model = blip_ps(image_size=config['image_size'], vit=config['vit'], max_length=config['max_words'])

    checkpoint = torch.load(config['checkpoint'], map_location='cpu')
    state_dict = checkpoint['model']

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % config['checkpoint'])
    print(msg)

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start filtering")
    start_time = time.time()

    itc_score, itm_score = evaluate(model_without_ddp, data_loader, device, config)
    if utils.is_main_process():
        dataset.filter(itc_score, itm_score, config['filter_file'],
                       strategy=config['filter_strategy'], threshold=config['filter_threshold'])

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_coco.yaml')
    parser.add_argument('--output_dir', default='output/Caption_coco')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
