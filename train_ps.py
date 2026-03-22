'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import datetime
import json
import math
import os
import random
import time
from pathlib import Path
# from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
# from ruamel.yaml import YAML
import ruamel_yaml as yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

import utils
from data import create_dataset, create_sampler, create_loader
from models.blip_ps_gmm import blip_ps, GMMCache
# from models.blip_ps import blip_ps
from models.vit import interpolate_pos_embed
from utils import cosine_lr_schedule


def train_gmm(model, data_loader, optimizer, epoch, device, config, gmm_cache):
    # train
    print('train-------------------')
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")  
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    sim_pool = []

    for i, (image1, image2, caption, prob, person) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image1 = image1.to(device, non_blocking=True)
        image2 = image2.to(device, non_blocking=True)
        prob = prob.to(device, non_blocking=True)
        person = person.to(device, non_blocking=True)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1., i / len(data_loader))

        # ============ ori model =============
        # loss_ita, loss_itm = model(image1, image2, caption, prob, alpha=alpha, idx=person, confident=config['confident'])   
        # ============ GMM Model =============
        # progress = (epoch+1 / config['max_epoch'])  # Value from 0 to 1
        progress = 0.2  # Value from 0 to 1

        # loss_ita, loss_itm, sim_diag, attnweights = model(image1, image2, caption, prob, alpha, idx=person, confident=config['confident'], keepsim=True, progress=progress)
        loss_ita, loss_itm, sim_diag = model(image1, image2, caption, prob, alpha, idx=person, confident=config['confident'], keepsim=True, progress=progress)
        sim_pool.append(sim_diag)
        # ====================================
        

        loss = loss_ita + loss_itm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # if epoch > 0:
    all_sim = np.concatenate(sim_pool, axis=0)
    gmm_cache.fit(all_sim)
    sim_pool = []
    print("GMM fitted!!!")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def train(model, data_loader, optimizer, epoch, device, config):
    # train
    print('train-------------------')
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")  
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image1, image2, caption, prob, person) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image1 = image1.to(device, non_blocking=True)
        image2 = image2.to(device, non_blocking=True)
        prob = prob.to(device, non_blocking=True)
        person = person.to(device, non_blocking=True)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1., i / len(data_loader))

        loss_ita, loss_itm = model(image1, image2, caption, prob, alpha=alpha, idx=person, confident=config['confident'])   
        
        loss = loss_ita + loss_itm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def mSD_rank(similarity, matches, max_rank=10):
    similarity = similarity / 2 + 0.5 # Normalize similarity to [0, 1]
    indices = torch.argsort(similarity, dim=1, descending=True)
    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    postive_idx = (indices+1)*matches
    postive_idx[postive_idx>=1] = 1
    postive_similarity = similarity*postive_idx

    num_rel = matches.sum(1)
    postive_similarity_sum = postive_similarity.sum(1)
    negative_similarity_sum = similarity.sum(1)-postive_similarity_sum
    postive_similarity_average = postive_similarity_sum/num_rel
    negative_similarity_average = negative_similarity_sum/(similarity.shape[1]-num_rel)
    pn_ratio = (postive_similarity_average / negative_similarity_average).numpy()
    pn_ratio = torch.tensor([1 - math.exp(-x) for x in pn_ratio])
    similarity_cmc = torch.cumsum(similarity, dim=1)
    positive_similarity_cmc = torch.cumsum(postive_similarity, dim=1)
    
    sd_cmc = positive_similarity_cmc/similarity_cmc
    sd_cmc = sd_cmc * matches
    SD = (sd_cmc.sum(1) / num_rel) * pn_ratio
    mSD = SD.mean() * 100

    # tmp_cmc = matches.cumsum(1)
    # tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    # tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    # AP = tmp_cmc.sum(1) / num_rel
    # mAP = AP.mean() * 100
    # return all_cmc, mSD, mAP
    return mSD

@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()
    
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=config['max_words'], return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    # print(text_output.attentions)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_ids[:, 0] = model.tokenizer.enc_token_id

    image_feats = []
    image_embeds = []

    for image_path, image, img_id in data_loader:
        image = image.to(device)

        # image_feat, attnweights_i = model.visual_encoder(image)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = text_embeds @ image_embeds.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        topk_idx = topk_idx.to(image_feats.device)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start + i].repeat(config['k_test'], 1),
                                    attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    output_attentions=True
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim


    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    return score_matrix_t2i.cpu()


@torch.no_grad()
def itm_eval(scores_t2i, img2person, txt2person, eval_mAP):
    img2person = torch.tensor(img2person)
    txt2person = torch.tensor(txt2person)

    index = torch.argsort(scores_t2i, dim=-1, descending=True)
    pred_person = img2person[index]
    matches = (txt2person.view(-1, 1).eq(pred_person)).long()

    def acc_k(matches, k=1):
        matches_k = matches[:, :k].sum(dim=-1)
        matches_k = torch.sum((matches_k > 0))
        return 100.0 * matches_k / matches.size(0)

    # Compute metrics
    ir1 = acc_k(matches, k=1).item()
    ir5 = acc_k(matches, k=5).item()
    ir10 = acc_k(matches, k=10).item()
    ir_mean = (ir1 + ir5 + ir10) / 3

    eval_mAP=True

    if eval_mAP:
        real_num = matches.sum(dim=-1)
        tmp_cmc = matches.cumsum(dim=-1).float()
        order = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long)
        tmp_cmc /= order
        tmp_cmc *= matches
        AP = tmp_cmc.sum(dim=-1) / real_num
        mAP = AP.mean() * 100.0
        # mFR = matches.argmax(dim=-1).float().mean()
        # mSD = mSD_rank(torch.tensor(scores_t2i), matches)


        eval_result = {'r1': ir1,
                       'r5': ir5,
                       'r10': ir10,
                       'r_mean': ir_mean,
                       'mAP': mAP.item(),
                    #    "mFR": mFR.item(),
                    #    "mSD": mSD.item()
                       }
    else:
        eval_result = {'r1': ir1,
                       'r5': ir5,
                       'r10': ir10,
                       'r_mean': ir_mean,
                       'mAP': mAP.item(),
                       }

    return eval_result


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
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('ps', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']] + [config['batch_size_test']] * 2,
                                                          num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])

    #### Model #### 
    print("Creating model")
    model = blip_ps(image_size=config['image_size'], vit=config['vit'], max_length=config['max_words'],
                    vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                    queue_size=config['queue_size'])
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    start_epoch = 0
    best = 0
    best_epoch = 0

    checkpoint = torch.load(config['checkpoint'], map_location='cpu')
    state_dict = checkpoint['model']
    
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best = checkpoint['best']
        best_epoch = checkpoint['best_epoch']
    else:
        state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)

    # don't load parameters of queue.
    for k in list(state_dict.keys()):
        if 'queue' in k:
            state_dict.pop(k)

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % config['checkpoint'])
    print(f'missing_keys: {msg.missing_keys}')

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    try:
        gmm_cache = GMMCache(n_components=2)
        model.module.set_gmm_cache(gmm_cache)
    except:
        pass

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train_gmm(model, train_loader, optimizer, epoch, device, config, gmm_cache)


        if epoch >= config['eval_epoch'] or args.evaluate:
            score_test_t2i = evaluation(model_without_ddp, test_loader, device, config)

            if utils.is_main_process():

                test_result = itm_eval(score_test_t2i, test_dataset.img2person, test_dataset.txt2person, args.eval_mAP)
                print('Test:', test_result, '\n')
                
                if args.evaluate:
                    log_stats = {
                        **{f'test_{k}': v for k, v in test_result.items()},
                                 }
                    with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                else:
                    checkpoint_best = False
                    if test_result['r1'] > best:
                        best = test_result['r1']
                        best_epoch = epoch
                        checkpoint_best = True

                    log_stats = {'time': time.strftime("%Y-%m-%d", time.localtime()),
                                 'epoch': epoch,
                                 'best_epoch': best_epoch,
                                 **{f'test_{k}': v for k, v in test_result.items()},
                                 **{f'train_{k}': v for k, v in train_stats.items()},
                                 }

                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'best': best,
                        'best_epoch': best_epoch
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_epoch%02d.pth' % epoch))
                    if checkpoint_best:
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))

        if args.evaluate:
            break

        dist.barrier()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f"best epoch: {best_epoch} / {config['max_epoch']}\n")

import os
import torch
import torch.nn.functional as F
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_mAP', action='store_true')
    parser.add_argument('--resume', action='store_true')
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
