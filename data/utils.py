import json
import os
import re
import torch
import torch.distributed as dist
from models.blip import create_vit, init_tokenizer
import numpy as np
import utils
import random

random.seed(3407)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def pre_prob(sims, max_words=50):
    sim_tensor = torch.tensor(sims)
    # sim_tensor[sim_tensor == 1] -= 1e-4

    if sim_tensor.size(0) < max_words:
        sim_tensor = torch.cat([sim_tensor, torch.ones(max_words - len(sim_tensor))])
    elif sim_tensor.size(0) > max_words:
        sim_tensor = sim_tensor[:max_words]

    return sim_tensor

def save_file(new_res_list, filename='data.json'):
    with open(filename, 'w') as f:
        json.dump(new_res_list, f, ensure_ascii=False, indent=4)
    print(f"----------Saved progress to '{filename}'---------------")


import random
import numpy as np
import torch
from PIL import Image

def random_mask_image(image, mask_prob=0.1, mask_size=(16, 16)):
    image_np = np.array(image)
    
    height, width, _ = image_np.shape
    num_pixels = height * width
    
    num_masks = int(num_pixels * mask_prob / (mask_size[0] * mask_size[1]))

    for _ in range(num_masks):
        top_left_x = random.randint(0, width - mask_size[0])
        top_left_y = random.randint(0, height - mask_size[1])

        image_np[top_left_y:top_left_y + mask_size[1], top_left_x:top_left_x + mask_size[0]] = 0

    masked_image = Image.fromarray(image_np)
    return masked_image



def mask_words(words, probs, tokenizer, seed=3407):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    probs = [1 - p for p in probs] 
    probs = [min(max(p, 0), 0.15) for p in probs]
    posterior_probs = probs

    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param words: list of str, tokenized sentence.
    :param probs: list of float, probability for masking each word.
    :param tokenizer: A tokenizer instance used for token decoding.
    :return: (list of str, list of int), masked tokens and related labels for MLM prediction
    """
    # Ensure the length of words and probabilities match
    if len(words) != len(probs):
        raise ValueError(f"Length of words ({len(words)}) and probs ({len(probs)}) do not match.")
    
    # Precompute a list of valid tokens for random replacements
    token_range = list(range(1, len(tokenizer)))  # 1 ~ 49405
    valid_tokens = [
        tokenizer.decode(token_id).replace(" ", "") for token_id in token_range
        if re.match(r'^[a-zA-Z]+$', tokenizer.decode(token_id).replace(" ", ""))
    ]
    
    # Initialize labels and result list
    labels = []
    masked_words = []

    for i, (word, p) in enumerate(zip(words, posterior_probs)):
        if random.random() < p:
            p_ = random.random()
            if p_ < 0.8:
                # 80% chance to replace with [MASK]
                labels.append(word)
                masked_words.append("[MASK]")
            elif p_ < 0.9:
                # 10% chance to replace with a random word
                labels.append(word)
                masked_words.append(random.choice(valid_tokens))
            else:
                # 10% chance to leave unchanged (label = 0)
                labels.append(0)
                masked_words.append(word)
        else:
            # No masking applied
            labels.append(0)
            masked_words.append(word)
    
    return masked_words, labels

def prior_mask(p):
    prior_mapping = {
        (0, 0.2): 0.85,
        (0.2, 0.4): 0.6,
        (0.6, 0.8): 0.4,
        (0.8, 1): 0.15
    }
    for (low, high), prob in prior_mapping.items():
        if low <= p < high:
            return prob
    return 0.5 


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_likelihood(p, m):
    return sigmoid(15 * (0.5 - p)) if m == 1 else 1 - sigmoid(5 * (0.5 - p))


def marginal_probability(p, samples=1000):
    m = np.random.choice([0, 1], size=samples)
    likelihoods = np.array([compute_likelihood(p, m_i) for m_i in m])
    priors = np.array([prior_mask(m_i) for m_i in m])
    return np.mean(likelihoods * priors)


def bayesian_inference(p, samples=1000):
    marginal = marginal_probability(p, samples)
    if marginal == 0:
        return 0 
    posterior = (compute_likelihood(p, 1) * prior_mask(1)) / marginal
    return min(posterior, 1) 


def pre_words(words, max_words=50):
    caption = ' '.join(words)
    cap = pre_caption(caption, max_words)
    return cap

def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


def pre_caption_mask(caption, sims, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    


    words, labels = mask_words(caption_words, sims)
    # if len(words) > max_words:
    caption = ' '.join(words[:max_words])

    return caption, labels
        

def pre_question(question, max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    )
    question = question.rstrip(' ')

    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])

    return question


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json' % (filename, utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json' % filename)

    json.dump(result, open(result_file, 'w'))

    dist.barrier()

    if utils.is_main_process():
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json' % (filename, rank))
            res = json.load(open(result_file, 'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        json.dump(result, open(final_result_file, 'w'))
        print('result file saved to %s' % final_result_file)

    return final_result_file


from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val': 'coco_karpathy_val_gt.json', 'test': 'coco_karpathy_test_gt.json'}

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

    return coco_eval


def cuhk_caption_eval(gt_file, results_file):
    """
    use coco_eval to evaluate the performance of generated captions.
    """

    # create coco object and coco_result object
    coco = COCO(gt_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

    return coco_eval


if __name__ == '__main__':
    cuhk_caption_eval('../annotation/cuhk_gt/cuhk_test_gt.json', '../output/caption_ps/nucleus_few0.01_ep30/result/test_epoch0.json')