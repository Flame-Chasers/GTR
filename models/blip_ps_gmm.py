from sklearn.mixture import GaussianMixture
import torch
import torch.nn.functional as F
from torch import nn
import random
from models.blip import create_vit, init_tokenizer
from models.med import BertConfig, BertModel
from transformers import BatchEncoding

class BLIP_PS(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 embed_dim=256,
                 queue_size=57600,
                 momentum=0.995,
                 max_length=50,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        self.max_length = max_length
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width

        encoder_config.output_hidden_states = True
        encoder_config.output_attentions = True

        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)
        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

        # create momentum encoders
        self.visual_encoder_m, vision_width = create_vit(vit, image_size)
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=encoder_config, add_pooling_layer=False)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("prob_queue", torch.ones(queue_size))
        self.register_buffer("idx_queue", torch.full((1, queue_size), -100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.gmm_cache = None

    def set_gmm_cache(self, gmm_cache):
        self.gmm_cache = gmm_cache

    def forward(self, image1, image2, caption1, prob, alpha, idx, confident=True, keepsim=False, progress=0.5):
        """
        Args:
            prob (Tensor):
                the probability of the generated caption

            confident (bool):
                False: consider the probability of the generated caption when computing loss.
                True: otherwise
        """
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        device = image1.device
        
        image_embeds = self.visual_encoder(image1)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        
        text = self.tokenizer(caption1, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").to(device)
        
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text', output_attentions=True,)
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
            
        ###============== Image-text Contrastive Learning ===================###
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image2)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_m_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_m_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            prob_all = torch.cat([prob, self.prob_queue.clone().detach()], dim=-1)

            sim_i2t_m = image_feat_m @ text_feat_m_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_m_all / self.temp

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_m_all / self.temp
        sim_t2i = text_feat @ image_feat_m_all / self.temp

        with torch.no_grad():
            
            sim = ((sim_t2i + sim_i2t) / 2).diag().detach().cpu().numpy()  # [B]
            prob_np = prob.detach().cpu().numpy() 
            
            sim_scaled = (sim - sim.mean()) / (sim.std() + 1e-6)
            prob_scaled = (prob_np - prob_np.mean()) / (prob_np.std() + 1e-6)
            
            progress=0.2
            
            combined = progress * sim_scaled + (1.0 - progress) * prob_scaled

            clean_probs = self.gmm_cache.get_probs(combined)   # [B]
            weights = torch.from_numpy(clean_probs).to(sim_i2t.device)  # [B]

        loss_i2t_per_sample = -(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets).sum(dim=1)  # [B]
        loss_t2i_per_sample = -(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets).sum(dim=1)  # [B]
        loss_i2t = (loss_i2t_per_sample * weights).sum() / (weights.sum() + 1e-6)
        loss_t2i = (loss_t2i_per_sample * weights).sum() / (weights.sum() + 1e-6)
        loss_ita = (loss_i2t + loss_t2i) / 2

        idxs = concat_all_gather(idx)
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idxs, prob)

        ###============== Image-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # forward the positive image-text pair
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )

        with torch.no_grad():
            mask = torch.eq(idx, idx.t())

            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp

            weights_i2t = F.softmax(sim_i2t, dim=1)
            weights_i2t.masked_fill_(mask, 0)

            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_t2i.masked_fill_(mask, 0)

            # select a negative image for each text
            image_neg_idx = torch.multinomial(weights_t2i, 1).flatten()
            image_embeds_neg = image_embeds[image_neg_idx]

            # select a negative text for each image
            text_neg_idx = torch.multinomial(weights_i2t, 1).flatten()
            text_ids_neg = encoder_input_ids[text_neg_idx]
            text_atts_neg = text.attention_mask[text_neg_idx]
            prob_neg = prob[text_neg_idx]

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask=text_atts_all,
                                       encoder_hidden_states=image_embeds_all,
                                       encoder_attention_mask=image_atts_all,
                                       return_dict=True,
                                       )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        bs = image1.size(0)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(device)
        weights_itm = torch.cat([weights, torch.ones(2*bs, dtype=torch.float).to(device)], dim=0)
        loss_itm = (F.cross_entropy(vl_output, itm_labels, reduction='none') * weights_itm).sum() / (weights_itm.sum() + 1e-6)

        if keepsim:
            return loss_ita, loss_itm, weights.detach().cpu().numpy()
        else:
            return loss_ita, loss_itm

    def generate(self, input_ids, attention_mask, weights):
        bs = input_ids.size(0)

        weights = weights.flatten(0, -2)
        pred = torch.multinomial(weights, 1).view(bs, -1)

        exp_pred = torch.full_like(input_ids, self.tokenizer.enc_token_id)
        exp_pred[:, 1:] = pred
        # pad_token_id is 0
        generated_input_ids = exp_pred * attention_mask

        labels = (exp_pred != input_ids) * attention_mask
        labels[generated_input_ids == self.tokenizer.pad_token_id] = -100
        labels[generated_input_ids == self.tokenizer.enc_token_id] = -100

        return generated_input_ids, labels

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs, prob):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        probs = concat_all_gather(prob)

        batch_size = image_feats.shape[0]

        ptr = int(self.ptr_queue)
        # assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        empty = self.image_queue.size(1) - ptr
        if batch_size <= empty:
            self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
            self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
            self.prob_queue[ptr:ptr + batch_size] = probs
            self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        else:
            self.image_queue[:, ptr:] = image_feats[:empty].T
            self.text_queue[:, ptr:] = text_feats[:empty].T
            self.prob_queue[ptr:] = probs[:empty]
            self.idx_queue[:, ptr:] = idxs[:empty].T

            self.image_queue[:, :batch_size - empty] = image_feats[empty:].T
            self.text_queue[:, :batch_size - empty] = text_feats[empty:].T
            self.prob_queue[:batch_size - empty] = probs[empty:]
            self.idx_queue[:, :batch_size - empty] = idxs[empty:].T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.ptr_queue[0] = ptr
    
    def mask(self, input_ids, vocab_size, targets=None, masked_indices=None, probability_matrix=None):

        device = input_ids.device
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False  
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        masked_indices.to(device)
        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=device)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(input_ids.device)
        input_ids[indices_random] = random_words[indices_random]
        
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

def blip_ps(**kwargs):
    model = BLIP_PS(**kwargs)
    return model


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


def fit_gmm_and_get_clean_probs(sim_matrix, n_components=2):
    sims = sim_matrix.detach().cpu().numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=0, max_iter=50)
    gmm.fit(sims)
    means = gmm.means_.flatten()
    clean_comp = means.argmax()
    probs = gmm.predict_proba(sims)[:, clean_comp]
    return torch.tensor(probs, device=sim_matrix.device).reshape(sim_matrix.shape)

import numpy as np
from sklearn.mixture import GaussianMixture

class GMMCache:
    def __init__(self, n_components=2):
        self.gmm = GaussianMixture(n_components=n_components, random_state=0, max_iter=50)
        self.fitted = False

    def fit(self, sims):
        sims = np.array(sims).reshape(-1, 1)
        self.gmm.fit(sims)
        self.fitted = True

    def get_probs(self, sims):
        # sims: [N]
        if not self.fitted:
            return np.ones(sims.shape)
        
        if isinstance(sims, torch.Tensor):
            sims = sims.detach().cpu().numpy()
    
        sims = np.array(sims).reshape(-1, 1)
        means = self.gmm.means_.flatten()
        clean_idx = np.argmax(means)
        probs = self.gmm.predict_proba(sims)[:, clean_idx]
        return probs
