import json
import os
import numpy as np

from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

from collections import defaultdict
from data.utils import pre_caption

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ps_pretrain_dataset(Dataset):
    def __init__(self, ann_files, transform, image_roots, max_words=30,
                 ra_probability=0.1, id_agnostic=False, select_index_file=None):

        if select_index_file is not None and len(ann_files) != 1:
            raise ValueError('cannot assign select_index_file when multi ann files.')

        self.transform = transform
        self.max_words = max_words
        self.ra_probability = ra_probability

        person2text = defaultdict(list)
        person_id2idx = {}
        n = 0
        all_pairs = []

        for ann_file, image_root in zip(ann_files, image_roots):
            # for each dataset, create separate person collection
            person_id2idx.clear()

            anns = json.load(open(ann_file))
            for ann in anns:
                # if id_agnostic is True, we assume that every image
                # is captured from different identity.
                if not id_agnostic:
                    person_id = ann['id']
                else:
                    person_id = n

                if person_id not in person_id2idx.keys():
                    person_id2idx[person_id] = n
                    n += 1
                person_idx = person_id2idx[person_id]
                for cap in ann['captions']:
                    image_path = os.path.join(image_root, ann['file_path'])
                    all_pairs.append((image_path, cap, person_idx))
                    person2text[person_idx].append(cap)

        self.person2text = defaultdict(list)
        self.pairs = []
        if select_index_file is None:
            self.pairs = all_pairs
            self.person2text = person2text
        else:
            select_index = json.load(open(select_index_file))
            for index in select_index:
                self.pairs.append(all_pairs[index])
                image_path, cap, person_idx = all_pairs[index]
                self.person2text[person_idx].append(cap)

    def __len__(self):
        return len(self.pairs)

    def augment(self, caption, person):
        caption_aug = caption
        if np.random.random() < self.ra_probability:
            caption_aug = np.random.choice(self.person2text[person], 1).item()
        if caption_aug == caption:
            replace = 0
        else:
            replace = 1
        return caption_aug, replace

    def __getitem__(self, index):
        image_path, caption, person = self.pairs[index]
        caption_aug, replace = self.augment(caption, person)

        image = Image.open(image_path).convert('RGB')
        image1 = self.transform(image)
        image2 = self.transform(image)

        caption1 = pre_caption(caption, self.max_words)
        caption2 = pre_caption(caption_aug, self.max_words)

        return image1, image2, caption1, caption2, person, replace
