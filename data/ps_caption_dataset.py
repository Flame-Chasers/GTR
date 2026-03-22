import json
import os

from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

from collections import defaultdict
from data.utils import pre_caption

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ps_caption_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, prompt='', few_shot_file=None):
        anns = []
        for f in ann_file:
            anns += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        person_id2idx = {}
        n = 0
        all_pairs = []

        for ann in anns:
            person_id = ann['id']
            if person_id not in person_id2idx.keys():
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]
            for cap in ann['captions']:
                all_pairs.append((ann['file_path'], cap, person_idx))

        self.pairs = []
        if few_shot_file is None:
            self.pairs = all_pairs
        else:
            few_shot_index = json.load(open(few_shot_file))
            for index in few_shot_index:
                self.pairs.append(all_pairs[index])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, caption, person = self.pairs[index]

        image_path = os.path.join(self.image_root, image_path)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = self.prompt + pre_caption(caption, self.max_words)

        return image, caption, person


class ps_caption_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['file_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index
