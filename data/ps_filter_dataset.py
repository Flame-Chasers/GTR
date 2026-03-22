import json
import os

from PIL import Image
from torch.utils.data import Dataset

from data.utils import pre_caption


class ps_filter_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.pairs = []
        for img_id, ann in enumerate(self.ann):
            for caption in ann['captions']:
                cap = pre_caption(caption, self.max_words)
                self.pairs.append((ann['file_path'], cap))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, caption = self.pairs[index]

        image_path = os.path.join(self.image_root, image_path)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, caption, index

    def filter(self, itc_score, itm_score, save_file, strategy='itm', threshold=0.5):
        score = itc_score if strategy == 'itc' else itm_score
        keep = (score > threshold)
        print(f'kept pairs size: {sum(keep)} / {keep.size(0)}')
        print(f'kept pairs ratio: {sum(keep) / keep.size(0)}')

        new_annotation = []
        text_id = 0
        for ann in self.ann:
            new_ann = {'id': ann['id'], 'file_path': ann['file_path'], 'captions': []}

            for caption in ann['captions']:
                if keep[text_id]:
                    new_ann['captions'].append(caption)
                text_id += 1
            if new_ann['captions']:
                new_annotation.append(new_ann)

        json.dump(new_annotation, open(save_file, 'w'))


