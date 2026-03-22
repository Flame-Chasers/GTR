import json
import os
from collections import defaultdict
import re
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
import random
from data.utils import pre_caption, random_mask_image
from nltk.corpus import stopwords

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# from .rm_tool import *

class ps_train_dataset(Dataset):
    def __init__(self, ann_files, transform, image_root, max_words=30):
        self.transform = transform
        self.max_words = max_words

        person_id2idx = {}
        n = 0
        self.pairs = []

        for ann_file in ann_files:
            anns = json.load(open(ann_file))
            # for ann in random.sample(anns, 21000):
            for ann in anns:
                person_id = ann['id']
                if person_id not in person_id2idx:
                    person_id2idx[person_id] = n
                    n += 1
                person_idx = person_id2idx[person_id]
                captions = ann['captions']
                image_path = os.path.join(image_root, ann['file_path'])
                sample_num = min(len(captions), 1)
                sampled_indices = random.sample(range(len(captions)), sample_num)

                for idx in sampled_indices:
                    cap = captions[idx]
                    if len(cap.split(" ")) > 200 or len(cap.split(" ")) < 10:
                        continue  
                    prob = np.mean(ann['logits'][idx]) if 'logits' in ann else 0
                    # prob = np.dot(ann['prob'][idx]) if 'prob' in ann else 0
                
                    pattern = r'\b(image-1|image 1|image1|in image-1|in image 1|in the image-1|in the image 1|in this image|' \
                    r'in the image|from the image|shown in the image|depicted in the image|as shown in the image|' \
                    r'target image|target picture|target photo|reference image|reference picture|' \
                    r'the given image|this image|the image|above image|below image|image shown|' \
                    r'image above|image below|shown above|as seen|as shown)\b'

                    cap = re.sub(pattern, '', cap, flags=re.IGNORECASE)

                    # if random.random() < 0.15:
                    #     cap = remove_words(ann['token_ids'][idx], ann['logits'][idx])
                    # cap = cap.replace('image-1|in the Image-1|in the image', '')
                    
                    self.pairs.append((image_path, cap, person_idx, prob))


    def __len__(self):
        # print(len(self.pairs))
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, caption, person, prob = self.pairs[index]

        image = Image.open(image_path).convert('RGB')
        image1 = self.transform(image)
        image2 = self.transform(image)

        caption = pre_caption(caption, self.max_words)

        return image1, image2, caption, prob, person


class ps_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.image_root = '/Path/To/Your/Image/Root' # for convinience, we set the image root in the dataset, you can also set it outside and pass it in as an argument
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        if 'CUHK' in ann_file:
            self.image_root = os.path.join(self.image_root, 'CUHK-PEDES/imgs')
        elif 'ICFG' in ann_file:
            self.image_root = os.path.join(self.image_root, 'ICFG-PEDES/imgs')
        elif 'RSTP' in ann_file:
            self.image_root = os.path.join(self.image_root, 'RSTPReid/imgs')
        elif 'UFineBench' in ann_file:
            self.image_root = os.path.join(self.image_root, 'UFineBench/UFine6926')



        print(f'Loading {self.image_root} datas...')
        # self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2person = []
        self.img2person = []

        person2img = defaultdict(list)
        person2txt = defaultdict(list)

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(os.path.join(self.image_root, ann['file_path']))
            person_id = ann['id']
            person2img[person_id].append(img_id)
            self.img2person.append(person_id)
            for caption in ann['captions']:
                self.text.append(pre_caption(caption, self.max_words))
                person2txt[person_id].append(txt_id)
                self.txt2person.append(person_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['file_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image_path, image, index
