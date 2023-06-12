from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
import random
from multiprocessing import Pool
import torch
from PIL import Image

from torch.utils.data.distributed import DistributedSampler



from dataset.randaugment import RandomAugment

import torch
from torchvision import transforms

import os 
import re 



class COCOCaptionFineTuneDataset(Dataset):
    def __init__(self, split='karpathy_train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train', 
        data_dir='/data/mshukor/data', black_image=False):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.args.BUTD100 = False

        self.mode = mode

        dataset_dir = Path(data_dir)
        coco_dir = dataset_dir.joinpath('COCO')
        vg_dir = dataset_dir.joinpath('VG')
        coco_img_dir = coco_dir.joinpath('images/')
        coco_feature_dir = coco_dir.joinpath('features')

        self.black_image = black_image

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data source: ', self.source)


        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        

        self.train_transform = transforms.Compose([                        
                transforms.RandomResizedCrop(args.image_size,scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])  
        self.test_transform = transforms.Compose([
            transforms.Resize((args.image_size,args.image_size),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])  



        data_info_path = dataset_dir.joinpath('COCO/dataset_coco.json')
        with open(data_info_path) as f:
            karpathy_data = json.load(f)

        split_rename = {
            'train': 'train',
            'restval': 'train',
            'val': 'val',
            'test': 'test'
        }

        n_images = 0

        data = []
        for datum in karpathy_data['images']:
            re_split = split_rename[datum['split']]
            if re_split != self.source.split('_')[-1]:
                continue

            if re_split == 'train':
                for d in datum['sentences']:

                    img_id = datum['filename'].split('.')[0]
                    new_datum = {
                        'img_id': img_id,
                        'sent': d['raw'].strip(),
                        'targets': [d['raw'].strip() for d in datum['sentences']],
                        'is_train': True,
                    }
                    data.append(new_datum)
            else:

                img_id = datum['filename'].split('.')[0]
                new_datum = {
                    'img_id': img_id,
                    # 'sent': d['raw'],
                    'targets': [d['raw'].strip() for d in datum['sentences']],
                    'is_train': False,
                }
                data.append(new_datum)

            n_images += 1

        if self.verbose:
            print(f"{self.source} has {n_images} images")
            print(f"Loaded {len(data)} data from", split)


        
        if isinstance(self.topk, float) and (0 < self.topk <= 1):
            used_samples = int(self.topk * len(data))
            data = random.sample(data, used_samples)
            if self.verbose:
                print(f"Use only {len(data)} data")

        elif self.topk > 0:
            data = data[:int(self.topk)]
            if self.verbose:
                print(f"Use only {len(data)} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))


        self.image_size = self.args.image_size

        if mode == "train" and self.args.use_data_augmentation:
            self.transform = self.train_transform
        else:
            self.transform = self.test_transform

        self.source_to_h5 = {}

        self.source_to_h5.update({
            'train2014': coco_img_dir.joinpath(f'train2014'),
            'val2014': coco_img_dir.joinpath(f'val2014'),
        })


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        ###### Image ######
        img_id = datum['img_id']
        out_dict['img_id'] = img_id


        if self.args.BUTD100:
            source = self.source
        else:
            if 'train' in img_id:
                source = 'train2014'
            elif 'val' in img_id:
                source = 'val2014'

        path = self.source_to_h5[source].joinpath(f"{img_id}.jpg")
    
        image = Image.open(path).convert('RGB')   


        out_dict["image"] = self.transform(image)

        if self.black_image:
            out_dict["image"] = torch.zeros_like(out_dict["image"])

        if datum['is_train']:
            sent = datum['sent'].strip()

            out_dict['sent'] = sent


        if 'targets' in datum:
            out_dict['targets'] = datum['targets']


        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)



        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id


        targets = []
        img_ids = []
        img_paths = []
        input_text = []
        images = []
        sents = []

        for i, entry in enumerate(batch):

            images.append(entry['image'])
            img_ids.append(entry['img_id'])

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']



            if 'targets' in entry:
                targets.append(entry['targets'])
            if 'sent' in entry:
                sents.append(entry['sent'])


        batch_entry['images'] = torch.stack(images)
        batch_entry['img_id'] = img_ids
        batch_entry['img_paths'] = img_paths
        if 'sent' in entry:
            batch_entry['sent'] = sents



        batch_entry['targets'] = targets

        batch_entry['task'] = 'caption'

        return batch_entry


def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption



def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1, data_dir='/data/mshukor/data', local_rank=None, world_size=None, verbose=False, 
               config_dir=None, black_image=False):




    dataset = COCOCaptionFineTuneDataset(
        split,
        # raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode, data_dir=data_dir, black_image=black_image)


    if distributed and mode == 'train':
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
    else:
        train_sampler = None
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            sampler=None,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    if verbose:
        loader.evaluator = COCOCaptionEvaluator()

    loader.task = 'caption'

    return loader



class COCOCaptionEvaluator:
    def __init__(self):
        import language_evaluation
        self.evaluator = language_evaluation.CocoEvaluator(verbose=False)


    def evaluate(self, predicts, answers):

        results = self.evaluator.run_evaluation(predicts, answers)

        return results