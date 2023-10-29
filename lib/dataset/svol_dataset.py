import os
import csv
import json
import torch
import random
import numpy as np
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms
from lib.utils.tensor_utils import pad_sequences_1d
from lib.utils.box_utils import box_xyxy_to_cxcywh


class SVOLDataset(Dataset):
    '''Sketch Video Object Localization Dataset
    
    below is the exemplar structure of "targets" 
    targets = {
        'video': 'ILSVRC2015_train_00210000',
        'size': [480, 360],
        'sketch': 'rabbit0598',
        'category': 'rabbit',
        'track_ids': {0, 2},
        'total_boxes': 59,
        'num_boxes_per_frame': [1,1,1,1,0,0,0,1,1,1,1,2, ...],
        'bboxes': {
            0: [{'track_id': 0, 'bbox': tensor([0.5229, 0.4778, 0.2042, 0.3000])}], 
            13: [{'track_id': 0, 'bbox': tensor([0.5177, 0.4750, 0.2062, 0.2889])}], 
            25: [{'track_id': 0, 'bbox': tensor([0.4448, 0.4833, 0.2979, 0.2778])}], 
            38: [{'track_id': 0, 'bbox': tensor([0.1115, 0.5681, 0.2229, 0.2250])}], 
            51: [], 
            63: [], 
            76: [], 
            89: [{'track_id': 2, 'bbox': tensor([0.0771, 0.5778, 0.1542, 0.2889])}], 
            102: [{'track_id': 2, 'bbox': tensor([0.2688, 0.5819, 0.5333, 0.8139])}], 
            114: [{'track_id': 2, 'bbox': tensor([0.3771, 0.5639, 0.5875, 0.6944])}], 
            127: [{'track_id': 2, 'bbox': tensor([0.4635, 0.5625, 0.3187, 0.5306])}], 
            140: [{'track_id': 0, 'bbox': tensor([0.4604, 0.4667, 0.0542, 0.1333])}, {'track_id': 2, 'bbox': tensor([0.5854, 0.4583, 0.2167, 0.3278])}], 
            .
            .
            .
        }
    }
    '''
    CLASSES = {
        'sketchy': ['airplane', 'bear', 'bicycle', 'car', 'cat', 'cow', 'dog', 'elephant', 'horse', 'lion', 'lizard', 'motorcycle', 'rabbit', 'sheep', 'snake', 'squirrel', 'tiger', 'turtle', 'zebra'],
        'tu_berlin': ['airplane', 'bear', 'bicycle', 'bus', 'car', 'cat', 'cow', 'dog', 'elephant', 'horse', 'lion', 'monkey', 'motorcycle', 'panda', 'rabbit', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'zebra'],
        'quickdraw': ['airplane', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cat', 'cow', 'dog', 'elephant', 'horse', 'lion', 'monkey', 'motorcycle', 'panda', 'rabbit', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle', 'whale', 'zebra'],
        'union': ['airplane', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cat', 'cow', 'dog', 'elephant', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'panda', 'rabbit', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle', 'whale', 'zebra'],
        'intersection': ['airplane', 'bear', 'bicycle', 'car', 'cat', 'cow', 'dog', 'elephant', 'horse', 'lion', 'motorcycle', 'rabbit', 'sheep', 'snake', 'squirrel', 'tiger', 'zebra']
    }

    def __init__(self, root, anno_root, phase='train', num_frames=32,
                 video_dataset='imagenet_vid', sketch_dataset='sketchy',
                 tight_frame_sampling=False,
                 zeroshot_dataset_eval=False, 
                 zeroshot_category_eval=False,
                 unified_sketch_dataset=False):
        '''
        root: root directory for dataset
        anno_root: root directory for annotations
        phase: phase in ['train' , 'val', 'test']
        num_frames: number of input frames (i.e., sequence length) to transformer encoder
        tight_frame_sampling: if True, enable duplicate sampling to match the temporal size as "num_frames" for batching purpose.
        video_dataset: video dataset in ['imagenet_vid']
        sketch_dataset: sketch dataset in ['sketchy', 'tu_berlin', 'quickdraw']
        zeroshot_dataset_eval: use to train model on seen sketch dataset (styles) and evaluate on unseen sketch dataset (style). 
        zeroshot_category_eval: use to train model on seen classes and evaluate on unseen classes. 
        '''
        super(SVOLDataset).__init__()
        assert phase in ['train', 'val', 'test'], f'phase should be one of train/val/test.'
        assert video_dataset in ['imagenet_vid'], f'{video_dataset} is not available'
        assert sketch_dataset in ['sketchy', 'tu_berlin', 'quickdraw'], f'{sketch_dataset} is not available.'
        assert not (zeroshot_dataset_eval and zeroshot_category_eval)
        self.num_frames = num_frames
        self.tight_frame_sampling = tight_frame_sampling
        self.zeroshot_eval = zeroshot_dataset_eval or zeroshot_category_eval
        self.unified_sketch_dataset = unified_sketch_dataset

        #######################################################################################
        # speicify unseen categories for zershot category evaluation
        UNSEEN_CATEGORIES = ['airplane', 'bear', 'cat', 'cow', 'dog']
        #######################################################################################

        # load images
        if self.zeroshot_eval:
            # when zershot_eval is enabled, we use all video data regardless of train/val split
            # to solely evaluate zeroshot trasnferability on sketch
            folder = 'all'
        else:
            folder = 'val' if phase in ['val', 'test'] else 'train'
        if self.zeroshot_eval:
            self.video_root = os.path.join(root, video_dataset, 'Data', 'VID')
        else:
            self.video_root = os.path.join(root, video_dataset, 'Data', 'VID', folder)

        if self.unified_sketch_dataset:
            self.sketch_root = root
        else:
            self.sketch_root = os.path.join(root, f'{sketch_dataset}')

        # load annotations
        with open(os.path.join(root, video_dataset, 'Annotations', 'VID', f'{folder}.json')) as f:
            self.annos = json.load(f)

        with open(os.path.join(anno_root, f'{video_dataset}_{folder}.json')) as f:
            video_split = json.load(f)

        if self.unified_sketch_dataset:
            with open(os.path.join(anno_root, f'sketchy_{folder}.json')) as f:
                sketch_s_split = json.load(f)
            with open(os.path.join(anno_root, f'tu_berlin_{folder}.json')) as f:
                sketch_t_split = json.load(f)
            with open(os.path.join(anno_root, f'quickdraw_{folder}.json')) as f:
                sketch_q_split = json.load(f)
        else:
            with open(os.path.join(anno_root, f'{sketch_dataset}_{folder}.json')) as f:
                sketch_split = json.load(f)

        self.vid_sketch_pair = []
        if phase in ['train', 'test']:
            if self.unified_sketch_dataset:
                categories = self.CLASSES['intersection']
                for category in categories:
                    for video_id in video_split[category]:
                        for sketch_id in sketch_s_split[category]:
                            self.vid_sketch_pair.append([video_id, sketch_id, 'sketchy'])
                        for sketch_id in sketch_t_split[category]:
                            self.vid_sketch_pair.append([video_id, sketch_id, 'tu_berlin'])
                        for sketch_id in sketch_q_split[category]:
                            self.vid_sketch_pair.append([video_id, sketch_id, 'quickdraw'])
                random.shuffle(self.vid_sketch_pair)
            else:
                if zeroshot_category_eval:
                    categories = list(set(self.CLASSES[sketch_dataset]) - set(UNSEEN_CATEGORIES)) if phase == 'train' else UNSEEN_CATEGORIES
                else:
                    categories = self.CLASSES[sketch_dataset]
                for category in categories:
                    for video_id in video_split[category]:
                        for sketch_id in sketch_split[category]:
                            self.vid_sketch_pair.append([video_id, sketch_id])
                random.shuffle(self.vid_sketch_pair)
        elif phase == 'val':
            if self.unified_sketch_dataset:
                categories = self.CLASSES['intersection']
                self.vid_sketch_pair = []
                file_name = f'{video_dataset}_sketchy_{phase}.csv'
                with open(os.path.join(anno_root, file_name)) as f:
                    reader = csv.reader(f, delimiter=' ')
                    vid_s_pair = [(video, sketch, 'sketchy') for video, sketch in reader if sketch[:-4] in categories]
                file_name = f'{video_dataset}_tu_berlin_{phase}.csv'
                with open(os.path.join(anno_root, file_name)) as f:
                    reader = csv.reader(f, delimiter=' ')
                    vid_t_pair = [(video, sketch, 'tu_berlin') for video, sketch in reader if sketch[:-4] in categories]
                file_name = f'{video_dataset}_quickdraw_{phase}.csv'
                with open(os.path.join(anno_root, file_name)) as f:
                    reader = csv.reader(f, delimiter=' ')
                    vid_q_pair = [(video, sketch, 'quickdraw') for video, sketch in reader if sketch[:-4] in categories]
                self.vid_sketch_pair.extend(vid_s_pair)
                self.vid_sketch_pair.extend(vid_t_pair)
                self.vid_sketch_pair.extend(vid_q_pair)
            else:
                if zeroshot_category_eval:
                    file_name = f'zeroshot_class_{video_dataset}_{sketch_dataset}_{phase}.csv'
                elif zeroshot_dataset_eval:
                    file_name = f'zeroshot_style_{video_dataset}_{sketch_dataset}_{phase}.csv'
                else:
                    file_name = f'{video_dataset}_{sketch_dataset}_{phase}.csv'
                with open(os.path.join(anno_root, file_name)) as f:
                    reader = csv.reader(f, delimiter=' ')
                    self.vid_sketch_pair = [(video, sketch) for video, sketch in reader]

    def __len__(self):
        ''' number of total pairs '''
        return len(self.vid_sketch_pair)

    def __getitem__(self, idx):
        if self.unified_sketch_dataset:
            video_id, sketch_id, sketch_dataset = self.vid_sketch_pair[idx]
        else:
            video_id, sketch_id = self.vid_sketch_pair[idx]
        category = sketch_id[:-4]  # airplane0001 -> airplane

        vid_annos = self.annos[video_id]
        num_frames = vid_annos['num_frames']

        if not self.tight_frame_sampling and num_frames < self.num_frames:
            sampled_idxs = list(range(num_frames))
        else:
            sampling_rate = num_frames / self.num_frames
            sampled_idxs = [round(sampling_rate*i) for i in range(self.num_frames)]
        
        # input video
        video_dir = os.path.join(self.video_root, video_id)
        if self.zeroshot_eval:
            if os.path.exists(os.path.join(self.video_root, 'train', video_id)):
                video_dir = os.path.join(self.video_root, 'train', video_id)
            elif os.path.exists(os.path.join(self.video_root, 'val', video_id)):
                video_dir = os.path.join(self.video_root, 'val', video_id)
            else:
                raise ValueError
        video_frames = []
        for sampled_idx in sampled_idxs:
            frame_dir = os.path.join(video_dir, '{:06d}'.format(sampled_idx)+'.JPEG')
            frame_img = Image.open(frame_dir).convert('RGB')
            video_frames.append(frame_img)

        # input sketch
        if self.unified_sketch_dataset:
            sketch_dir = os.path.join(self.sketch_root, sketch_dataset, category, sketch_id+'.png')
            sketch_img = Image.open(sketch_dir).convert('RGB')
        else:
            sketch_dir = os.path.join(self.sketch_root, category, sketch_id+'.png')
            sketch_img = Image.open(sketch_dir).convert('RGB')

        # transform
        video_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        sketch_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomCrop((168, 168)),
            # transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        video_frames = torch.stack([video_transform(frame_img) for frame_img in video_frames], dim=0) # TxCxHxW
        sketch_img = sketch_transform(sketch_img)  # CxHxW
        sketch_img = sketch_img.unsqueeze(0)  # 1xCxHxW; add dimension 0 for batching purpose

        # targets
        targets = dict()
        bboxes = defaultdict(list) # bboxes per video (indexed by trackid & label)
        track_ids = set()
        w, h = vid_annos['size']
        for frame_idx, anno_per_frame in vid_annos['frames'].items():
            # frames are uniformly sampled (e.g., 64 frames) across the video
            # skip frames that are not included in sampled indices
            if int(frame_idx) not in sampled_idxs: 
                continue

            # assign bbox as [] if sketch object is not included in the current frame
            obj_per_frame = [obj_anno['label'] for obj_anno in anno_per_frame]
            if category not in obj_per_frame: 
                bboxes[frame_idx] = []
                continue

            for obj_anno in anno_per_frame:
                track_id = obj_anno['track_id']
                label = obj_anno['label']

                if category == label:
                    bbox = obj_anno['bbox']  # bbox per object
                    bbox = torch.tensor(bbox)
                    bbox = box_xyxy_to_cxcywh(bbox)
                    bbox = bbox / torch.tensor([w, h, w, h], dtype=torch.float32)

                    bboxes[frame_idx].append({
                        'track_id': track_id,
                        'bbox': bbox
                    })
                    track_ids.add(track_id)

        # num_boxes_per_frame = [len(boxes) for boxes in bboxes.values()]
        num_boxes_per_frame = [0] * self.num_frames
        for i, boxes in enumerate(bboxes.values()):
            num_boxes_per_frame[i] = len(boxes)
        total_boxes = sum(num_boxes_per_frame)
        assert total_boxes != 0, \
            f'there is no sampled bboxes for category "{category}" in current video "{video_id}"'

        model_inputs = dict()
        model_inputs['input_sketch'] = sketch_img
        model_inputs['input_video'] = video_frames

        targets['video'] = video_id
        targets['size'] = [w, h]
        targets['sketch'] = sketch_id
        targets['category'] = category
        targets['track_ids'] = list(track_ids)
        targets['total_boxes'] = total_boxes
        targets['num_boxes_per_frame'] = num_boxes_per_frame
        targets['bboxes'] = bboxes

        return dict(model_inputs=model_inputs, targets=targets)

    def get_vid_info(self):
        # TODO
        pass


def build_dataset(args, phase):
    return SVOLDataset(
        phase=phase,
        root=args.root,
        anno_root=args.anno_root,
        num_frames=args.num_frames,
        video_dataset=args.video_dataset,
        sketch_dataset=args.sketch_dataset,
        tight_frame_sampling=args.tight_frame_sampling,
        zeroshot_dataset_eval=args.zeroshot_dataset_eval, 
        zeroshot_category_eval=args.zeroshot_category_eval,
        unified_sketch_dataset=args.unified_sketch_dataset
    )


def collate_fn(batch):
    batched_targets = [d['targets'] for d in batch]
    input_keys = batch[0]['model_inputs'].keys()
    batched_inputs = dict()
    for k in input_keys:
        batched_inputs[k] = pad_sequences_1d(
            [d['model_inputs'][k] for d in batch],
            dtype=torch.float32,
        )
    return batched_inputs, batched_targets


def prepare_batch_inputs(batched_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_sketch = batched_inputs['input_sketch'][0].to(device, non_blocking=non_blocking),  # Nx1x3x224x224
        src_sketch_mask = batched_inputs['input_sketch'][1].to(device, non_blocking=non_blocking),  # Nx1
        src_video = batched_inputs['input_video'][0].to(device, non_blocking=non_blocking),  # NxTx3x224x224
        src_video_mask = batched_inputs['input_video'][1].to(device, non_blocking=non_blocking)  # NxT
    )
    return model_inputs


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = SVOLDataset(
        root='/home/sangmin/drive3/sangmin/data/svol/',
        video_dataset='imagenet_vid',
        sketch_dataset='sketchy',
        phase='train',
        num_frames=32,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    dataiter = iter(dataloader)
    batch = dataiter.next()
    print(batch)
    print(batch['model_inputs']['input_sketch'].shape) # 1x768
    print(batch['model_inputs']['input_video'].shape) # 1x32x768