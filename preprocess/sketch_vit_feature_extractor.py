import os
import glob
import torch
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import ViTFeatureExtractor, ViTModel
import albumentations


def extract_sketch_features(root_dir, save_dir, hflip=False):
    
    CLASS_COUNT = {}
    for class_name in ID_TO_NAME.values():
        CLASS_COUNT[class_name] = 0 

    for class_name in ID_TO_NAME.values():
        os.makedirs(os.path.join(save_dir, 'before_norm', 'class_token', class_name), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'before_norm', 'feature_avg', class_name), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'after_norm', 'class_token', class_name), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'after_norm', 'feature_avg', class_name), exist_ok=True)

    transform = transforms.PILToTensor()
    dataset = ImageFolder(root_dir, transform)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

    for _ in trange(5, desc='times'):
        for i, (image, target) in tqdm(enumerate(dataloader),
                                       desc='Extracting Features',
                                       total=len(dataloader)):
            image = image.squeeze()
            tf = albumentations.Compose([
                albumentations.HorizontalFlip(p=0.2),
                albumentations.VerticalFlip(p=0.2),
                albumentations.RandomRotate90(p=0.2),
                albumentations.Transpose(p=0.2),
                albumentations.ElasticTransform(p=0.2, border_mode=1)
            ])
            image = np.asarray(image).transpose(1, 2, 0)
            image = tf(image=image)
            image = image['image']
            image = torch.tensor(image)

            # if hflip:
            #     # horizontal flip
            #     image = transforms.functional.hflip(image)
            
            inputs = feature_extractor(images=[image], return_tensors="pt")
            outputs = model(pixel_values=inputs['pixel_values'].cuda(), output_hidden_states=True)
            before_norm = outputs.hidden_states[-1].squeeze().detach().cpu()
            after_norm = outputs.last_hidden_state.squeeze().detach().cpu()

            class_name = ID_TO_NAME[int(target)]
            CLASS_COUNT[class_name] += 1
            if not hflip:
                torch.save(before_norm[0].contiguous(), os.path.join(save_dir, 'before_norm', 'class_token', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}.pt'))
                torch.save(before_norm[1:].mean(0), os.path.join(save_dir, 'before_norm', 'feature_avg', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}.pt'))
                torch.save(after_norm[0].contiguous(), os.path.join(save_dir, 'after_norm', 'class_token', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}.pt'))
                torch.save(after_norm[1:].mean(0), os.path.join(save_dir, 'after_norm', 'feature_avg', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}.pt'))
            else:
                torch.save(before_norm[0].contiguous(), os.path.join(save_dir, 'before_norm', 'class_token', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}h.pt'))
                torch.save(before_norm[1:].mean(0), os.path.join(save_dir, 'before_norm', 'feature_avg', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}h.pt'))
                torch.save(after_norm[0].contiguous(), os.path.join(save_dir, 'after_norm', 'class_token', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}h.pt'))
                torch.save(after_norm[1:].mean(0), os.path.join(save_dir, 'after_norm', 'feature_avg', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}h.pt'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract features.')
    parser.add_argument('--root_dir', default='/mnt/server12_hard3/sangmin/data/svol')
    parser.add_argument('--save_dir', default='/mnt/server12_hard3/sangmin/data/svol')
    parser.add_argument('--dataset', default='sketchy', choices=['sketchy', 'tu_berlin', 'quickdraw'])
    parser.add_argument('--hflip', action='store_true')
    args = parser.parse_args()

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').cuda()
    model.eval()
    
    args.root_dir = os.path.join(args.root_dir, args.dataset)
    args.save_dir = os.path.join(args.save_dir, f'{args.dataset}_features', 'freeze')

    if args.dataset == 'sketchy':
        ID_TO_NAME = {
            0:'airplane',1:'bear',2:'bicycle',3:'car',4:'cat',5:'cow',6:'dog',7:'elephant',8:'horse',9:'lion',10:'lizard',11:'motorcycle',12:'rabbit',13:'sheep',14:'snake',15:'squirrel',16:'tiger',17:'turtle',18:'zebra'
        }
    elif args.dataset == 'tu_berlin':
        ID_TO_NAME = {
            0:'airplane',1:'bear',2:'bicycle',3:'bus',4:'car',5:'cat',6:'cow',7:'dog',8:'elephant',9:'horse',10:'lion',11:'monkey',12:'motorcycle',13:'panda',14:'rabbit',15:'sheep',16:'snake',17:'squirrel',18:'tiger',19:'train',20:'zebra'
        }
    elif args.dataset == 'quickdraw':
        ID_TO_NAME = {
            0:'airplane',1:'bear',2:'bicycle',3:'bird',4:'bus',5:'car',6:'cat',7:'cow',8:'dog',9:'elephant',10:'horse',11:'lion',12:'monkey',13:'motorcycle',14:'panda',15:'rabbit',16:'sheep',17:'snake',18:'squirrel',19:'tiger',20:'train',21:'turtle',22:'whale',23:'zebra'
        }

    extract_sketch_features(args.root_dir, args.save_dir, args.hflip)