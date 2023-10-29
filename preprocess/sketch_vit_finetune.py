import os
import glob
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import ViTFeatureExtractor, ViTForImageClassification
import albumentations


class AverageMeter(object):
    """Computes and stores the average and current/max/min value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10

    def update(self, val, n=1):
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SketchViT(nn.Module):
    def __init__(self, args, pretrained_model):
        super(SketchViT, self).__init__()

        ViTBlocks, Classifier = list(pretrained_model.children())
        # 0: ViTBlocks, 1: Classifier
        ViTEmbeddings, ViTLayers, LayerNorm = list(ViTBlocks.children())
        # 0: ViTEmbeddings, 1: ViTLayers, 2: LayerNorm

        vit_layers = list(list(ViTLayers.children())[0].children()) # 12 layers
        layers_freeze = vit_layers[:-int(args.finetune_layers)] # layers to freeze
        layers_unfreeze = vit_layers[-int(args.finetune_layers):] # layers to fine-tune

        # TO FREEZE: ViTEmbeddings -- layers_freeze
        # TO FINE-TUNE: layers_unfreeze -- LayerNorm -- Classifier
        for param in ViTEmbeddings.parameters():
            param.requires_grad = False
        for layer in layers_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in layers_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        for param in LayerNorm.parameters():
            param.requires_grad = True
        for param in Classifier.parameters():
            param.requires_grad = True

        # Classifier = nn.Linear(768, 19) if args.dataset == 'sketchy' else nn.Linear(768, 21)

        self.ViTEmbeddings = ViTEmbeddings
        self.layers_freeze = nn.Sequential(*layers_freeze)
        self.layers_unfreeze = nn.Sequential(*layers_unfreeze)
        self.layer_norm = LayerNorm
        self.Classifier = Classifier

    def forward(self, x):
        x = self.ViTEmbeddings(x)
        x = self.layers_freeze(x)
        before_norm = self.layers_unfreeze(x)
        after_norm = self.layer_norm(before_norm)
        out = self.Classifier(after_norm)
        return before_norm, after_norm, out


def finetune_sketch_vit(args, feature_extractor):
    if args.dataset == 'sketchy':
        num_labels = 19
    elif args.dataset == 'tu_berlin':
        num_labels = 21
    elif args.dataset == 'quickdraw':
        num_labels = 24
    pretrained_model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=num_labels
        )
    model = SketchViT(args, pretrained_model)
    print(model)
    print('Num. of trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = model.cuda()
    model.train()

    param_dicts = [{'params': [param for name, param in model.named_parameters() if param.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss().cuda()

    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(args.root_dir, transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32, num_workers=4)

    loss_meter = AverageMeter()
    for epoch_i in tqdm(range(args.epochs), desc='Training Sketch-ViT'):
        for i, (images, targets) in tqdm(enumerate(dataloader),
                                         desc='Training Epoch',
                                         total=len(dataloader)):
            optimizer.zero_grad()
            input_images = []
            for img in images:
                tf = albumentations.Compose([
                    albumentations.HorizontalFlip(p=0.2),
                    albumentations.VerticalFlip(p=0.2),
                    albumentations.RandomRotate90(p=0.2),
                    albumentations.Transpose(p=0.2),
                    albumentations.ElasticTransform(p=0.2, border_mode=1)
                ])
                img = np.asarray(img).transpose(1, 2, 0)
                img = tf(image=img)
                img = img['image']
                img = torch.tensor(img)

                inputs = feature_extractor(images=img, return_tensors="pt")
                input_images.append(inputs['pixel_values'])
            input_images = torch.cat(input_images, dim=0)

            _ , _, outputs = model(input_images.cuda())
            loss = criterion(outputs[:, 0, :].contiguous(), targets.cuda())
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.detach().cpu().float())

            if (i+1) % args.print_interval == 0:
                print(f'[{epoch_i+1}/{args.epochs}|{i+1}/{len(dataloader)}] loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})')

        if (epoch_i+1) % args.save_interval == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch_i
            }
            torch.save(
                checkpoint,
                f'{args.model_dir.split(".")[0]}_{str(epoch_i)}.{args.model_dir.split(".")[1]}'
            )
    return model


def extract_sketch_features(args, feature_extractor, model):
    
    CLASS_COUNT = {}
    for class_name in ID_TO_NAME.values():
        CLASS_COUNT[class_name] = 0 

    for class_name in ID_TO_NAME.values():
        os.makedirs(os.path.join(args.save_dir, 'before_norm', 'class_token', class_name), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'before_norm', 'feature_avg', class_name), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'after_norm', 'class_token', class_name), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'after_norm', 'feature_avg', class_name), exist_ok=True)

    transform = transforms.PILToTensor()
    dataset = ImageFolder(args.root_dir, transform)
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

            inputs = feature_extractor(images=[image], return_tensors="pt")
            before_norm, after_norm, _ = model(inputs['pixel_values'].cuda())

            before_norm = before_norm.squeeze().detach().cpu() # 1x197x768 -> 197x768
            after_norm = after_norm.squeeze().detach().cpu() # 1x197x768 -> 197x768

            class_name = ID_TO_NAME[int(target)]
            CLASS_COUNT[class_name] += 1
            torch.save(before_norm[0].contiguous(), os.path.join(args.save_dir, 'before_norm', 'class_token', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}.pt'))
            torch.save(before_norm[1:].mean(0), os.path.join(args.save_dir, 'before_norm', 'feature_avg', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}.pt'))
            torch.save(after_norm[0].contiguous(), os.path.join(args.save_dir, 'after_norm', 'class_token', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}.pt'))
            torch.save(after_norm[1:].mean(0), os.path.join(args.save_dir, 'after_norm', 'feature_avg', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}.pt'))

    # for class_name in ID_TO_NAME.values():
    #     CLASS_COUNT[class_name] = 0 

    # for i, (image, target) in tqdm(enumerate(dataloader),
    #                                desc='Extracting Features',
    #                                total=len(dataloader)):
    #     image = image.squeeze()
    #     # horizontal flip
    #     image = transforms.functional.hflip(image)
    #     inputs = feature_extractor(images=[image], return_tensors="pt")
    #     before_norm, after_norm, _ = model(inputs['pixel_values'].cuda())

    #     before_norm = before_norm.squeeze().detach().cpu() # 1x197x768 -> 197x768
    #     after_norm = after_norm.squeeze().detach().cpu() # 1x197x768 -> 197x768

    #     class_name = ID_TO_NAME[int(target)]
    #     CLASS_COUNT[class_name] += 1
    #     torch.save(before_norm[0].contiguous(), os.path.join(args.save_dir, 'before_norm', 'class_token', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}h.pt'))
    #     torch.save(before_norm[1:].mean(0), os.path.join(args.save_dir, 'before_norm', 'feature_avg', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}h.pt'))
    #     torch.save(after_norm[0].contiguous(), os.path.join(args.save_dir, 'after_norm', 'class_token', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}h.pt'))
    #     torch.save(after_norm[1:].mean(0), os.path.join(args.save_dir, 'after_norm', 'feature_avg', class_name, f'{class_name}_{str(CLASS_COUNT[class_name])}h.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features.')
    parser.add_argument('--root_dir', default='/mnt/server12_hard3/sangmin/data/svol')
    parser.add_argument('--save_dir', default='/mnt/server12_hard3/sangmin/data/svol')
    parser.add_argument('--dataset', default='sketchy', choices=['sketchy', 'tu_berlin', 'quickdraw'])
    parser.add_argument('--finetune_layers', default=1)
    parser.add_argument('--model_dir', default='sketch_vit.pt')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--wd', default=1e-4)
    parser.add_argument('--epochs', default=20)
    parser.add_argument('--print_interval', default=50)
    parser.add_argument('--save_interval', default=10)
    args = parser.parse_args()
    
    args.root_dir = os.path.join(args.root_dir, args.dataset)
    args.save_dir = os.path.join(args.save_dir, f'{args.dataset}_features', 'finetune')

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    if not args.resume:
        model = finetune_sketch_vit(args, feature_extractor)
    else:
        if args.dataset == 'sketchy':
            num_labels = 19
        elif args.dataset == 'tu_berlin':
            num_labels = 21
        elif args.dataset == 'quickdraw':
            num_labels = 24
        pretrained_model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=num_labels
        )
        model = SketchViT(args, pretrained_model)
        checkpoint = torch.load(args.model_dir, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f'Loaded model saved at epoch {checkpoint["epoch"]} from checkpoint: {args.model_dir}')


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

    model = model.cuda()
    model.eval()
    with torch.no_grad():
        extract_sketch_features(args, feature_extractor, model)