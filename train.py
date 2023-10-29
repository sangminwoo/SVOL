import os
import time
import random
import pprint
import numpy as np
from datetime import timedelta
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from lib.modeling.model import build_model
from lib.modeling.loss import build_loss
from lib.dataset.svol_dataset import prepare_batch_inputs
from lib.dataset.svol_dataloader import build_dataloader
from lib.utils.comm import get_rank, get_world_size, reduce_tensor
from lib.utils.misc import cur_time, AverageMeter
from lib.utils.model_utils import count_parameters
from lib.utils.logger import setup_logger
from lib.configs import args
from test import inference


def set_seed(seed, use_cuda=True):
    # fix seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_setup(logger):
    use_cuda = True if torch.cuda.is_available() and args.use_gpu else False
    if args.seed:
        set_seed(args.seed, use_cuda)

    if args.debug: # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    model = build_model(args)

    if args.sync_bn:
        import apex
        logger.info("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.to(device=local_rank, memory_format=memory_format)

    param_dicts = [{'params': [param for name, param in model.named_parameters() if param.requires_grad]}]
    #######################################################################################
    # Uncomment below to modify backbone optimization
    #######################################################################################
    # backbone_params = []
    # head_params = []
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         if 'backbone' in name:
    #             if args.freeze_backbone:
    #                 param.requires_grad = False
    #             backbone_params.append(param)
    #         if 'head' in name:
    #             head_params.append(param)

    # if len(backbone_params) > 0:
    #     param_dicts = [{'params':backbone_params}, {'params':head_params}]
    # else:
    #     param_dicts = [{'params':head_params}]
    #######################################################################################    

    # optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_dicts, lr=args.lr, weight_decay=args.wd)
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.wd)
    
    #######################################################################################
    # Uncomment below to modify backbone optimization
    #######################################################################################
    # if len(backbone_params) > 0:
    #     optimizer.param_groups[0]['lr'] = args.lr*0.1
    #     optimizer.param_groups[1]['lr'] = args.lr
    #######################################################################################

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale)

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    # By default, apex.parallel.DistributedDataParallel overlaps communication with
    # computation in the backward pass.
    # model = DDP(model)
    # delay_allreduce delays all communication to the end of the backward pass.
    model = DDP(model, delay_allreduce=True)

    # loss criterion
    criterion = build_loss(args).to(local_rank)

    # scheduler
    if args.scheduler == 'steplr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop_step)
    if args.scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.lr_drop_step], gamma=0.1)
    if args.scheduler == 'reducelronplateau':
        # TODO
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=1,
            threshold=0.5,
            verbose=True
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(local_rank))
        model.load_state_dict(checkpoint['model'])
        if args.resume_all:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            amp.load_state_dict(checkpoint['amp'])
            args.start_iter = checkpoint['iter'] + 1

        logger.info(f'Loaded model saved at iter {checkpoint["iter"]} from checkpoint: {args.resume}')

    return model, criterion, optimizer, lr_scheduler


def train_val(logger, run=None):
    model, criterion, optimizer, lr_scheduler = train_setup(logger)
    if local_rank == 0:
        logger.info(f'Model {model}')
        n_all, n_trainable, mem, mem_params, mem_bufs = count_parameters(model)
        if run:
            run[f"num_params"].log(n_all)
            run[f"num_trainable_params"].log(n_trainable)
            run[f"mem"].log(n_all)
            run[f"mem_params"].log(n_trainable) 
            run[f"mem_bufs"].log(n_all)
        logger.info(f'Start Training...')

    train_loader = build_dataloader(args, phase='train', distributed=False)
    #######################################################################################
    # TODO: switch the sketch dataset for zeroshot dataset evaluation
    if args.zeroshot_dataset_eval:
        # args.sketch_dataset = 'sketchy'
        # args.sketch_dataset = 'tu_berlin'
        args.sketch_dataset = 'quickdraw'
    #######################################################################################
    val_loader = build_dataloader(args, phase='val', distributed=False)
    
    # create checkpoint
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    if args.start_iter is None:
        start_iter = -1 if args.eval_untrained else 0
    else:
        start_iter = start_iter

    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    # for early stop purpose
    best_metric = 0 # np.inf
    early_stop_count = 0

    # let all processes sync up before the next iteration
    dist.barrier()

    tictoc = time.time()
    #######################################################################################
    # Train
    #######################################################################################
    for iter_i, batch in tqdm(enumerate(train_loader),
                              desc='Training',
                              total=len(train_loader)):
        optimizer.zero_grad()

        model.train()
        criterion.train()

        time_meters['dataloading_time'].update(time.time() - tictoc)
        tictoc = time.time()
        
        batched_inputs, targets = batch
        model_inputs = prepare_batch_inputs(batched_inputs, local_rank, non_blocking=args.pin_memory)
        time_meters['prepare_inputs_time'].update(time.time() - tictoc)
        tictoc = time.time()

        # compute output
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        time_meters['model_forward_time'].update(time.time() - tictoc)
        tictoc = time.time()

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # compute gradient and do gradient descent step
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()

        optimizer.step()
        time_meters['model_backward_time'].update(time.time() - tictoc)

        lr_scheduler.step()

        # Average losses across all GPU processes for logging purpose
        loss_dict['loss_overall'] = reduce_tensor(losses, world_size)

        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        if local_rank == 0:
            if iter_i % args.log_interval == 0:
                logger.info(
                    "Training Logs\n"
                    "[Iter] {iter:06d}\n"
                    "[Time]\n{time_stats}\n"
                    "[Loss]\n{loss_str}\n".format(
                        time_str=time.strftime("%Y-%m-%d %H:%M:%S"),
                        iter=iter_i,
                        time_stats="\n".join("\t> {} {:.4f}".format(k, v.avg) for k, v in time_meters.items()),
                        loss_str="\n".join(["\t> {} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()])
                    )
                )

                # train log
                if run:
                    for k, v in loss_meters.items():
                        run[f"Train/{k}"].log(v.avg)  

            #######################################################################################
            # Save
            #######################################################################################
            if args.save_interval > 0 and (iter_i + 1) % args.save_interval == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'amp': amp.state_dict(),
                    'iter': iter_i,
                    'args': args
                }
                torch.save(
                    checkpoint,
                    os.path.join(
                        args.checkpoint,
                        f'{iter_i:04d}_model_{args.video_dataset}_{args.sketch_dataset}_{args.sketch_head}_{args.backbone}_' \
                        f'{args.num_layers}l_{args.num_frames}f_{args.num_queries}q_' \
                        f'{args.set_cost_bbox}_{args.set_cost_giou}_{args.set_cost_class}.ckpt'
                    )
                )

        #######################################################################################
        # Validation
        #######################################################################################
        if (iter_i + 1) % args.val_interval == 0:
            with torch.no_grad():
                results_filename = f'{cur_time()}_{args.video_dataset}_{args.sketch_dataset}_{args.sketch_head}_{args.backbone}_' \
                                   f'{args.num_layers}l_{args.num_frames}f_{args.num_queries}q_' \
                                   f'{args.set_cost_bbox}_{args.set_cost_giou}_{args.set_cost_class}_val.jsonl'
                metrics, eval_loss_meters, _ = \
                    inference(model, val_loader, results_filename, criterion, local_rank, logger=logger)

            cur_metric = metrics['brief']['SVOL-full-mIoU@R1']
            # cur_metric = sum(metrics['brief'].values()) # TODO

            if local_rank == 0:
                logger.info(
                    "\n>>>>> Evalutation\n"
                    "[Iter] {iter:03d}\n"
                    "[Loss]\n{loss_str}\n"
                    "[Metrics]\n{metrics}\n".format(
                        time_str=time.strftime("%Y-%m-%d %H:%M:%S"),
                        iter=iter_i+1,
                        loss_str="\n".join(["\t> {} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                        metrics=pprint.pformat(metrics["brief"], indent=4)
                    )
                )

                # val log
                if run:
                    for k, v in eval_loss_meters.items():
                        run[f"Val/{k}"].log(v.avg) 

                    for k, v in metrics["brief"].items():
                        run[f"Val/{k}"].log(float(v))

            #######################################################################################
            # Save
            #######################################################################################
            if cur_metric > best_metric:
                early_stop_count = 0
                best_metric = cur_metric
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'amp': amp.state_dict(),
                    'iter': iter_i,
                    'args': args
                }
                if local_rank == 0:
                    torch.save(
                        checkpoint,
                        os.path.join(
                            args.checkpoint,
                            f'best_model_{args.video_dataset}_{args.sketch_dataset}_{args.sketch_head}_{args.backbone}_' \
                            f'{args.num_layers}l_{args.num_frames}f_{args.num_queries}q_' \
                            f'{args.set_cost_bbox}_{args.set_cost_giou}_{args.set_cost_class}.ckpt'
                        )
                    )
            #######################################################################################
            # Early Stop
            #######################################################################################
            else:
                early_stop_count += 1
                if args.early_stop_patience > 0 and early_stop_count > args.early_stop_patience:
                    logger.info(f'\n>>>>> Early Stop at Iter {iter_i+1} (best miou: {best_metric})\n')
                    break

        # let all processes sync up before the next iteration
        dist.barrier()

        tictoc = time.time()


if __name__ == '__main__':
    # Distributed
    dist.init_process_group(
        backend=args.dist_backend,
        init_method="env://",
        # timeout=timedelta(seconds=10)
    )
    local_rank = get_rank()
    world_size = get_world_size()
    # device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Logger
    logger = setup_logger('SVOL', args.log_dir, distributed_rank=local_rank, filename=cur_time()+"_train.txt")
    logger.info(f"Total GPUs: {world_size}")

    # Neptune
    run = None
    if args.use_neptune:
        import neptune.new as neptune

        # Neptune init
        if local_rank == 0:
            run = neptune.init(
                project='kaist-cilab/SVOL',
                api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjNGExZDUyYS05ODk0LTQ0MWUtOGNmYS0wYjNjM2Q0NGZlYzgifQ==',
            )

            # Neptune save args
            params = vars(args)
            run['parameters'] = params

    train_val(logger, run=run)
    
    if local_rank == 0 and args.use_neptune:
        run.stop()