import os
import random
import pprint
import numpy as np
import more_itertools as mit
from datetime import timedelta
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
from lib.utils.misc import cur_time, save_jsonl, save_json, AverageMeter
from lib.utils.model_utils import count_parameters
from lib.utils.logger import setup_logger
from lib.utils.box_utils import box_cxcywh_to_xyxy
from lib.evaluate.eval import eval_results
from lib.configs import args


def set_seed(seed, use_cuda=True):
    # fix seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def eval_setup(logger):
    use_cuda = True if torch.cuda.is_available() and args.use_gpu else False
    if args.seed:
        set_seed(args.seed, use_cuda)

    cudnn.benchmark = True
    cudnn.deterministic = False

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
    # model = DDP(model, delay_allreduce=True)

    # loss criterion
    criterion = build_loss(args).to(local_rank)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(local_rank))
        state_dict = checkpoint['model']

        if 'module' in list(state_dict.keys())[0]:
            keys = state_dict.keys()
            values = state_dict.values()
            new_keys = []
            for key in keys:
                new_key = key[7:]    # remove the 'module.'
                new_keys.append(new_key)

            from collections import OrderedDict
            new_dict = OrderedDict(list(zip(new_keys, values)))
            model.load_state_dict(new_dict)
        else:
            model.load_state_dict(checkpoint['model'])
        logger.info(f'Loaded model saved at iter {checkpoint["iter"]} from checkpoint: {args.resume}')
    else:
        logger.warning('If you intend to evaluate the model, please specify --resume with ckpt path')

    return model, criterion


def eval_post_processing(results, results_filename, logger, local_rank):
    logger.info('Saving/Evaluating results')
    results_path = os.path.join(args.results_dir, results_filename)
    save_jsonl(results, results_path)

    metrics = eval_results(
        results,
        verbose=args.debug,
        logger=logger
    )
    save_metrics_path = results_path.replace('.jsonl', '_metrics.json')
    save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
    latest_file_paths = [results_path, save_metrics_path]

    return metrics, latest_file_paths


@torch.no_grad()
def get_results(model, eval_loader, criterion, local_rank):
    model.eval()
    criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    results = []
    for batch in tqdm(eval_loader,
                      desc='Evaluation',
                      total=len(eval_loader)):
        batched_inputs, targets = batch
        model_inputs = prepare_batch_inputs(batched_inputs, local_rank, non_blocking=args.pin_memory)
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict['loss_overall'] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        prob = F.softmax(outputs['pred_logits'], -1)  # (batch_size, #queries, #classes=2)
        scores = prob[..., 0]  # (batch_size, #queries) foreground label is 0
        pred_boxes = outputs['pred_boxes']  # (batch_size, #queries, 4)

        # compose predictions
        for target, boxes, score in zip(targets,
                                        pred_boxes.cpu(),
                                        scores.cpu()):
            frame_idxs = list(target['bboxes'].keys())
            # w, h = target['shape']
            
            # cxcywh to xyxy
            boxes = torch.clamp(box_cxcywh_to_xyxy(boxes), min=0, max=1)
            # boxes[..., [0, 2]] *= w
            # boxes[..., [1, 3]] *= h

            # (#queries, 5), [xmin(float), ymin(float), xmax(float), xmax(float), score(float)]
            preds = torch.cat([boxes, score[:, None]], dim=1)

            # split predictions with #frames
            preds = preds.chunk(args.num_frames, dim=0)  # [#frames x (batch_size, #queries_per_frame, 4)]
            for preds_per_frame, fidx in zip(preds, frame_idxs):
                if not args.no_sort_results:
                    # sort by objectness score
                    sorted_preds = sorted(preds_per_frame, key=lambda x: x[4], reverse=True)
                sorted_preds = [[float(f'{e:.4f}') for e in row] for row in sorted_preds]
                gt_boxes = [{'track_id': id_bbox['track_id'],
                             'bbox': box_cxcywh_to_xyxy(id_bbox['bbox']).tolist()} \
                             for id_bbox in target['bboxes'][fidx]]
                result = dict(
                    video=target['video'],
                    sketch=target['sketch'],
                    shape=target['size'],
                    frame=fidx,
                    gt_boxes=gt_boxes,
                    pred_boxes=sorted_preds,
                )
                results.append(result)

        if args.debug:
            break

    return results, loss_meters


def inference(model, eval_loader, results_filename, criterion, local_rank, logger=None):
    model.eval()
    criterion.eval()

    results, loss_meters = get_results(model, eval_loader, criterion, local_rank)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    if args.no_sort_results:
        results_filename = results_filename.replace(".jsonl", "_unsorted.jsonl")

    metrics, latest_file_paths = eval_post_processing(
        results, results_filename, logger, local_rank)
    return metrics, loss_meters, latest_file_paths


def test(logger, run=None):
    model, criterion = eval_setup(logger)
    test_loader = build_dataloader(args, phase='test')
    results_filename = f'{cur_time()}_{args.video_dataset}_{args.sketch_dataset}_{args.sketch_head}_{args.backbone}_' \
                       f'{args.num_layers}l_{args.num_frames}f_{args.num_queries}q_' \
                       f'{args.set_cost_bbox}_{args.set_cost_giou}_{args.set_cost_class}_test.jsonl'
    logger.info("Start inference...")
    with torch.no_grad():
        metrics, eval_loss_meters, latest_file_paths = \
            inference(model, test_loader, results_filename, criterion, local_rank, logger=logger)

    # test log
    if local_rank == 0:
        if run:
            for k, v in eval_loss_meters.items():
                run[f"Test/{k}"].log(v.avg) 
            for k, v in metrics["brief"].items():
                run[f"Test/{k}"].log(float(v))
        logger.info(f'metrics {pprint.pformat(metrics["brief"], indent=4)}')


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
    logger = setup_logger('SVOL_eval', args.log_dir, distributed_rank=local_rank, filename=cur_time()+"_eval.txt")
    logger.info(f"=> current gpu: {local_rank} | total gpus: {world_size}")

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
    
    test(logger, run=run)

    if local_rank == 0 and args.use_neptune:
        run.stop()