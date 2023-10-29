import argparse
from lib.utils.misc import dict_to_markdown
from lib.utils.comm import get_rank


# meta config
parser = argparse.ArgumentParser(description='Sketch Localization Transformer')
parser.add_argument('--root', type=str, default='/mnt/server15_hard2/sangmin/data/svol/',
                    help='root directory of dataset')
parser.add_argument('--anno_root', type=str, default='/mnt/server15_hard2/sangmin/data/svol/annos/',
                    help='root directory of annotations')
parser.add_argument('--video_dataset', type=str, default='imagenet_vid')
parser.add_argument('--sketch_dataset', type=str, default='sketchy',
                    choices=['sketchy', 'tu_berlin', 'quickdraw'])
parser.add_argument('--results_dir', type=str, default='results',
                    help='directory for saving results')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1). if seed=0, seed is not fixed.')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many iters to wait before logging training status')
parser.add_argument('--val_interval', type=int, default=1000, metavar='N',
                    help='how many iters to wait before validation')
parser.add_argument('--save_interval', type=int, default=-1, metavar='N',
                    help='how many iters to wait before saving a model, use -1 to disable save')
parser.add_argument('--no_gpu', dest='use_gpu', action='store_false',
                    help='disable use of gpu')
parser.add_argument('--debug', action='store_true',
                    help='debug (fast) mode, break all loops, do not load all data into memory.')
parser.add_argument('--eval_untrained', action='store_true',
                    help='Evaluate on untrained model')
parser.add_argument('--log_dir', type=str, default='logs',
                    help='directory for saving logs')
parser.add_argument('--checkpoint', type=str, default='./save',
                    help='dir to save checkpoint')
parser.add_argument('--resume', type=str, default=None,
                    help='checkpoint path to resume or evaluate, without --resume_all this only load model weights')
parser.add_argument('--resume_all', action='store_true',
                    help='if --resume_all, load optimizer/scheduler/epoch as well')
parser.add_argument('--use_neptune', action='store_true',
                    help='enable use of neptune for logging purpose')


# distributed config
parser.add_argument('--dist-backend', type=str, default='nccl', 
                    choices=['nccl', 'gloo'],
                    help='distributed backend')
parser.add_argument('--use_amp', type=bool, default=True,
                    help='enable mixed precision training')
parser.add_argument('--sync_bn', action='store_true',
                    help='enabling apex sync BN.')
parser.add_argument('--channels-last', type=bool, default=False)
parser.add_argument('--opt-level', type=str, default='O0',
                    help='O0: "Pure FP32", '
                         'O1: "Official mixed precision recipe (recommended)", '
                         'O2: "Almost FP16", '
                         'O3: "Pure FP16"')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None,
                    help='only applicable for O2 and O3.')
parser.add_argument('--loss-scale', type=str, default=None,
                    help='if opt-level == O0 or O3: loss-scale=1.0; '
                         'if opt-level == O1 or O2: loss-scale="dynamic".')


# training config
parser.add_argument('--start_iter', type=int, default=None,
                    help='if None, will be set automatically when using --resume_all')
parser.add_argument('--num_iters', type=int, default=50000,
                    help='number of iters to run')
parser.add_argument('--early_stop_patience', type=int, default=10,
                    help='number of times to wait for early stop, use -1 to disable early stop')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--lr_drop_step', type=int, default=20000,
                    help='drop learning rate to 1/10 every lr_drop_step iters')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay (default=0.0001)')
parser.add_argument('--optimizer', type=str, default='adamw',
                    help='which optimizer to use (e.g., sgd, adam, adamw)')
parser.add_argument('--scheduler', type=str, default='steplr',
                    help='which scheduler to use')
parser.add_argument('--freeze_backbone', action='store_true',
                    help='optimize backbone architecture while training.')
parser.add_argument('--zeroshot_dataset_eval', action='store_true',
                    help='evaluate zero-shot transfer to unseen dataset (sketch-style).')
parser.add_argument('--zeroshot_category_eval', action='store_true',
                    help='evaluate zero-shot transfer to unseen classes.')
parser.add_argument('--unified_sketch_dataset', action='store_true',
                    help='use all sketch datasets as a whole.')


# data config
parser.add_argument('--bs', type=int, default=16, # FIXME
                    help='batch size')
parser.add_argument('--eval_bs', type=int, default=16, # FIXME
                    help='batch size at inference, for query')
parser.add_argument('--num_workers', type=int, default=4,
                    help='num subprocesses used to load the data, 0: use main process')
parser.add_argument('--no_pin_memory', dest='pin_memory', action='store_false',
                    help='No use of pin_memory for data loading.'
                         'If pin_memory=True, the data loader will copy Tensors into CUDA pinned memory before returning them.')
parser.add_argument('--num_frames', default=32, type=int,
                    help='number of input frames (i.e., sequence length) to transformer encoder.')
parser.add_argument('--num_input_sketches', default=1, type=int,
                    help='number of sketches to predict.')
parser.add_argument('--tight_frame_sampling', action='store_true',
                    help='if True, enable duplicate sampling to match the temporal size as "num_frames" for batching purpose.')
parser.add_argument('--aspect_ratio_grouping', type=bool, default=False)


# model config
parser.add_argument('--sketch_head', type=str, default='svanet',
                    choices=['svanet', 'sketch_detr'],
                    help='sketch heads')
parser.add_argument('--backbone', type=str, default='vit',
                    choices=['vit', 'resnet', 's3d'],
                    help='backbone')
parser.add_argument('--hidden_dim', type=int, default=256,
                    help='hidden dimension of Transformer')
parser.add_argument('--nheads', type=int, default=8,
                    help='number of Transformer attention heads')
parser.add_argument('--num_layers', type=int, default=4,
                    help='Number of encoding layers in the transformer')
parser.add_argument('--num_queries', default=320, type=int,
                    help='Total number of learnable queries across the video')
parser.add_argument('--num_queries_per_frame', default=10, type=int,
                    help='Number of queries per frame')
parser.add_argument('--input_dropout', default=0.4, type=float,
                    help='Dropout applied to input')
parser.add_argument('--use_sketch_pos', default=True, type=bool,
                    help='enable position_embedding for sketch.')
parser.add_argument('--n_input_proj', type=int, default=2,
                    help='#layers to encoder input')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout applied in the transformer')
parser.add_argument('--dim_feedforward', type=int, default=1024,
                    help='Intermediate size of the feedforward layers in the transformer blocks')
parser.add_argument('--pre_norm', action='store_true',
                    help='apply normalize before attention')
parser.add_argument('--sketch_position_embedding', default='sine', type=str,
                    choices=['trainable', 'sine', 'learned'],
                    help='Type of positional embedding to use on top of the sketch features')
parser.add_argument('--video_position_embedding', default='sine', type=str,
                    choices=['trainable', 'sine', 'learned'],
                    help='Type of positional embedding to use on top of the image features')


# loss config
parser.add_argument('--matcher', type=str, default='per_frame_matcher',
                    choices=['per_frame_matcher', 'video_matcher'],
                    help='design choices of matcher')
parser.add_argument('--set_cost_bbox', default=5, type=int,
                    help='L1 bbox coefficient in the matching cost')
parser.add_argument('--set_cost_giou', default=1, type=int,
                    help='giou bbox coefficient in the matching cost')
parser.add_argument('--set_cost_class', default=2, type=int,
                    help='Class coefficient in the matching cost')
parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                    help='Disable auxiliary decoding losses (loss at each layer)')
parser.add_argument('--eos_coef', default=0.1, type=float,
                    help='Relative classification weight of the no-object class')


# evaluation config
parser.add_argument('--bbox_type', default='cxcywh', type=str,
                    choices=['cxcywh', 'xyxy'],
                    help='Type of bbox (cxcywh: center-width-height / xyxy: topleft-bottomright)')
parser.add_argument('--no_sort_results', action='store_true',
                    help='do not sort results, use this for bbox visualization')


# feature plot config
parser.add_argument('--vis_mode', type=str, default=None,
                    help='choose embedding projector (e.g.,umap, tsne)')
parser.add_argument('--use_vis_mean', action='store_true',
                    help='apply means to embedding values of frames')
parser.add_argument('--n_neighbor', default=15, type=int,
                    help='local versus global structure balance parameter for umap')

args = parser.parse_args()

# Display settings
if get_rank() == 0:
     print(dict_to_markdown(vars(args), max_str_len=120))
