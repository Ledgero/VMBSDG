import os
from loguru import logger
import torch
from argparse import ArgumentParser
from warnings import warn
from torch import nn
from torch.utils.data import ConcatDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import wandb
import warnings
from tqdm import tqdm # Import tqdm
from romatch.benchmarks import MegadepthDenseBenchmark
from romatch.datasets.megadepth import MegadepthBuilder
from romatch.src.lightning.data import MultiSceneDataModule
from romatch.src.config.default import get_cfg_defaults
from romatch.losses.robust_loss import RobustLosses
from romatch.benchmarks import MegaDepthPoseEstimationBenchmark, MegadepthDenseBenchmark, HpatchesHomogBenchmark
from romatch.src.utils.profiler import build_profiler, PassThroughProfiler # Import profiler

try:
    import yagmail # Import yagmail for sending emails
except ImportError:
    yagmail = None
    logger.warning("yagmail not installed. Email notifications will be disabled.")

import traceback # Import traceback for error details

from romatch.train.train import train_k_epochs
from romatch.models.matcher import *
from romatch.models.transformer import Block, TransformerDecoder, MemEffAttention
from romatch.models.encoders import *
from romatch.checkpointing import CheckPoint
import time # Import time

resolutions = {"low":(448, 448), "medium":(14*8*5, 14*8*5), "high":(14*8*6, 14*8*6)}
DATA_PATH = "/home/ozq/LoFT/LoFTR/data/megadepth"
warnings.filterwarnings('ignore', category=UserWarning)

def send_with_retry(to_list, subject, contents, attachments=None,
                    max_retry=3, base_backoff=2):
    """
    带重试的发送函数
    :param max_retry: 最多重试次数
    :param base_backoff: 基础退避秒数（指数增长）
    """
    attempt = 0
    while attempt < max_retry:
        try:
            yag = yagmail.SMTP(os.environ.get("GMAIL_USER"), 
                    password=os.environ.get("GMAIL_PASS"),
                    host='smtp.qq.com',
                    port=465)
            yag.send(to=to_list, subject=subject, contents=contents, attachments=attachments)
            logger.info(f"Email sent successfully to {to_list}")
            return   # 成功则直接返回
        except Exception as e:
            attempt += 1
            if attempt >= max_retry:
                print('最终发送失败：', e)
                raise
            wait = base_backoff ** attempt   # 2, 4, 8 秒...
            print(f'第 {attempt} 次发送失败：{e},{wait} 秒后重试...')
            time.sleep(wait)

def get_model(pretrained_backbone=True, resolution = "medium", **kwargs):
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    gp_dim = 512
    feat_dim = 512
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64
    coordinate_decoder = TransformerDecoder(
        nn.Sequential(*[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]), 
        decoder_dim, 
        cls_to_coord_res**2 + 1,
        is_classifier=True,
        amp = True,
        pos_enc = False,)
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True
    
    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512+128+(2*7+1)**2,
                2 * 512+128+(2*7+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius = 7,
                corr_in_other = True,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "8": ConvRefiner(
                2 * 512+64+(2*3+1)**2,
                2 * 512+64+(2*3+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius = 3,
                corr_in_other = True,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "4": ConvRefiner(
                2 * 256+32+(2*2+1)**2,
                2 * 256+32+(2*2+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius = 2,
                corr_in_other = True,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "2": ConvRefiner(
                2 * 64+16,
                128+16,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "1": ConvRefiner(
                2 * 9 + 6,
                24,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks = hidden_blocks,
                displacement_emb = displacement_emb,
                displacement_emb_dim = 6,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"16": gp16})
    proj16 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512))
    proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
    proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
    proj = nn.ModuleDict({
        "16": proj16,
        "8": proj8,
        "4": proj4,
        "2": proj2,
        "1": proj1,
        })
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    decoder = Decoder(coordinate_decoder, 
                      gps, 
                      proj, 
                      conv_refiner, 
                      detach=True, 
                      scales=["16", "8", "4", "2", "1"], 
                      displacement_dropout_p = displacement_dropout_p,
                      gm_warp_dropout_p = gm_warp_dropout_p)
    h,w = resolutions[resolution]
    encoder = CNNandDinov2(
        cnn_kwargs = dict(
            pretrained=pretrained_backbone,
            amp = True),
        amp = True,
    )
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w,**kwargs)
    return matcher

def train(args):
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group('nccl')
        gpus = int(os.environ["WORLD_SIZE"])
        rank = dist.get_rank()
        print(f"Start running DDP on rank {rank}")
    else:
        gpus = 1
        rank = 0
        print("Running in single GPU mode.")

    try:
        #torch._dynamo.config.verbose=True
        # create model and move it to GPU with id rank
        device_id = rank % torch.cuda.device_count()
        romatch.LOCAL_RANK = device_id
        torch.cuda.set_device(device_id)
        
        resolution = args.train_resolution
        wandb_log = not args.dont_log_wandb
        experiment_name = os.path.splitext(os.path.basename(__file__))[0]
        wandb_mode = "online" if wandb_log and rank == 0 else "disabled"
        wandb.init(project="romatch", entity=args.wandb_entity, name="1014test1", reinit=False, mode = wandb_mode)
        #checkpoint_dir = "workspace/checkpoints/"
        checkpoint_dir = "checkpoints/"
        h,w = resolutions[resolution]
        
        # Initialize profiler
        profiler = build_profiler(args.profiler_name, save_dir=checkpoint_dir) if args.profiler_name else PassThroughProfiler()
        
        model = get_model(pretrained_backbone=True, resolution=resolution, attenuate_cert = False).to(device_id)
        # Num steps
        global_step = 0
        batch_size = args.batch_size
        step_size = gpus*batch_size
        romatch.STEP_SIZE = step_size
        
        # Data
        mega = MegadepthBuilder(data_root=DATA_PATH, loftr_ignore=True, imc21_ignore = True, loftr=args.loftr)
        use_horizontal_flip_aug = True
        rot_prob = 0
        depth_interpolation_mode = "bilinear"
        if not args.loftr:
            logger.info("Using original LoFTR scenes loading way")
            megadepth_train1 = mega.build_scenes(
                split="train_loftr", min_overlap=0.01, shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug, rot_prob = rot_prob,
                ht=h,wt=w,
            )
            megadepth_train2 = mega.build_scenes(
                split="train_loftr", min_overlap=0.35, shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug, rot_prob = rot_prob,
                ht=h,wt=w,
            )
        else:
            config = get_cfg_defaults()
            if args.main_cfg_path is not None:
                config.merge_from_file(args.main_cfg_path) #merge_from_file(): 将用户指定的配置文件内容合并到默认配置中
            config.merge_from_file(args.data_cfg_path)
            data_module = MultiSceneDataModule(args, config)
            megadepth_train1 = data_module._setup_dataset(
                data_root=config.DATASET.TRAIN_DATA_ROOT,
                split_npz_root=config.DATASET.TRAIN_NPZ_ROOT,
                scene_list_path=config.DATASET.TRAIN_LIST_PATH,
                intri_path=None,
                mode='train',
                min_overlap_score=0.01,
                pose_dir=None,
                loftr=args.loftr,
                split=True,shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug, rot_prob = rot_prob,
                ht=h,wt=w,
            )
            megadepth_train2 = data_module._setup_dataset(
                data_root=config.DATASET.TRAIN_DATA_ROOT,
                split_npz_root=config.DATASET.TRAIN_NPZ_ROOT,
                scene_list_path=config.DATASET.TRAIN_LIST_PATH,
                intri_path=None,
                mode='train',
                min_overlap_score=0.35,
                pose_dir=None,
                loftr=args.loftr,
                split=True,shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug, rot_prob = rot_prob,
                ht=h,wt=w,
            )
        megadepth_train = ConcatDataset(megadepth_train1 + megadepth_train2)
        mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)
        logger.info(f"Total training dataset size (megadepth_train): {len(megadepth_train)}")
        logger.info(f"Weighted sampler distribution size (mega_ws): {len(mega_ws)}")
        # Loss and optimizer
        depth_loss = RobustLosses(
            ce_weight=0.01, 
            local_dist={1:4, 2:4, 4:8, 8:8},
            local_largest_scale=8,
            depth_interpolation_mode=depth_interpolation_mode,
            alpha = 0.5,
            c = 1e-4,)
        parameters = [
            {"params": model.encoder.parameters(), "lr": romatch.STEP_SIZE * 5e-6 / 8},
            {"params": model.decoder.parameters(), "lr": romatch.STEP_SIZE * 1e-4 / 8},
        ]
        optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[(9*N/romatch.STEP_SIZE)//10])
        megadense_benchmark = MegaDepthPoseEstimationBenchmark(DATA_PATH, batch_size=batch_size)
        
        checkpointer = CheckPoint(checkpoint_dir, experiment_name, top_k=3, exp_name=args.exp_name)
        model, optimizer, lr_scheduler, global_step = checkpointer.load(model, optimizer, lr_scheduler, global_step, exp_name=args.exp_name)
        romatch.GLOBAL_STEP = global_step
        ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters = False, gradient_as_bucket_view=True)
        grad_scaler = torch.amp.GradScaler('cuda',growth_interval=1_000_000)
        grad_clip_norm = 0.01
        
        # initial_global_step = romatch.GLOBAL_STEP
        # cumulative_block_duration = 0.0
        # completed_blocks_count = 0
        
        # Initializing for epoch based training
        start_epoch = 0
        end_epoch = args.max_epochs

        # Data loader for epoch-based training
        mega_dataloader = torch.utils.data.DataLoader(
            megadepth_train,
            batch_size = batch_size,
            sampler = mega_ws, # Use mega_ws as sampler for epoch-based training
            num_workers = args.num_workers,
            shuffle = False, # Set shuffle to False when using a sampler
        )

        train_k_epochs(
            start_epoch, end_epoch, mega_dataloader, ddp_model, depth_loss, optimizer, lr_scheduler,
            grad_scaler=grad_scaler, grad_clip_norm=grad_clip_norm, profiler=profiler,
            megadense_benchmark=megadense_benchmark, checkpointer=checkpointer, experiment_name=experiment_name
        )

        if romatch.RANK == 0:
            logger.info(f"="*20+"\nTraining Completed!\n")

        # Print profiler summary at the end of training
        if profiler and rank == 0:
            print("\n" + profiler.summary())
            # total_duration = time.time() - initial_global_step # Use initial_global_step for overall duration
            # logger.info(f"Total training duration: {total_duration:.2f} seconds")

        if rank == 0 and args.mail:
            send_with_retry(to_list=['1559457127@qq.com'],
                            subject=f"Training Completed Successfully: {experiment_name}",
                            contents=[f"Training for {experiment_name} finished successfully."])

    except Exception as e:
        if rank == 0 and args.mail:
            error_message = traceback.format_exc()
            logger.error(f"Training encountered an error: {error_message}")
            send_with_retry(to_list=['1559457127@qq.com'],
                            subject=f"Training Error: {experiment_name}",
                            contents=[f"Training for {experiment_name} failed with an error:\n{error_message}"])
        raise e # Re-raise the exception to terminate the program
    

def test_mega_8_scenes(model, name):
    mega_8_scenes_benchmark = MegaDepthPoseEstimationBenchmark("data/megadepth",
                                                scene_names=['mega_8_scenes_0019_0.1_0.3.npz',
                                                    'mega_8_scenes_0025_0.1_0.3.npz',
                                                    'mega_8_scenes_0021_0.1_0.3.npz',
                                                    'mega_8_scenes_0008_0.1_0.3.npz',
                                                    'mega_8_scenes_0032_0.1_0.3.npz',
                                                    'mega_8_scenes_1589_0.1_0.3.npz',
                                                    'mega_8_scenes_0063_0.1_0.3.npz',
                                                    'mega_8_scenes_0024_0.1_0.3.npz',
                                                    'mega_8_scenes_0019_0.3_0.5.npz',
                                                    'mega_8_scenes_0025_0.3_0.5.npz',
                                                    'mega_8_scenes_0021_0.3_0.5.npz',
                                                    'mega_8_scenes_0008_0.3_0.5.npz',
                                                    'mega_8_scenes_0032_0.3_0.5.npz',
                                                    'mega_8_scenes_1589_0.3_0.5.npz',
                                                    'mega_8_scenes_0063_0.3_0.5.npz',
                                                    'mega_8_scenes_0024_0.3_0.5.npz'])
    mega_8_scenes_results = mega_8_scenes_benchmark.benchmark(model, model_name=name)
    print(mega_8_scenes_results)
    json.dump(mega_8_scenes_results, open(f"results/mega_8_scenes_{name}.json", "w"))

def test_mega1500(model, name, batch_size=4):
    os.makedirs("results", exist_ok=True)          # ← 新增
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark(DATA_PATH, batch_size=batch_size)
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"results/mega1500_{name}.json", "w"))

def test_mega_dense(model, name):
    megadense_benchmark = MegadepthDenseBenchmark("data/megadepth", num_samples = 1000)
    megadense_results = megadense_benchmark.benchmark(model)
    json.dump(megadense_results, open(f"results/mega_dense_{name}.json", "w"))
    
def test_hpatches(model, name):
    hpatches_benchmark = HpatchesHomogBenchmark("data/hpatches")
    hpatches_results = hpatches_benchmark.benchmark(model)
    json.dump(hpatches_results, open(f"results/hpatches_{name}.json", "w"))


if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1" # For BF16 computations
    os.environ["OMP_NUM_THREADS"] = "16"
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    warn('Current version of romatch is not tested for training, use at your own risk.')
    import romatch
    parser = ArgumentParser()
    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument("--only_test", action='store_true')
    parser.add_argument("--debug_mode", action='store_true')
    parser.add_argument("--dont_log_wandb", action='store_true')
    parser.add_argument("--train_resolution", default='medium')
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wandb_entity", required = False, default="own_ozq")
    parser.add_argument("--exp_name", required = False, default="roma")
    parser.add_argument("--loftr", action='store_true')
    parser.add_argument("--max_epochs", default=30, type=int) # New argument for max epochs
    parser.add_argument(
        '--profiler_name', type=str, default="pytorch", help='options: [inference, pytorch], or leave it unset') # Add profiler_name argument
    parser.add_argument("--mail", action='store_true')
    args, _ = parser.parse_known_args()
    romatch.DEBUG_MODE = args.debug_mode
    if not args.only_test:
        train(args)
