import argparse
import glob
import os
from pathlib import Path
from typing import List

import yaml

from munch import munchify
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torchvision.io import read_image
from torchvision.utils import save_image

from util import set_seed, process_text
from sd3_sampler import get_solver
from functions.degradation import get_degradation
from functions.dataloader import get_dataloader
from utils.eval_util import compute_psnr_folder, compute_ssim_folder, compute_fid_folder, print_stats

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None


def init_distributed(args):
    if not args.use_ddp:
        return 0, 1, 0

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not dist.is_initialized():
        dist.init_process_group(backend=args.ddp_backend)

    return rank, world_size, local_rank

@torch.no_grad
def precompute(args, prompts:List[str], solver) -> List[torch.Tensor]:
    prompt_emb_set = []
    pooled_emb_set = []

    num_samples = args.num_samples if args.num_samples > 0 else len(prompts)
    for prompt in prompts[:num_samples]:
        prompt_emb, pooled_emb = solver.encode_prompt(prompt, batch_size=1)
        prompt_emb_set.append(prompt_emb)
        pooled_emb_set.append(pooled_emb)

    return prompt_emb_set, pooled_emb_set


@torch.no_grad()
def compute_fid_rank0_no_sync(gt_dir: Path, pred_dir: Path, device: torch.device) -> float:
    if FrechetInceptionDistance is None:
        raise RuntimeError("FID requires torchmetrics image extras (torch-fidelity).")

    gt_paths = sorted(glob.glob(str(gt_dir / "*.png")))
    pred_paths = sorted(glob.glob(str(pred_dir / "*.png")))
    if len(gt_paths) == 0 or len(pred_paths) == 0:
        raise RuntimeError("No images found for FID computation.")
    if len(gt_paths) != len(pred_paths):
        raise RuntimeError(f"Count mismatch: gt={len(gt_paths)} vs pred={len(pred_paths)}")

    try:
        fid = FrechetInceptionDistance(feature=2048, sync_on_compute=False).to(device)
    except TypeError:
        fid = FrechetInceptionDistance(feature=2048).to(device)
        if hasattr(fid, "sync_on_compute"):
            fid.sync_on_compute = False

    for p in gt_paths:
        img = read_image(p).to(device)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] > 3:
            img = img[:3]
        fid.update(img.unsqueeze(0), real=True)

    for p in pred_paths:
        img = read_image(p).to(device)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] > 3:
            img = img[:3]
        fid.update(img.unsqueeze(0), real=False)

    return float(fid.compute().item())

def run(args):
    rank, world_size, local_rank = init_distributed(args)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if args.use_ddp else "cuda")
        if args.use_ddp:
            torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    if args.clean_workdir and rank == 0:
        for subdir in ['input', 'recon', 'label']:
            for fp in args.workdir.joinpath(subdir).glob('*.png'):
                fp.unlink()

    if args.use_ddp:
        dist.barrier()

    # load solver
    solver = get_solver(args.method)
    local_batch_size = args.batch_size_per_gpu
    
    cfg_data = yaml.safe_load(open(args.dataset))
    dataloader_cfg = dict(cfg_data["dataloader"])
    dataloader_cfg.pop("device", None)
    dataloader_cfg["batch_size"] = local_batch_size

    base_loader = get_dataloader(**dataloader_cfg)
    dataset = base_loader.dataset

    total_samples = len(dataset)
    if args.num_samples > 0:
        total_samples = min(total_samples, args.num_samples)

    rank_indices = list(range(rank, total_samples, world_size))
    rank_dataset = Subset(dataset, rank_indices)
    loader = DataLoader(
        rank_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=dataloader_cfg.get("num_workers", 4),
        drop_last=dataloader_cfg.get("drop_last", False),
        pin_memory=dataloader_cfg.get("pin_memory", True),
    )

    # load text prompts
    prompts = process_text(prompt=args.prompt, prompt_file=args.prompt_file) if (args.prompt is not None or args.prompt_file is not None) else None
    solver.text_enc_1.to(device)
    solver.text_enc_2.to(device)
    solver.text_enc_3.to(device)

    use_precomputed_prompt = args.efficient_memory and prompts is not None
    if use_precomputed_prompt:
        # precompute text embedding and remove encoders from GPU
        # This will allow us 1) fast inference 2) with lower memory requirement (<24GB)
        with torch.no_grad():
            prompt_emb_set, pooled_emb_set = precompute(args, prompts, solver)
            null_emb, null_pooled_emb = solver.encode_prompt([''], batch_size=1)

        del solver.text_enc_1
        del solver.text_enc_2
        del solver.text_enc_3
        torch.cuda.empty_cache()

        prompt_embs = [[x, y] for x, y in zip(prompt_emb_set, pooled_emb_set)]
        null_embs = [null_emb, null_pooled_emb]
    else:
        prompt_embs = None
        null_embs = None

    if rank == 0:
        print("Prompts are processed.")

    solver.vae.to(device)
    solver.transformer.to(device)

    # problem setup
    deg_config = munchify({
        'channels': 3,
        'image_size': args.img_size,
        'deg_scale': args.deg_scale
        })
    operator = get_degradation(args.task, deg_config, device)

    # solve problem (true batched inference)
    local_ptr = 0
    pbar = tqdm(loader, desc=f"Solving(rank{rank})", disable=(rank != 0))
    for batch in pbar:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            imgs, batch_prompts = batch
        else:
            imgs, batch_prompts = batch, None

        imgs = imgs.to(device)

        batch_size = imgs.size(0)
        batch_global_indices = rank_indices[local_ptr: local_ptr + batch_size]
        local_ptr += batch_size
        imgs = imgs * 2 - 1

        y = operator.A(imgs)
        y = y + 0.03 * torch.randn_like(y)

        if batch_prompts is not None:
            prompt_texts = list(batch_prompts)
            prompt_emb = None
        else:
            if prompts is None:
                prompt_texts = [""] * batch_size
                prompt_emb = None
            elif len(prompts) == 1:
                prompt_texts = [prompts[0]] * batch_size
                prompt_emb = prompt_embs[0] if use_precomputed_prompt else None
            else:
                prompt_texts = [prompts[i] for i in batch_global_indices]
                if len(prompt_texts) != batch_size:
                    raise RuntimeError("Not enough prompts for current batch size.")
                if use_precomputed_prompt:
                    batch_prompt_emb = [prompt_embs[i] for i in batch_global_indices]
                    prompt_emb = [
                        torch.cat([x[0] for x in batch_prompt_emb], dim=0),
                        torch.cat([x[1] for x in batch_prompt_emb], dim=0),
                    ]
                else:
                    prompt_emb = None

        null_emb_for_batch = None
        if use_precomputed_prompt and null_embs is not None:
            null_emb_for_batch = [
                null_embs[0].expand(batch_size, -1, -1).contiguous(),
                null_embs[1].expand(batch_size, -1).contiguous(),
            ]

        out = solver.sample(measurement=y,
                            operator=operator,
                            prompts=prompt_texts,
                            NFE=args.NFE,
                            img_shape=(args.img_size, args.img_size),
                            task=args.task,
                            batch_size=batch_size,
                            prompt_emb=prompt_emb,
                            null_emb=null_emb_for_batch
                            )

        for b in range(batch_size):
            img = imgs[b:b + 1]
            y_b = y[b:b + 1]
            out_b = out[b:b + 1]
            global_idx = batch_global_indices[b]

            save_image(operator.At(y_b).reshape(img.shape),
                       args.workdir.joinpath(f'input/{str(global_idx).zfill(4)}.png'),
                       normalize=True)
            save_image(out_b,
                       args.workdir.joinpath(f'recon/{str(global_idx).zfill(4)}.png'),
                       normalize=True)
            save_image(img,
                       args.workdir.joinpath(f'label/{str(global_idx).zfill(4)}.png'),
                       normalize=True)

    if args.use_ddp:
        dist.barrier()

    if rank == 0:
        psnr_rec = compute_psnr_folder(args.workdir.joinpath('label'), args.workdir.joinpath('recon'), device=device)
        ssim_rec = compute_ssim_folder(args.workdir.joinpath('label'), args.workdir.joinpath('recon'), device=device)
        try:
            if args.use_ddp:
                fid_rec = compute_fid_rank0_no_sync(
                    args.workdir.joinpath('label'),
                    args.workdir.joinpath('recon'),
                    device=device,
                )
            else:
                fid_rec = compute_fid_folder(args.workdir.joinpath('label'), args.workdir.joinpath('recon'), device=device)
        except RuntimeError as e:
            fid_rec = float('nan')
            print(f"[Warning] {e}")

        print("\n================== Evaluation (rank0) ==================")
        print("[GT vs recon]")
        print(f"  PSNR: {psnr_rec:.4f}")
        print(f"  SSIM: {ssim_rec:.6f}")
        print(f"  FID : {fid_rec:.4f}")
        print("========================================================\n")

    if args.use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # sampling params
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--NFE', type=int, default=28)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--img_size', type=int, default=768)

    # workdir params
    parser.add_argument('--workdir', type=Path, default='workdir/batch')

    # data params
    parser.add_argument('--img_path', type=Path)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--prompt_file', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="configs/DIV2K_train.yml")

    # problem params
    parser.add_argument('--task', type=str, default='sr_avgpool')
    parser.add_argument('--method', type=str, default='flowdps')
    parser.add_argument('--deg_scale', type=int, default=12)

    # solver params
    parser.add_argument('--step_size', type=float, default=15.0)
    parser.add_argument('--efficient_memory',default=False, action='store_true')
    parser.add_argument('--batch_size_per_gpu', type=int, default=2)
    parser.add_argument('--use_ddp', default=False, action='store_true', help='Use torchrun distributed sharding for inference.')
    parser.add_argument('--ddp_backend', type=str, default='nccl')
    parser.add_argument('--clean_workdir', default=False, action='store_true')
    args = parser.parse_args()


    # workdir creation and seed setup
    set_seed(args.seed)
    args.workdir.joinpath('input').mkdir(parents=True, exist_ok=True)
    args.workdir.joinpath('recon').mkdir(parents=True, exist_ok=True)
    args.workdir.joinpath('label').mkdir(parents=True, exist_ok=True)

    # run main script
    run(args)
