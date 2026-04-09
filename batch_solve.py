import argparse
from pathlib import Path
from typing import List

import yaml

from munch import munchify
from tqdm import tqdm
import torch
from torchvision.utils import save_image

from util import set_seed, process_text
from sd3_sampler import get_solver
from functions.degradation import get_degradation
from functions.dataloader import get_dataloader
from utils.eval_util import compute_psnr_folder, compute_ssim_folder, compute_fid_folder, print_stats

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

def run(args):
    # load solver
    solver = get_solver(args.method)
    local_batch_size = 2
    
    cfg_data = yaml.safe_load(open(args.dataset))
    dataloader_cfg = dict(cfg_data["dataloader"])
    dataloader_cfg["batch_size"] = local_batch_size
    loader = get_dataloader(**dataloader_cfg)

    # load text prompts
    prompts = process_text(prompt=args.prompt, prompt_file=args.prompt_file) if (args.prompt is not None or args.prompt_file is not None) else None
    solver.text_enc_1.to('cuda')
    solver.text_enc_2.to('cuda')
    solver.text_enc_3.to('cuda')

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

    print("Prompts are processed.")

    solver.vae.to('cuda')
    solver.transformer.to('cuda')

    # problem setup
    deg_config = munchify({
        'channels': 3,
        'image_size': args.img_size,
        'deg_scale': args.deg_scale
        })
    operator = get_degradation(args.task, deg_config, solver.transformer.device)

    # solve problem (true batched inference)
    sample_idx = 0
    pbar = tqdm(loader, desc="Solving")
    for batch in pbar:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            imgs, batch_prompts = batch
        else:
            imgs, batch_prompts = batch, None

        imgs = imgs.to(solver.vae.device)

        if args.num_samples > 0:
            remain = args.num_samples - sample_idx
            if remain <= 0:
                break
            if imgs.size(0) > remain:
                imgs = imgs[:remain]
                if batch_prompts is not None:
                    batch_prompts = batch_prompts[:remain]

        batch_size = imgs.size(0)
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
                prompt_texts = prompts[sample_idx:sample_idx + batch_size]
                if len(prompt_texts) != batch_size:
                    raise RuntimeError("Not enough prompts for current batch size.")
                if use_precomputed_prompt:
                    batch_prompt_emb = [prompt_embs[i] for i in range(sample_idx, sample_idx + batch_size)]
                    prompt_emb = [
                        torch.cat([x[0] for x in batch_prompt_emb], dim=0),
                        torch.cat([x[1] for x in batch_prompt_emb], dim=0),
                    ]
                else:
                    prompt_emb = None

        out = solver.sample(measurement=y,
                            operator=operator,
                            prompts=prompt_texts,
                            NFE=args.NFE,
                            img_shape=(args.img_size, args.img_size),
                            cfg_scale=args.cfg_scale,
                            step_size=args.step_size,
                            task=args.task,
                            batch_size=batch_size,
                            prompt_emb=prompt_emb,
                            null_emb=None if batch_size > 1 else null_embs
                            )

        for b in range(batch_size):
            img = imgs[b:b + 1]
            y_b = y[b:b + 1]
            out_b = out[b:b + 1]

            save_image(operator.At(y_b).reshape(img.shape),
                       args.workdir.joinpath(f'input/{str(sample_idx).zfill(4)}.png'),
                       normalize=True)
            save_image(out_b,
                       args.workdir.joinpath(f'recon/{str(sample_idx).zfill(4)}.png'),
                       normalize=True)
            save_image(img,
                       args.workdir.joinpath(f'label/{str(sample_idx).zfill(4)}.png'),
                       normalize=True)
            sample_idx += 1

        if args.num_samples > 0 and sample_idx >= args.num_samples:
            break

    psnr_rec = compute_psnr_folder(args.workdir.joinpath('label'), args.workdir.joinpath('recon'), device=solver.vae.device)
    ssim_rec = compute_ssim_folder(args.workdir.joinpath('label'), args.workdir.joinpath('recon'), device=solver.vae.device)
    fid_rec = compute_fid_folder(args.workdir.joinpath('label'), args.workdir.joinpath('recon'), device=solver.vae.device)

    print("\n================== Evaluation (rank0) ==================")
    print("[GT vs recon]")
    print(f"  PSNR: {psnr_rec:.4f}")
    print(f"  SSIM: {ssim_rec:.6f}")
    print(f"  FID : {fid_rec:.4f}")
    print("========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # sampling params
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--NFE', type=int, default=28)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--img_size', type=int, default=768)

    # workdir params
    parser.add_argument('--workdir', type=Path, default='workdir')

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
    parser.add_argument('--clean_workdir', default=False, action='store_true')
    args = parser.parse_args()


    # workdir creation and seed setup
    set_seed(args.seed)
    args.workdir.joinpath('input').mkdir(parents=True, exist_ok=True)
    args.workdir.joinpath('recon').mkdir(parents=True, exist_ok=True)
    args.workdir.joinpath('label').mkdir(parents=True, exist_ok=True)

    if args.clean_workdir:
        for subdir in ['input', 'recon', 'label']:
            for fp in args.workdir.joinpath(subdir).glob('*.png'):
                fp.unlink()

    # run main script
    run(args)
