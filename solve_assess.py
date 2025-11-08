import argparse
from pathlib import Path
from typing import List

from datetime import datetime
import numpy as np

from munch import munchify
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from util import set_seed, get_img_list, process_text
from sd3_sampler import get_solver
from functions.degradation import get_degradation

# optional deps
try:
    import lpips
except Exception:
    lpips = None

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None  # type: ignore

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

    # load text prompts      
    prompts = process_text(prompt=args.prompt, prompt_file=args.prompt_file)
    solver.text_enc_1.to('cuda')
    solver.text_enc_2.to('cuda')
    solver.text_enc_3.to('cuda')

    if args.efficient_memory:
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
        prompt_embs = [[None, None]] * len(prompts)
        null_embs = [None, None]

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

    # solve problem
    tf = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor()
        ])

    # per-run directory: workdir/method/timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = args.workdir.joinpath(args.method, timestamp)
    (run_dir / 'input').mkdir(parents=True, exist_ok=True)
    (run_dir / 'recon').mkdir(parents=True, exist_ok=True)
    (run_dir / 'label').mkdir(parents=True, exist_ok=True)

    # tensorboard
    writer = SummaryWriter(log_dir=str(run_dir / 'tb'))

    # lpips
    lpips_model = None
    if lpips is not None:
        try:
            lpips_model = lpips.LPIPS(net='alex').to(solver.vae.device).eval()
        except Exception:
            lpips_model = None

    # fid
    fid_metric = None
    if FrechetInceptionDistance is not None:
        try:
            # keep FID on CPU for widest compatibility
            fid_metric = FrechetInceptionDistance(feature=2048)
        except Exception:
            fid_metric = None

    psnr_list, ssim_list, lpips_list = [], [], []

    pbar = tqdm(get_img_list(args.img_path), desc="Solving")
    for i, path in enumerate(pbar):
        img = tf(Image.open(path).convert('RGB'))
        img = img.unsqueeze(0).to(solver.vae.device)
        img = img * 2 - 1

        y = operator.A(img)
        y = y + 0.03 * torch.randn_like(y)

        out = solver.sample(measurement=y,
                            operator=operator,
                            prompts=prompts[i] if len(prompts)>1 else prompts[0],
                            NFE=args.NFE,
                            img_shape=(args.img_size, args.img_size),
                            cfg_scale=args.cfg_scale,
                            step_size=args.step_size,
                            task=args.task,
                            prompt_emb=prompt_embs[i] if len(prompt_embs)>1 else prompt_embs[0],
                            null_emb=null_embs
                            )
        # save results
        inp = operator.At(y).reshape(img.shape)
        save_image(inp, run_dir.joinpath(f'input/{str(i).zfill(4)}.png'), normalize=True)
        save_image(out, run_dir.joinpath(f'recon/{str(i).zfill(4)}.png'), normalize=True)
        save_image(img, run_dir.joinpath(f'label/{str(i).zfill(4)}.png'), normalize=True)

        # metrics using libraries
        # to [0,1] numpy HWC
        img_01 = ((img.clamp(-1, 1) + 1) * 0.5).detach().cpu()[0]
        out_01 = ((out.clamp(-1, 1) + 1) * 0.5).detach().cpu()[0]
        img_np = np.transpose(img_01.numpy(), (1, 2, 0))
        out_np = np.transpose(out_01.numpy(), (1, 2, 0))

        psnr_val = peak_signal_noise_ratio(img_np, out_np, data_range=1.0)
        # channel_axis=2 for HWC
        ssim_val = structural_similarity(img_np, out_np, channel_axis=2, data_range=1.0)
        psnr_list.append(float(psnr_val))
        ssim_list.append(float(ssim_val))

        lpips_val = None
        if lpips_model is not None:
            try:
                lp = lpips_model(out, img)
                lpips_val = float(lp.mean().detach().cpu().item())
                lpips_list.append(lpips_val)
            except Exception:
                lpips_val = None

        if fid_metric is not None:
            try:
                # torchmetrics FID: accepts uint8 [0,255] or float [0,1]
                fid_metric.update((img_01 * 255.0).to(torch.uint8).cpu(), real=True)
                fid_metric.update((out_01 * 255.0).to(torch.uint8).cpu(), real=False)
            except Exception:
                pass

        # console
        if lpips_val is not None:
            print(f"[{i}] PSNR: {psnr_val:.4f} SSIM: {ssim_val:.4f} LPIPS: {lpips_val:.4f}")
        else:
            print(f"[{i}] PSNR: {psnr_val:.4f} SSIM: {ssim_val:.4f}")

        # tensorboard scalars and images
        writer.add_scalar('metrics/psnr', float(psnr_val), i)
        writer.add_scalar('metrics/ssim', float(ssim_val), i)
        if lpips_val is not None:
            writer.add_scalar('metrics/lpips', float(lpips_val), i)
        writer.add_image('images/input', (inp[0].clamp(-1,1)+1)*0.5, i)
        writer.add_image('images/label', (img[0].clamp(-1,1)+1)*0.5, i)
        writer.add_image('images/recon', (out[0].clamp(-1,1)+1)*0.5, i)

        if (i+1) == args.num_samples:
            break

    # finalize summary
    fid_val = None
    if fid_metric is not None:
        try:
            fid_val = float(fid_metric.compute().detach().cpu().item())
        except Exception:
            fid_val = None

    avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    avg_lpips = float(np.mean(lpips_list)) if lpips_list else None

    # csv summary
    lines = [
        f"count,{len(psnr_list)}",
        f"avg_psnr,{avg_psnr:.6f}",
        f"avg_ssim,{avg_ssim:.6f}",
    ]
    if avg_lpips is not None:
        lines.append(f"avg_lpips,{avg_lpips:.6f}")
    if fid_val is not None:
        lines.append(f"fid,{fid_val:.6f}")
    (run_dir / 'metrics.csv').write_text("\n".join(lines))

    # console summary
    print("=== Summary ===")
    print(f"Images: {len(psnr_list)}")
    print(f"Avg PSNR: {avg_psnr:.4f}")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    if avg_lpips is not None:
        print(f"Avg LPIPS: {avg_lpips:.4f}")
    if fid_val is not None:
        print(f"FID: {fid_val:.4f}")

    # tensorboard summary
    writer.add_scalar('metrics_summary/avg_psnr', avg_psnr, 0)
    writer.add_scalar('metrics_summary/avg_ssim', avg_ssim, 0)
    if avg_lpips is not None:
        writer.add_scalar('metrics_summary/avg_lpips', avg_lpips, 0)
    if fid_val is not None:
        writer.add_scalar('metrics_summary/fid', fid_val, 0)
    writer.close()

    # optionally launch tensorboard
    if getattr(args, 'auto_tb', False):
        tb_bin = shutil.which('tensorboard')
        if tb_bin is not None:
            tb_log = (run_dir / 'tensorboard_stdout.log')
            try:
                with open(tb_log, 'ab', buffering=0) as f:
                    subprocess.Popen([
                        tb_bin,
                        f"--logdir={str(run_dir / 'tb')}",
                        f"--port={args.tb_port}"
                    ], stdout=f, stderr=subprocess.STDOUT, close_fds=True)
                print(f"TensorBoard started on port {args.tb_port}. Logdir: {str(run_dir / 'tb')}")
            except Exception as e:
                print(f"Failed to start TensorBoard automatically: {e}")
        else:
            print("tensorboard executable not found in PATH. Please install or add to PATH.")


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

    # problem params
    parser.add_argument('--task', type=str, default='sr_avgpool')
    parser.add_argument('--method', type=str, default='flowdps')
    parser.add_argument('--deg_scale', type=int, default=12)

    # solver params
    parser.add_argument('--step_size', type=float, default=15.0)
    parser.add_argument('--efficient_memory',default=False, action='store_true')
    parser.add_argument('--auto_tb', default=True, action='store_true', help='Auto-launch TensorBoard after run')
    parser.add_argument('--tb_port', type=int, default=6006, help='TensorBoard port')
    args = parser.parse_args()


    # workdir creation and seed setup
    set_seed(args.seed)

    # per-run subdirectories are created inside run()

    # run main script
    run(args)

