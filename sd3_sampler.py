from typing import List, Tuple, Optional
import math
import torch

from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline


# =======================================================================
# Factory
# =======================================================================

__SOLVER__ = {}

def register_solver(name:str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls
    return wrapper

def get_solver(name:str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)

# =======================================================================


class StableDiffusion3Base():
    def __init__(self, model_key:str='stabilityai/stable-diffusion-3-medium-diffusers', device='cuda', dtype=torch.float16):
        self.device = device
        self.dtype = dtype

        pipe = StableDiffusion3Pipeline.from_pretrained(model_key, torch_dtype=self.dtype)

        self.scheduler = pipe.scheduler

        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2
        self.text_enc_3 = pipe.text_encoder_3

        self.vae=pipe.vae
        self.transformer = pipe.transformer
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)-1) if hasattr(self, "vae") and self.vae is not None else 8
        )

        del pipe

    def encode_prompt(self, prompt: List[str], batch_size:int=1) -> List[torch.Tensor]:
        '''
        We assume that
        1. number of tokens < max_length
        2. one prompt for one image
        '''
        # CLIP encode (used for modulation of adaLN-zero)
        # now, we have two CLIPs
        text_clip1_ids = self.tokenizer_1(prompt,
                                          padding="max_length",
                                          max_length=77,
                                          truncation=True,
                                          return_tensors='pt').input_ids
        text_clip1_emb = self.text_enc_1(text_clip1_ids.to(self.text_enc_1.device), output_hidden_states=True)
        pool_clip1_emb = text_clip1_emb[0].to(dtype=self.dtype, device=self.text_enc_1.device)
        text_clip1_emb = text_clip1_emb.hidden_states[-2].to(dtype=self.dtype, device=self.text_enc_1.device)

        text_clip2_ids = self.tokenizer_2(prompt,
                                          padding="max_length",
                                          max_length=77,
                                          truncation=True,
                                          return_tensors='pt').input_ids
        text_clip2_emb = self.text_enc_2(text_clip2_ids.to(self.text_enc_2.device), output_hidden_states=True)
        pool_clip2_emb = text_clip2_emb[0].to(dtype=self.dtype, device=self.text_enc_2.device)
        text_clip2_emb = text_clip2_emb.hidden_states[-2].to(dtype=self.dtype, device=self.text_enc_2.device)

        # T5 encode (used for text condition)
        text_t5_ids = self.tokenizer_3(prompt,
                                       padding="max_length",
                                       max_length=77,
                                       truncation=True,
                                       add_special_tokens=True,
                                       return_tensors='pt').input_ids
        text_t5_emb = self.text_enc_3(text_t5_ids.to(self.text_enc_3.device))[0]
        text_t5_emb = text_t5_emb.to(dtype=self.dtype, device=self.text_enc_3.device)


        # Merge
        clip_prompt_emb = torch.cat([text_clip1_emb, text_clip2_emb], dim=-1)
        clip_prompt_emb = torch.nn.functional.pad(
            clip_prompt_emb, (0, text_t5_emb.shape[-1] - clip_prompt_emb.shape[-1])
        )
        prompt_emb = torch.cat([clip_prompt_emb, text_t5_emb], dim=-2)
        pooled_prompt_emb = torch.cat([pool_clip1_emb, pool_clip2_emb], dim=-1)

        return prompt_emb, pooled_prompt_emb


    def initialize_latent(self, img_size:Tuple[int], batch_size:int=1, **kwargs):
        H, W = img_size
        lH, lW = H//self.vae_scale_factor, W//self.vae_scale_factor
        lC = self.transformer.config.in_channels
        latent_shape = (batch_size, lC, lH, lW)

        z = torch.randn(latent_shape, device=self.device, dtype=self.dtype)

        return z

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(image).latent_dist.sample()
        z = (z-self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = (z/self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]

    def predict_vector(self, z, t, prompt_emb, pooled_emb):
        v = self.transformer(hidden_states=z,
                             timestep=t,
                             pooled_projections=pooled_emb,
                             encoder_hidden_states=prompt_emb,
                             return_dict=False)[0]
        return v

class SD3Euler(StableDiffusion3Base):
    def __init__(self, model_key:str='stabilityai/stable-diffusion-3-medium-diffusers', device='cuda'):
        super().__init__(model_key=model_key, device=device)

    def inversion(self, src_img, prompts: List[str], NFE:int, cfg_scale: float=1.0, batch_size: int=1,
                  prompt_emb:Optional[List[torch.Tensor]]=None,
                  null_emb:Optional[List[torch.Tensor]]=None):

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb = prompt_emb.to(self.transformer.device)
            pooled_emb = pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""] * batch_size, batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb = null_prompt_emb.to(self.transformer.device)
            null_pooled_emb = null_pooled_emb.to(self.transformer.device)

        # initialize latent
        src_img = src_img.to(device=self.vae.device, dtype=self.dtype)
        with torch.no_grad():
            z = self.encode(src_img).to(self.transformer.device)

        # timesteps (default option. You can make your custom here.)
        self.scheduler.set_timesteps(NFE, device=self.transformer.device)
        timesteps = self.scheduler.timesteps
        timesteps = torch.cat([timesteps, torch.zeros(1, device=self.transformer.device)])
        timesteps = reversed(timesteps)
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps[:-1], total=NFE, desc='SD3 Euler Inversion')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.transformer.device)
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1]

            z = z + (sigma_next - sigma) * (pred_null_v + cfg_scale * (pred_v - pred_null_v))

        return z

    def sample(self, prompts: List[str], NFE:int, img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb = prompt_emb.to(self.transformer.device)
            pooled_emb = pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""] * batch_size, batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb = null_prompt_emb.to(self.transformer.device)
            null_pooled_emb = null_pooled_emb.to(self.transformer.device)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # timesteps (default option. You can make your custom here.)
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SD3 Euler')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)
            pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
            if cfg_scale != 1.0:
                pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
            else:
                pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            z = z + (sigma_next - sigma) * (pred_null_v + cfg_scale * (pred_v - pred_null_v))

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img

@register_solver("retroflowdps")
class SD3FlowDPS(SD3Euler):
    def data_consistency(self, z0t, operator, measurement, task, stepsize:int=30.0):
        z0t = z0t.requires_grad_(True)
        num_iters = 3
        for _ in range(num_iters):
            x0t = self.decode(z0t).float()
            if "sr" in task:
                loss = torch.linalg.norm((operator.A_pinv(measurement) - operator.A_pinv(operator.A(x0t))).view(1, -1))
            else:
                loss = torch.linalg.norm((operator.At(measurement) - operator.At(operator.A(x0t))).view(1, -1))
            grad = torch.autograd.grad(loss, z0t)[0].half()
            z0t = z0t - stepsize*grad

        return z0t.detach()

    def sample(self, measurement, operator, task,
            prompts: List[str], NFE:int,
            img_shape: Optional[Tuple[int]]=None,
            cfg_scale: float=1.0, batch_size: int = 1,
            step_size: float=15.0,
            latent:Optional[List[torch.Tensor]]=None,
            prompt_emb:Optional[List[torch.Tensor]]=None,
            null_emb:Optional[List[torch.Tensor]]=None):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb = prompt_emb.to(self.transformer.device)
            pooled_emb = pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""] * batch_size, batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb = null_prompt_emb.to(self.transformer.device)
            null_pooled_emb = null_pooled_emb.to(self.transformer.device)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # timesteps
        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # 回溯到中段（默认约一半步数），避免过早回溯导致 warm 信息不足
        retro_idx = max(1, (NFE // 2) - 1)

        # =========================
        # first pass
        # =========================
        pbar = tqdm(timesteps, total=NFE, desc='SD3-FlowDPS')
        do_retro = False
        # [Theory] accumulate per-step predicted noise endpoints z1t along the
        # flow trajectory.  Each z1t = z + (1-sigma)*v estimates the source
        # noise z1.  Averaging them across steps yields a manifold-averaged
        # noise anchor z1_avg with lower variance than any single-step estimate
        # (analogous to FlowEdit's source inversion, but reference-free).
        z1_hist = []

        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)

            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            # denoising
            pred = pred_null_v + cfg_scale * (pred_v - pred_null_v)
            z0t = z - sigma * pred
            z1t = z + (1 - sigma) * pred
            delta = sigma - sigma_next

            # record predicted noise endpoint for manifold averaging
            z1_hist.append(z1t.detach().clone())

            z0y = self.data_consistency(z0t, operator, measurement, task=task, stepsize=step_size)
            z0y = (1 - sigma) * z0t + sigma * z0y

            # At NFE-2 trigger retro: use manifold-averaged z1 as noise anchor.
            # z1_avg has lower variance than the single-step z1y inferred from
            # the last step alone, giving a more stable retro initialization.
            if i == NFE - 2:
                sigma_retro = sigmas[retro_idx]
                z1_avg = torch.stack(z1_hist, dim=0).mean(dim=0)
                z = (1 - sigma_retro) * z0y + sigma_retro * z1_avg
                do_retro = True
                break

            # normal update
            noise = math.sqrt(sigma_next) * z1t + math.sqrt(1 - sigma_next) * torch.randn_like(z1t)
            z = z0y + (sigma - delta) * (noise - z0y)
            # equal to z = (1 - sigma_next) * z0y + sigma_next * noise

        # =========================
        # resample from step 14 to NFE
        # =========================
        if do_retro:
            pbar2 = tqdm(range(retro_idx, NFE), total=NFE-retro_idx, desc='Retro resample from step 14')
            for i in pbar2:
                t = timesteps[i]
                timestep = t.expand(z.shape[0]).to(self.device)

                with torch.no_grad():
                    pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                    if cfg_scale != 1.0:
                        pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                    else:
                        pred_null_v = 0.0

                sigma = sigmas[i]
                sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

                pred = pred_null_v + cfg_scale * (pred_v - pred_null_v)
                z0t = z - sigma * pred
                z1t = z + (1 - sigma) * pred
                delta = sigma - sigma_next

                z0y = self.data_consistency(z0t, operator, measurement, task=task, stepsize=step_size)
                z0y = (1 - sigma) * z0t + sigma * z0y

                noise = math.sqrt(sigma_next) * z1t + math.sqrt(1 - sigma_next) * torch.randn_like(z1t)
                z = z0y + (sigma - delta) * (noise - z0y)

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img

@register_solver("flowdps")
class SD3FlowDPS(SD3Euler):
    def data_consistency(self, z0t, operator, measurement, task, stepsize:int=30.0):
        z0t = z0t.requires_grad_(True)
        num_iters = 3
        for _ in range(num_iters):
            x0t = self.decode(z0t).float()
            if "sr" in task:
                loss = torch.linalg.norm((operator.A_pinv(measurement) - operator.A_pinv(operator.A(x0t))).view(1, -1))
            else:
                loss = torch.linalg.norm((operator.At(measurement) - operator.At(operator.A(x0t))).view(1, -1))
            grad = torch.autograd.grad(loss, z0t)[0].half()
            z0t = z0t - stepsize*grad

        return z0t.detach()

    def sample(self, measurement, operator, task,
               prompts: List[str], NFE:int,
               img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               step_size: float=15.0,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb.to(self.transformer.device)
            pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""] * batch_size, batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb.to(self.transformer.device)
            null_pooled_emb.to(self.transformer.device)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # timesteps (default option. You can make your custom here.)
        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SD3-FlowDPS')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)

            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            # denoising
            z0t = z - sigma * (pred_null_v + cfg_scale * (pred_v-pred_null_v))
            z1t = z + (1-sigma) * (pred_null_v + cfg_scale * (pred_v-pred_null_v))
            delta = sigma - sigma_next

            if i < NFE:
                z0y = self.data_consistency(z0t, operator, measurement, task=task, stepsize=step_size)
                z0y = (1-sigma) * z0t + sigma * z0y

            # renoising
            noise = math.sqrt(sigma_next) * z1t + math.sqrt(1-sigma_next) * torch.randn_like(z1t)
            z = z0y + (sigma-delta) * (noise - z0y)
            #equal to z = (1 - sigma_next) * z0y + sigma_next * noise

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img







