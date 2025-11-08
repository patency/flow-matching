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
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""])
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
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
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

@register_solver('flow_resample')
class SD3FlowResample(SD3Euler):
    def pixel_optimization(self, measurement, x_prime, operator, max_iters:int=200, lr:float=1e-2, eps:float=1e-3):
        loss_fn = torch.nn.MSELoss()
        opt_var = x_prime.detach().clone().requires_grad_(True)
        optimizer = torch.optim.AdamW([opt_var], lr=lr)
        measurement = measurement.detach()
        for _ in range(max_iters):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(measurement, operator.A(opt_var))
            loss.backward()
            optimizer.step()
            if loss.item() < eps * eps:
                break
        return opt_var.detach()

    def latent_optimization(self, measurement, z_init, operator, max_iters:int=200, lr:float=5e-3, eps:float=1e-3):
        loss_fn = torch.nn.MSELoss()
        z = z_init.detach().clone().requires_grad_(True)
        optimizer = torch.optim.AdamW([z], lr=lr)
        measurement = measurement.detach()
        for _ in range(max_iters):
            optimizer.zero_grad(set_to_none=True)
            x = self.decode(z).float()
            loss = loss_fn(measurement, operator.A(x))
            loss.backward()
            optimizer.step()
            if loss.item() < eps * eps:
                break
        return z.detach()

    def stochastic_resample(self, z_pseudo, z_t, sigma:float, sigma_next:float, k:float=40.0, noise_scale:float=1.0):
        # Adapted from ReSample: use a scalar to weight pseudo vs previous state, with small Gaussian noise
        delta = max(float(sigma - sigma_next), 1e-6)
        rs = k * delta
        weight_pseudo = rs / (rs + 1.0)
        weight_prev = 1.0 - weight_pseudo
        noise_std = math.sqrt(max(float(sigma_next), 0.0)) * noise_scale
        return weight_pseudo * z_pseudo + weight_prev * z_t + noise_std * torch.randn_like(z_t)

    def sample(self, measurement, operator, task,
               prompts: List[str], NFE:int,
               img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               step_size: float=0.0,
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
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
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

        # scheduling helpers
        splits = 3
        index_split = NFE // splits

        pbar = tqdm(timesteps, total=NFE, desc='SD3-FlowResample')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)

            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0
                pred_v = pred_null_v + cfg_scale * (pred_v - pred_null_v)

            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else torch.tensor(0.0, device=self.device, dtype=sigma.dtype)

            # Euler step
            z0t = z - sigma * pred_v
            z1t = z + (1 - sigma) * pred_v
            delta = sigma - sigma_next
            base = z0t + (sigma - delta) * (z1t - z0t)

            # periodic resample stages (time-travel inspired)
            index = NFE - i - 1
            if (index > 0) and (index % 10 == 0):
                z_prev = z.detach()

                # build pseudo x0 in latent
                z_pseudo = z0t.detach()

                if index >= index_split:
                    # pixel optimization stage
                    x_pseudo = self.decode(z_pseudo).float()
                    x_opt = self.pixel_optimization(measurement=measurement.to(x_pseudo.device, dtype=x_pseudo.dtype),
                                                    x_prime=x_pseudo,
                                                    operator=operator,
                                                    max_iters=200,
                                                    lr=1e-2)
                    z_opt = self.encode(x_opt.to(self.vae.device, dtype=self.dtype))
                else:
                    # latent optimization stage
                    z_opt = self.latent_optimization(measurement=measurement.to(self.vae.device, dtype=torch.float32),
                                                      z_init=z_pseudo,
                                                      operator=operator,
                                                      max_iters=200,
                                                      lr=5e-3)

                # stochastic resampling
                z = self.stochastic_resample(z_opt.to(base.device, dtype=base.dtype),
                                             z_prev.to(base.device, dtype=base.dtype),
                                             float(sigma.item()),
                                             float(sigma_next.item()) if isinstance(sigma_next, torch.Tensor) else float(sigma_next),
                                             k=40.0,
                                             noise_scale=1.0)
            else:
                z = base

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
               step_size: float=30.0,
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
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
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

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img

@register_solver("flowchef")
class SD3FlowChef(SD3Euler):
    def data_consistency(self, z0t, operator, measurement, task):
        z0t = z0t.requires_grad_(True)
        x0t = self.decode(z0t).float()
        if "sr" in task:
            loss = torch.linalg.norm((operator.A_pinv(measurement) - operator.A_pinv(operator.A(x0t))).view(1, -1))
        else:
            loss = torch.linalg.norm((operator.At(measurement) - operator.At(operator.A(x0t))).view(1, -1))
        grad = torch.autograd.grad(loss, z0t)[0].half()
        return grad.detach()


    def sample(self, measurement, operator, task,
               prompts: List[str], NFE:int,
               img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               step_size: float=30.0,
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
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
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
        pbar = tqdm(timesteps, total=NFE, desc='SD3-FlowChef')
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
                grad = self.data_consistency(z0t, operator, measurement, task=task)

            # renoising
            z = z0t + (sigma-delta) * (z1t - z0t) - step_size*grad
            # break the graph for next step
            z = z.detach()

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img


@register_solver('psld')
class SD3PSLD(SD3Euler):
    def sample(self, measurement, operator, task,            
               prompts: List[str], NFE:int,
               img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               step_size: float=50.0,
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
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
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
        pbar = tqdm(timesteps, total=NFE, desc='SD3-PSLD')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)

            # freeze model gradients during prediction
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0
                pred_v = pred_null_v + cfg_scale * (pred_v - pred_null_v)

            # enable grad only for DC residue w.r.t z
            z = z.detach().requires_grad_(True)

            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            # denoising
            z0t = z - sigma * pred_v
            z1t = z + (1-sigma) * pred_v
            delta = sigma - sigma_next

            # DC & goodness of z0t
            x_pred = self.decode(z0t).float()
            meas = measurement.to(x_pred.device, dtype=x_pred.dtype)
            y_pred = operator.A(x_pred)
            y_residue = torch.linalg.norm((y_pred - meas).view(1, -1))

            if "sr" in task:
                ortho_proj = x_pred.reshape(1, -1) - operator.A_pinv(y_pred).reshape(1, -1)
                parallel_proj = operator.A_pinv(measurement).reshape(1, -1)
            else:
                ortho_proj = x_pred.reshape(1, -1) - operator.At(y_pred).reshape(1, -1)
                parallel_proj = operator.At(measurement).reshape(1, -1)
            proj = parallel_proj + ortho_proj

            proj_img = proj.reshape(1, 3, imgH, imgW).clamp(-1, 1)
            recon_z = self.encode(proj_img.half())
            z0_residue = torch.linalg.norm((z0t - recon_z).view(1, -1))

            residue = 1 * y_residue + 0.05 * z0_residue
            grad = torch.autograd.grad(residue, z)[0]

            # renoising
            z = z0t + (sigma-delta) * (z1t - z0t) - step_size*grad

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img

@torch.no_grad()
def s_safe_ortho_only(
    r: torch.Tensor,                # (B, D) 原始修正量（-step_size*grad 展平）
    g: Optional[torch.Tensor]=None, # (B, D) 基线向量（pred_v 展平），可为 None
    mode: str="orth",               # "orth" | "along"
    trust_ratio: float=1.0,         # ||w|| <= trust_ratio * ||r||
    min_gain_ratio: float=0.0,      # 可选下限：||w|| >= min_gain_ratio * ||r||
    eps: float=1e-8,
) -> torch.Tensor:
    """
    仅做与 g 的正交/沿向分解 + 信赖域缩放；不做均值/能量中和。
    返回 w: (B, D)
    """
    if g is not None:
        coef  = (r * g).sum(-1, keepdim=True) / (g.pow(2).sum(-1, keepdim=True) + eps)
        projg = coef * g
    else:
        projg = torch.zeros_like(r)

    if mode == "along":
        p = projg
    else:  # "orth"
        p = r - projg if g is not None else r

    # 信赖域缩放
    nr = r.norm(dim=1, keepdim=True) + 1e-8
    nw = p.norm(dim=1, keepdim=True) + 1e-8

    # 上限
    scale_up = torch.clamp((nr / nw) * trust_ratio, max=1.0)
    w = p * scale_up

    # 下限（可选）
    if min_gain_ratio > 0.0:
        nw2 = w.norm(dim=1, keepdim=True) + 1e-8
        need_boost = (nw2 < (min_gain_ratio * nr)).float()
        boost = (min_gain_ratio * nr) / (nw2 + 1e-8)
        w = w * (1.0 - need_boost) + w * boost * need_boost

    return w

@torch.no_grad()
def s_safe_lite_stable(
    r: torch.Tensor,           # (B, D) 原始后验修正量（如 -step_size * grad）
    x: torch.Tensor,           # (B, D) 统计用的“位置”（建议用 z0t 展平）
    g: Optional[torch.Tensor]=None,  # (B, D) 基线/去噪器速度（pred_v 展平）
    use_track: bool=True,      # True: 仅保留沿 g 的分量；False: 去掉沿 g 的分量
    eps: float=1e-8,
    mean_only: bool=True,      # True: 只做均值中和（强烈推荐作为默认）
    gamma_max: float=0.05,     # 若启用能量中和，限制 |gamma| 的上限
    trust_ratio: float=1.0,    # 置信域缩放：||w|| <= trust_ratio * ||r||
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    返回:
      w:      (B, D) 安全修正量（用于替换原始 -step_size*grad）
      b:      (D,)   均值中和偏置（日志）
      gamma:  float  能量中和系数（日志）
    """
    B, D = r.shape

    # proj onto g（不分配 BxDxD）
    if g is not None:
        coef  = (r * g).sum(-1, keepdim=True) / (g.pow(2).sum(-1, keepdim=True) + eps)  # (B,1)
        projg = coef * g                                                                 # (B,D)
    else:
        projg = torch.zeros_like(r)

    # 二选一：保留沿 g（在轨）或去掉沿 g（正交）
    p = projg if (use_track and g is not None) else (r - projg if g is not None else r)

    # 批统计（弱守恒闭式）
    m  = p.mean(dim=0)           # (D,)
    mu = x.mean(dim=0)           # (D,)

    if mean_only:
        b = m
        gamma = 0.0
        w = p - b
    else:
        S = x.pow(2).sum(dim=-1).mean()              # scalar
        C = (x * p).sum(dim=-1).mean()               # scalar
        denom = (mu.pow(2).sum() - S) + eps          # scalar
        gamma = torch.clamp((mu @ m - C) / denom, -gamma_max, gamma_max).item()
        b = m - gamma * mu
        w = p - b - gamma * x

    # 信赖域缩放：不让修正超过原始 r 的量级
    nr = r.norm(dim=1, keepdim=True) + 1e-8
    nw = w.norm(dim=1, keepdim=True) + 1e-8
    scale = torch.clamp((nr / nw) * trust_ratio, max=1.0)
    w = w * scale

    return w, b, gamma

@register_solver('psld_p')
class SD3PSLD_P(SD3Euler):
    """
    PSLD with posterior-safe lite projection (psld_p)

    仅对“后验投影得到的梯度步 r_raw = -step_size*grad”应用安全修正算子，
    不改变去噪器输出 pred_v（主流）。
    """
    def __init__(self, *args,
                 use_track_projection: bool = True,  # True: 仅保留沿 pred_v 的方向
                 mean_only: bool = True,             # True: 仅均值中和（最稳）
                 gamma_max: float = 0.05,            # 若启用能量中和时的限幅
                 trust_ratio: float = 1.0,           # 信赖域缩放
                 safe_eps: float = 1e-8,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_track_projection = use_track_projection
        self.mean_only = mean_only
        self.gamma_max = gamma_max
        self.trust_ratio = trust_ratio
        self.safe_eps = safe_eps

    def sample(self, measurement, operator, task,
               prompts: List[str], NFE:int,
               img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               step_size: float=50.0,
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
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]
            null_prompt_emb = null_prompt_emb.to(self.transformer.device)
            null_pooled_emb = null_pooled_emb.to(self.transformer.device)

        # initialize latent
        z = self.initialize_latent((imgH, imgW), batch_size) if latent is None else latent

        # timesteps
        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SD3-PSLD-P')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)

            # 1) 预测主流向量（去噪器；不参与修正）
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0
                pred_v = pred_null_v + cfg_scale * (pred_v - pred_null_v)

            # 2) 构造 z0t/z1t
            z = z.detach().requires_grad_(True)
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            z0t = z - sigma * pred_v
            z1t = z + (1 - sigma) * pred_v
            delta = sigma - sigma_next

            # 3) DC & prior 残差（仅对 z 求梯度）
            x_pred = self.decode(z0t).float()
            meas = measurement.to(x_pred.device, dtype=x_pred.dtype)
            y_pred = operator.A(x_pred)
            y_residue = torch.linalg.norm((y_pred - meas).view(1, -1))

            if "sr" in task:
                ortho_proj = x_pred.reshape(1, -1) - operator.A_pinv(y_pred).reshape(1, -1)
                parallel_proj = operator.A_pinv(measurement).reshape(1, -1)
            else:
                ortho_proj = x_pred.reshape(1, -1) - operator.At(y_pred).reshape(1, -1)
                parallel_proj = operator.At(measurement).reshape(1, -1)
            proj = parallel_proj + ortho_proj

            proj_img = proj.reshape(1, 3, imgH, imgW).clamp(-1, 1)
            recon_z = self.encode(proj_img.half())
            z0_residue = torch.linalg.norm((z0t - recon_z).view(1, -1))

            residue = 1.0 * y_residue + 0.05 * z0_residue
            grad = torch.autograd.grad(residue, z, retain_graph=False, create_graph=False)[0]

            # 4) 仅修正“后验投影梯度步” —— 不影响 pred_v
            r_raw = - step_size * grad                        # (B, C, H, W) same as z
            with torch.no_grad():
                B = z.shape[0]
                x_vec = z0t.detach().view(B, -1)              # 用去噪后的 z0t 做统计更稳
                r_vec = r_raw.detach().view(B, -1)
                g_vec = pred_v.detach().view(B, -1)

                w_safe_vec = s_safe_ortho_only(
                    r=r_vec,
                    g=g_vec,
                    mode="orth",          # 或 "along"
                    trust_ratio=1.0,      # 不超过原始步长
                    min_gain_ratio=0.0,   # 如担心过小可设 0.1
                    eps=1e-8,
                )
                w_safe = w_safe_vec.view_as(z)

            # 5) 组合更新（保留 PSLD 的重噪形式），仅替换 -step_size*grad
            z = z0t + (sigma - delta) * (z1t - z0t) + w_safe
            z = z.detach()  # 下一步再需要 grad

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img





