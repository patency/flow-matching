# RetroFlowDPS: Retrospective Posterior Sampling for Flow-Based Inverse Problems
Abstract

❗️Flow matching has recently emerged as a powerful framework for generative modeling based on ordinary differential equations (ODEs). Existing flow-based inverse solvers mainly refine the current state using measurement consistency, but they typically continue sampling only along the forward trajectory.

❓ In inverse problems, a late-stage data-consistent estimate often contains useful information about a more plausible earlier latent state. However, existing solvers do not explicitly exploit this idea. Therefore, we introduce RetroFlowDPS, a simple training-free extension of flow-based posterior sampling that performs retrospection: after obtaining a refined endpoint at a late step, it reconstructs a more data-consistent earlier latent and resamples from that retrospectively corrected state.

👍 RetroFlowDPS can be seamlessly integrated into existing flow-based inverse solvers such as FlowDPS without additional training. It preserves the efficiency and flexibility of the original framework while providing improved reconstruction quality on linear inverse problems such as super-resolution and deblurring.

Quick Start
Environment Setup

First, clone this repository and install the requirements.

git clone https://github.com/yourname/RetroFlowDPS.git
cd RetroFlowDPS
conda create -n retroflowdps python==3.10
conda activate retroflowdps
pip install -r requirements.txt

The provided requirements.txt installs PyTorch with CUDA 11.8. If you are using a different CUDA version, please modify it accordingly.

For the motion blur task, also clone the following repository:

git clone https://github.com/LeviBorodenko/motionblur.git
Examples

You can quickly test the method with the following examples.

Example 1. Super-resolution ×12 (avg-pool) / Dog / FlowDPS
python solve.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory

Example 2. Super-resolution ×12 (avg-pool) / Dog / RetroFlowDPS
python solve.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory

Example 3. Super-resolution ×12 (bicubic) / Animal
python solve.py \
    --img_size 768 \
    --img_path samples/div2k_example.png \
    --prompt "a high quality photo of animal, bush, close-up, fox, grass, green, greenery, hide, panda, red, red panda, stare" \
    --task sr_bicubic \
    --deg_scale 12 \
    --efficient_memory \


The prompt after "a high quality photo of" can be extracted from the degraded measurement using DAPE.

Example 4. Motion Deblur / Human
python solve.py \
    --img_size 768 \
    --img_path samples/ffhq_example.png \
    --prompt "a photo of a closed face" \
    --task deblur_motion \
    --deg_scale 61 \
    --efficient_memory \
    --method retroflowdps
Main Idea of Retrospection

RetroFlowDPS follows the standard flow-based posterior sampling procedure during most of the trajectory. Near the end of sampling, it:

computes a data-consistent refined endpoint,
reconstructs an earlier latent state using the retrospective formula,
returns to that earlier step, and
resamples to the final step.

This procedure is different from simply storing and replaying an earlier latent. Instead, the earlier latent is reconstructed from the late-stage refined sample, making it more consistent with the measurement.

How to Choose Task and Solver

You can freely change the task and solver using the following arguments:

task : sr_avgpool / sr_bicubic / deblur_gauss / deblur_motion
method : psld / flowchef / flowdps / retroflowdps

If you want to change the amount of degradation, modify deg_scale.

For super-resolution tasks, deg_scale is the downscaling factor.
For deblurring tasks, deg_scale is the kernel size.
RetroFlowDPS Settings

The current implementation supports a simple retrospective resampling strategy near the end of sampling.

Typical settings include:

triggering retrospection at a late step such as NFE-2,
rewinding to an earlier step such as step 14,
reconstructing the rewind state from the refined late-stage sample,
resampling from the rewind step to the end.

If you expose these options through command-line arguments, you can document them like this:

retro_trigger_step : the step where retrospection is triggered
retro_step : the target rewind step
retro_num : number of retrospective resampling rounds

For example:

python solve.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory \
    --method retroflowdps \
    --retro_trigger_step -2 \
    --retro_step 14

If these arguments are not yet exposed in your code, you can remove this section for now or keep it as a note for future updates.

Arbitrary Size Problem

You can also solve inverse problems for rectangular images.

python solve_arbitrary.py \
    --imgH 768 \
    --imgW 1152 \
    --img_path samples/div2k_example.png \
    --prompt "a high quality photo of animal, bush, close-up, fox, grass, green, greenery, hide, panda, red, red panda, stare" \
    --task deblur_motion \
    --deg_scale 61 \
    --efficient_memory \
    --method retroflowdps
Measurement	Reconstruction

	
Efficient Inference

If you use --efficient_memory, the text encoder will precompute text embeddings and be removed from the GPU.

This allows inverse problem solving on a single GPU with 24GB VRAM.

Preliminary Observation

In our preliminary experiments, RetroFlowDPS improves reconstruction quality over the vanilla FlowDPS baseline with only modest additional sampling cost. For example, on the AFHQ super-resolution ×12 example, triggering retrospection near the end of sampling and rewinding to an earlier step yielded improved PSNR and SSIM.

You may report detailed benchmark results here once the full evaluation is completed.

Citation

If you find this project useful, please cite the corresponding paper once available.

@article{retroflowdps2026,
  title={RetroFlowDPS: Retrospective Flow-Driven Posterior Sampling for Inverse Problems},
  author={Anonymous},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
Acknowledgements

This project builds upon several excellent open-source works, including:

FlowDPS
FlowChef
motionblur
diffusers / transformers / PyTorch

Please also cite the original repositories and papers if you use this code.
