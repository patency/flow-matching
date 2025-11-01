**Example 1. Super-resolution x 12 (avg-pool) / Dog**
```
python solve.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a sharpened photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory \
    --method flowops;
```
python solve.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a sharpened photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory;

CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 python solve.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a sharpened photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory \
    --method psld;
