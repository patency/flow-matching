**Example 1. solve Super-resolution x 12 (avg-pool) / Dog**

python batch_solve.py --dataset configs/DIV2K_train.yml --num_samples 4 --method flowdps --clean_workdir
```
# flowdps
python solve.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "brown, corgi, dog, hang, leash, mouth, neckband, pink" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory \
    --method flowdps;

# flowchef
python solve.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory \
    --method flowchef;

# psld
python solve.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory \
    --method psld;

# resample
python solve.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory \
    --method flow_resample;

```

**Example 1. solve_assess Super-resolution x 12 (avg-pool) / Dog**
```
# flowdps
python solve_assess.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory \
    --method flowdps;

# flowchef
python solve_assess.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory \
    --method flowchef;

# psld
python solve_assess.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory \
    --method psld;

# psld_p
python solve_assess.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory \
    --method psld_simple;

# resample
python solve_assess.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory \
    --method flow_resample;

```

**Example 2. Super-resolution x 12 (bicubic) / Animal**
```
python solve.py \
    --img_size 768 \
    --img_path samples/div2k_example.png \
    --prompt "a high quality photo of animal, bush, close-up, fox, grass, green, greenery, hide, panda, red, red panda, stare" \
    --task sr_bicubic \
    --deg_scale 12 \
    --efficient_memory \
    --method flowdps;

python solve.py \
    --img_size 768 \
    --img_path samples/0001.png \
    --prompt "a high quality photo, close-up, coral reef, sea, pink, purple, red, reef, starfish" \
    --task sr_bicubic \
    --deg_scale 12 \
    --efficient_memory \
    --method flowdps;

python solve.py \
    --img_size 768 \
    --img_path samples/div2k_butterfly.png \
    --prompt "a high quality photo of insect, break, butterfly, leaf, green, greenery, monarch, perch, plant, stem" \
    --task sr_bicubic \
    --deg_scale 12 \
    --efficient_memory \
    --method flowdps;

python solve.py \
    --img_size 768 \
    --img_path samples/div2k_example.png \
    --prompt "a high quality photo of animal, bush, close-up, fox, grass, green, greenery, hide, panda, red, red panda, stare" \
    --task sr_bicubic \
    --deg_scale 12 \
    --efficient_memory \
    --method flowchef;

python solve.py \
    --img_size 768 \
    --img_path samples/div2k_example.png \
    --prompt "a high quality photo of animal, bush, close-up, fox, grass, green, greenery, hide, panda, red, red panda, stare" \
    --task sr_bicubic \
    --deg_scale 12 \
    --efficient_memory \
    --method psld;
```

**Example 3. Motion Deblur / Human**
```
python solve.py \
    --img_size 768 \
    --img_path samples/ffhq_example.png \
    --prompt "a photo of a closed face" \
    --task deblur_motion \
    --deg_scale 61 \
    --efficient_memory;
```
