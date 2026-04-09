import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def collect_images(input_path: Path):
    if input_path.is_file():
        return [input_path]

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [p for p in sorted(input_path.rglob("*")) if p.suffix.lower() in exts]
    return image_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("samples"), help="image file or folder")
    parser.add_argument("--output", type=Path, default=Path("samples/DAPE.txt"), help="output prompt file")
    parser.add_argument("--seesr_root", type=Path, default=Path("SeeSR-main"), help="path to SeeSR repo")
    parser.add_argument("--ram_model", type=Path, default=None, help="path to RAM base model .pth")
    parser.add_argument("--dape_model", type=Path, default=None, help="path to DAPE finetuned .pth")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    seesr_root = args.seesr_root.resolve()
    if not seesr_root.exists():
        raise FileNotFoundError(f"SeeSR root not found: {seesr_root}")

    sys.path.insert(0, str(seesr_root))
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference

    ram_model = args.ram_model or (seesr_root / "preset/models/ram_swin_large_14m.pth")
    dape_model = args.dape_model or (seesr_root / "preset/models/DAPE.pth")

    if not ram_model.exists():
        raise FileNotFoundError(f"RAM model not found: {ram_model}")

    pretrained_condition = str(dape_model) if dape_model.exists() else None
    if pretrained_condition is None:
        print(f"[WARN] DAPE model not found at {dape_model}, fallback to RAM base model")

    model = ram(
        pretrained=str(ram_model),
        pretrained_condition=pretrained_condition,
        image_size=384,
        vit="swin_l",
    )
    model.eval().to(args.device)

    image_paths = collect_images(args.input)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found under: {args.input}")

    ram_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), open(args.output, "w", encoding="utf-8") as f:
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            lq = ram_tf(img).unsqueeze(0).to(args.device)
            tags_en, _ = inference(lq, model)
            f.write(f"{img_path.name}: {tags_en}\n")

    print(f"Saved {len(image_paths)} prompts to: {args.output}")


if __name__ == "__main__":
    main()
