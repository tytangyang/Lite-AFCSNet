import argparse
import torch

from teacher_models.sys4_teacher import Sys4Teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--forward", action="store_true", help="Also run a small forward test")
    args = parser.parse_args()

    teacher = Sys4Teacher(args.teacher_ckpt, use_device_models=True)
    parameter_count = sum(p.numel() for p in teacher.parameters())
    trainable_count = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    print(f"teacher parameters: {parameter_count:,}")
    print(f"teacher trainable parameters: {trainable_count:,}")

    if args.forward:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher = teacher.to(device)
        x = torch.randn(9, 1, 256, 64, device=device)
        devices = ("a", "b", "c", "s1", "s2", "s3", "s4", "s5", "s6")
        with torch.inference_mode():
            y = teacher(x, devices)
        print("teacher output shape:", tuple(y.shape))
        print("all finite:", bool(torch.isfinite(y).all()))


if __name__ == "__main__":
    main()
