"""
train.py
========
Entry point chính để huấn luyện AlphaZero-Xiangqi.

Ví dụ:
    python train.py --num-iters 100 --num-eps 50 --num-channels 256
                    --num-res-blocks 5 --cuda
"""
from __future__ import annotations
import argparse
import os
import sys
import time

# Tăng recursion limit để MCTS không bị RecursionError
sys.setrecursionlimit(10000)

# Đảm bảo import được khi chạy từ thư mục dự án.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import dotdict
from xiangqi.XiangqiGame import XiangqiGame
from xiangqi.pytorch.NNet import NNetWrapper


def get_args() -> dotdict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--num-eps", type=int, default=100, help="Self-play games per iter")
    parser.add_argument("--max-len", type=int, default=200, help="Max moves per game")
    parser.add_argument("--num-channels", type=int, default=256)
    parser.add_argument("--num-res-blocks", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (safer for long training)")
    parser.add_argument("--safe-cuda", action="store_true", help="CUDA at 60%% power - balanced speed/safety")
    parser.add_argument("--num-mcts-sims", type=int, default=50)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/")
    parser.add_argument("--load-model", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--log-dir", type=str, default="logs/")
    # Tài nguyên
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0=safe)")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N eps (default: 5)")
    parser.add_argument("--train-every", type=int, default=10, help="Train every N eps for streaming (default: 10)")
    parser.add_argument("--auto-save-minutes", type=int, default=10, help="Auto-save checkpoint every N minutes (default: 10, 0=off)")
    args = parser.parse_args()
    return dotdict(vars(args))


def main():
    args = get_args()
    
    # Priority: CPU > Safe CUDA > CUDA > auto
    if args.cpu:
        args.cuda = False
        print("Running on CPU (safer for long training)")
    elif args.safe_cuda:
        args.cuda = True
        try:
            import torch
            # Gioi han VRAM 40% (1.6GB cho 4GB GPU)
            torch.cuda.set_per_process_memory_fraction(0.4)
            # Giam batch size cho RTX 3050 Ti 4GB
            if args.batch_size > 2:
                args.batch_size = 2
            # Giam epochs training
            if args.epochs > 2:
                args.epochs = 2
            # Giam channels manh cho GPU 4GB
            if args.num_channels > 64:
                args.num_channels = 64
            # Giam MCTS simulations de tiet kiem RAM
            if args.num_mcts_sims > 20:
                args.num_mcts_sims = 20
            # Giam res blocks
            if args.num_res_blocks > 4:
                args.num_res_blocks = 4
            print("Safe CUDA: 40%% VRAM, batch=2, epochs=2, channels=64, mcts=20, res_blocks=4")
            print("Auto-save every 5 episodes to avoid data loss")
        except Exception as e:
            print(f"Warning: Could not configure Safe CUDA: {e}")
    elif args.cuda:
        try:
            import torch
            # Kiem tra GPU memory
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU: {gpu_name} ({total_mem:.1f} GB)")
                
                # Tu dong giam cau hinh cho GPU nho
                if total_mem < 6:  # GPU < 6GB
                    torch.cuda.set_per_process_memory_fraction(0.30)
                    if args.batch_size > 1:
                        args.batch_size = 1
                    if args.num_channels > 4:
                        args.num_channels = 4
                    if args.num_res_blocks > 1:
                        args.num_res_blocks = 1
                    if args.num_mcts_sims > 5:
                        args.num_mcts_sims = 5
                    if args.epochs > 1:
                        args.epochs = 1
                    # accum khong giup forward pass, chi giu nguyen
                    args.accumulate_steps = 1
                    print(f"Ultra-minimal: batch={args.batch_size}, channels={args.num_channels}, res={args.num_res_blocks}, epochs={args.epochs}, mcts={args.num_mcts_sims}")
                elif total_mem < 10:  # GPU 6-10GB
                    torch.cuda.set_per_process_memory_fraction(0.8)
                    if args.batch_size > 16:
                        args.batch_size = 16
                else:  # GPU > 10GB
                    torch.cuda.set_per_process_memory_fraction(0.85)
                    
            print("CUDA memory limited to 70-85%")
        except Exception as e:
            print(f"Warning: Could not set CUDA memory limit: {e}")

    print(f"Args: {dict(args)}")

    game = XiangqiGame()
    nnet = NNetWrapper(game, args)
    
    # Tim checkpoint moi nhat de resume
    import glob
    checkpoints = sorted(glob.glob(os.path.join(args.checkpoint, "*.pth.tar")))
    if args.load_model and checkpoints:
        # Tim checkpoint auto-save cuoi cung
        auto_checkpoints = [c for c in checkpoints if "auto_" in c]
        if auto_checkpoints:
            latest = auto_checkpoints[-1]
            print(f"Resuming from: {latest}")
            nnet.load_checkpoint(folder=args.checkpoint, filename=os.path.basename(latest))
        else:
            try:
                nnet.load_checkpoint(folder=args.checkpoint, filename="best.pth.tar")
                print("Loaded best checkpoint.")
            except:
                print("No checkpoint to load.")
    else:
        print("Starting fresh training.")

    from alphazero.Coach import Coach
    coach = Coach(game, nnet, args)
    coach.learn()


if __name__ == "__main__":
    main()
