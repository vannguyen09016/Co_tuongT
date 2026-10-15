"""
NNet.py
=======
Wrapper cho XiangqiNNet cung cấp:
  - predict(state): (pi, v) numpy arrays
  - train(states, pis, vs): thực hiện một epoch train
  - save_checkpoint / load_checkpoint
  - predict_batch: dùng cho self-play (multiprocessing-safe)

Tương thích với Coach/MCTS trong alpha-zero-general: dùng pickle-friendly
state_dict của nn.Module.
"""
from __future__ import annotations
import os
import time
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from xiangqi.pytorch.XiangqiNNet import XiangqiNNet


class NNetWrapper:
    """Wrapper PyTorch quanh XiangqiNNet."""

    def __init__(self, game, args: dict):
        """
        Args:
            game: instance của Game (XiangqiGame).
            args: dict cấu hình chứa các key:
                - lr: learning rate
                - dropout: (optional)
                - epochs: số epoch per train call
                - batch_size: batch size
                - num_channels: (optional, default 256)
                - num_res_blocks: (optional, default 5)
                - cuda: bool, dùng GPU nếu có
        """
        self.game = game
        self.args = args
        self.nnet = XiangqiNNet(
            board_rows=game.getBoardSize()[0],
            board_cols=game.getBoardSize()[1],
            in_channels=15,  # 7*2 + 1 = 15 cho Xiangqi
            action_size=game.getActionSize(),
            num_channels=args.get("num_channels", 256),
            num_res_blocks=args.get("num_res_blocks", 5),
        )
        # Optimizer
        self.lr = args.get("lr", 1e-3)
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.lr)
        # Device
        use_cuda = bool(args.get("cuda", False)) and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.nnet.to(self.device)
        # Track loss
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []

    # ----------------------------------------------------------------------
    def train(self, examples: List[Tuple]) -> float:
        """examples: list of (canonical_board, pi, v).
        Returns: average policy+value loss.
        """
        if not examples:
            return 0.0
        epochs = int(self.args.get("epochs", 5))
        batch_size = int(self.args.get("batch_size", 64))

        # Encode examples to numpy arrays
        boards, pis, vs = [], [], []
        for board, pi, v in examples:
            # Convert board -> state tensor
            from xiangqi.XiangqiGame import board_to_tensor
            state = board_to_tensor(board, _convert_player_to_int(1))
            boards.append(state)
            pis.append(pi)
            vs.append([v])
        boards = np.asarray(boards, dtype=np.float32)
        pis = np.asarray(pis, dtype=np.float32)
        vs = np.asarray(vs, dtype=np.float32)

        # Move to torch
        boards_t = torch.from_numpy(boards).to(self.device)
        pis_t = torch.from_numpy(pis).to(self.device)
        vs_t = torch.from_numpy(vs).to(self.device)

        dataset = TensorDataset(boards_t, pis_t, vs_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)

        self.nnet.train()
        total_loss = 0.0
        steps = 0
        import gc
        accum = getattr(self.args, 'accumulate_steps', 1)
        
        # Mixed precision (FP16) cho tiết kiệm VRAM
        use_amp = self.device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        # Gradient Checkpointing cho GPU nhỏ (< 6GB)
        use_checkpoint = self.device.type == 'cuda'
        
        for _ in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            accum_loss = 0
            for b_x, b_pi, b_v in loader:
                self.optimizer.zero_grad(set_to_none=True)
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        pi_pred, v_pred = self.nnet(b_x, use_checkpoint=use_checkpoint)
                        policy_loss = -torch.sum(b_pi * pi_pred, dim=1).mean()
                        value_loss = F.mse_loss(v_pred.view(-1), b_v.view(-1))
                        loss = (policy_loss + value_loss) / accum
                    scaler.scale(loss).backward()
                    if (n_batches + 1) % accum == 0:
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    pi_pred, v_pred = self.nnet(b_x, use_checkpoint=use_checkpoint)
                    policy_loss = -torch.sum(b_pi * pi_pred, dim=1).mean()
                    value_loss = F.mse_loss(v_pred.view(-1), b_v.view(-1))
                    loss = (policy_loss + value_loss) / accum
                    loss.backward()
                    if (n_batches + 1) % accum == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                
                del b_x, b_pi, b_v, pi_pred, v_pred, loss, policy_loss, value_loss
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                n_batches += 1
                steps += 1
            total_loss += epoch_loss / max(1, n_batches)
        avg_loss = total_loss / max(1, epochs)
        self.train_loss_history.append(avg_loss)
        return avg_loss

    # ----------------------------------------------------------------------
    def predict(self, canonical_board) -> Tuple[np.ndarray, float]:
        """Dự đoán policy & value cho 1 board (canonical form)."""
        from xiangqi.XiangqiGame import board_to_tensor
        state = board_to_tensor(canonical_board, _convert_player_to_int(1))
        x = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.nnet.eval()
        with torch.no_grad():
            pi_logits, v = self.nnet(x)
        pi = np.exp(pi_logits.cpu().numpy()[0])
        return pi, float(v.cpu().numpy()[0, 0])

    def predict_batch(self, canonical_boards) -> Tuple[np.ndarray, np.ndarray]:
        """Batch predict (cho self-play ở nhiều worker cùng lúc)."""
        from xiangqi.XiangqiGame import board_to_tensor
        states = np.stack(
            [board_to_tensor(b, _convert_player_to_int(1)) for b in canonical_boards],
            axis=0,
        ).astype(np.float32)
        x = torch.from_numpy(states).to(self.device)
        self.nnet.eval()
        with torch.no_grad():
            pi_logits, v = self.nnet(x)
        pi = np.exp(pi_logits.cpu().numpy())
        vs = v.cpu().numpy().flatten()
        return pi, vs

    # ----------------------------------------------------------------------
    def save_checkpoint(self, folder: str = "checkpoint", filename: str = "best.pth.tar"):
        filepath = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)
        torch.save({
            "state_dict": self.nnet.state_dict(),
            "args": self.args,
        }, filepath)

    def load_checkpoint(self, folder: str = "checkpoint", filename: str = "best.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint at {filepath}")
        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
        self.nnet.load_state_dict(ckpt["state_dict"])
        if "args" in ckpt:
            # Cập nhật các args mới từ checkpoint
            for k, v in ckpt["args"].items():
                self.args.setdefault(k, v)


def _convert_player_to_int(player) -> int:
    """Hỗ trợ cả int (1/-1) lẫn '1'/'2' string ký hiệu.

    Trả về 1 (canonical RED) trong context này.
    """
    if isinstance(player, str):
        return 1 if player in ("1", "red", "RED") else -1
    return int(player)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from xiangqi.XiangqiGame import XiangqiGame
    from utils import dotdict

    g = XiangqiGame()
    args = dotdict({"lr": 1e-3, "epochs": 2, "batch_size": 32,
                    "num_channels": 64, "num_res_blocks": 3,
                    "cuda": False})
    nn_wrap = NNetWrapper(g, args)
    b = g.getInitBoard()
    pi, v = nn_wrap.predict(b)
    print(f"pi sum={pi.sum():.4f}, v={v:.4f}")
    # Dummy train
    import numpy as np
    examples = [(b, np.ones(g.getActionSize()) / g.getActionSize(), 0.0) for _ in range(64)]
    loss = nn_wrap.train(examples)
    print(f"Train loss: {loss:.4f}")
    # Save/load
    nn_wrap.save_checkpoint("checkpoints", "test.pth.tar")
    nn_wrap.load_checkpoint("checkpoints", "test.pth.tar")
    print("Save/Load OK")