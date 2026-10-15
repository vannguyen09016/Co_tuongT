"""
XiangqiNNet.py
==============
Kiến trúc mạng nơ-ron cho Cờ Tướng theo phong cách AlphaZero.

Thiết kế:
    - Input:  (B, C, 10, 9)  với C = 15 (7 loại quân đỏ + 7 loại quân đen + 1 lượt).
    - Backbone: Conv2D (3x3) -> n Residual Blocks (3x3, output channels 256).
    - Policy Head:
          Conv2D 3x3 (256 channels) -> BN -> ReLU
          Conv2D 1x1 (73 channels, ...) -> Flatten
          -> Linear (action_size=8100) -> log_softmax
    - Value Head:
          Conv2D 1x1 (16 channels) -> BN -> ReLU
          Flatten -> Linear(256) -> ReLU -> Linear(1) -> tanh

Lưu ý về padding:
    * Cờ Tướng 10x9 có kích thước nhỏ, dùng padding="same" để giữ nguyên shape
      xuyên suốt mạng (chỉ áp dụng cho mạng policy zero cần depth+2).
"""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels: int, out_channels: int, bias: bool = False) -> nn.Conv2d:
    """3x3 convolution với padding 1 để giữ nguyên kích thước."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)


class ResBlock(nn.Module):
    """Residual block cổ điển (AlphaZero/AlphaGo).

    Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN -> add skip -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Skip connection: 1x1 conv nếu số channel thay đổi
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.skip(x)
        out = F.relu(out, inplace=True)
        return out

    def forward_with_checkpoint(self, x: torch.Tensor) -> torch.Tensor:
        """Forward với gradient checkpointing - tiết kiệm VRAM."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.skip(x)
        out = F.relu(out, inplace=True)
        return out


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------
class XiangqiNNet(nn.Module):
    """ResNet policy + value network cho Cờ Tướng."""

    def __init__(self,
                 board_rows: int = 10,
                 board_cols: int = 9,
                 in_channels: int = 15,
                 action_size: int = 8100,
                 num_channels: int = 256,
                 num_res_blocks: int = 5,
                 policy_hidden: int = 256,
                 value_hidden: int = 256):
        """
        Args:
            board_rows: số hàng (10).
            board_cols: số cột (9).
            in_channels: số kênh input (15 = 7 loại quân x2 + 1 lượt).
            action_size: số action có thể (8100).
            num_channels: số kênh của conv chính trong ResBlocks (default 256).
            num_res_blocks: số ResBlock xếp chồng (5..19 tuỳ dự án).
        """
        super().__init__()
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.action_size = action_size
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks

        # ---------- Backbone ----------
        # Initial conv: 3x3, num_channels, padding=1 (giữ nguyên shape)
        self.conv_input = nn.Sequential(
            conv3x3(in_channels, num_channels),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

        # Stack of residual blocks
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_channels, num_channels) for _ in range(num_res_blocks)]
        )

        # ---------- Policy head ----------
        # Conv 3x3 -> BN -> ReLU -> Conv 1x1 (2 channels = move + pass? ta dùng 1)
        # Đơn giản: dùng 1 filter ở layer cuối rồi flatten ra action_size.
        # Phương án phổ biến: Conv 3x3 num_channels -> BN -> ReLU
        #                       Conv 1x1 -> action_size (depth đầu vào cuối cùng)
        #                       -> Softmax cho output
        # Tuy nhiên, Conv 1x1 với output channels = action_size = 8100 tốn kém.
        # Cchess-zero dùng Conv 1x1 với output=action_size (1 phổ biến).
        self.policy_conv = nn.Sequential(
            conv3x3(num_channels, num_channels),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        # 1x1 conv chiếu sang 2 channels rồi FC -> action_size.
        # Lưu ý: action_size = 8100 khá lớn, ta dùng FC trên flatten.
        # Phương án an toàn: 1x1 conv với 1 channel (or hidden) rồi FC.
        self.policy_conv_out = conv3x3(num_channels, policy_hidden)
        self.policy_bn = nn.BatchNorm2d(policy_hidden)
        self.policy_relu = nn.ReLU(inplace=True)
        # Flatten -> Linear -> log_softmax
        self.policy_fc = nn.Linear(policy_hidden * board_rows * board_cols,
                                   action_size)

        # ---------- Value head ----------
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(16 * board_rows * board_cols, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        # ---------- Init weights ----------
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
        # Đặt value_fc2 với trọng số nhỏ để ban đầu v ~ 0
        nn.init.uniform_(self.value_fc2.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.value_fc2.bias)

    def forward(self, x: torch.Tensor, use_checkpoint: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: tensor (B, in_channels, board_rows, board_cols).
            use_checkpoint: nếu True, dùng gradient checkpointing để tiết kiệm VRAM.
        Returns:
            pi_logits: (B, action_size) raw logits (KHÔNG softmax).
            v: (B, 1) trong [-1, 1].
        """
        # Backbone
        out = self.conv_input(x)
        if use_checkpoint:
            # Gradient checkpointing: chia res_blocks thành từng checkpoint
            for i, rb in enumerate(self.res_blocks):
                if i % 2 == 0:
                    out = torch.utils.checkpoint.checkpoint(
                        rb.forward_with_checkpoint, out, use_reentrant=False
                    )
                else:
                    out = rb(out)
        else:
            for rb in self.res_blocks:
                out = rb(out)

        # Policy head
        p = self.policy_conv(out)
        p = self.policy_conv_out(p)
        p = self.policy_bn(p)
        p = self.policy_relu(p)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        # Trả về log_softmax để dùng với NLLLoss.
        pi = F.log_softmax(p, dim=1)

        # Value head
        v = self.value_conv(out)
        v = v.view(v.size(0), -1)
        v = self.value_fc1(v)
        v = F.relu(v, inplace=True)
        v = self.value_fc2(v)
        v = torch.tanh(v)
        return pi, v


# ===========================================================================
# Self-test
# ===========================================================================
if __name__ == "__main__":
    # Chỉ import nếu chạy trực tiếp.
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("XiangqiNNet basic test (CPU)")
    net = XiangqiNNet(board_rows=10, board_cols=9,
                      in_channels=15, action_size=8100,
                      num_channels=64, num_res_blocks=3)
    x = torch.randn(2, 15, 10, 9)
    pi, v = net(x)
    print("pi shape:", tuple(pi.shape), "(expect (2, 8100))")
    print("v  shape:", tuple(v.shape),  "(expect (2, 1))")
    print("v  values:", v.flatten().tolist())
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
    assert pi.shape == (2, 8100)
    assert v.shape == (2, 1)
    print("OK")