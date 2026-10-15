"""
XiangqiPlayers.py
=================
Player thường dùng cho Xiangqi.
"""
from __future__ import annotations
import random
from typing import Any


class RandomPlayer:
    """Chọn action ngẫu nhiên trong valid moves."""

    def __init__(self, game):
        self.game = game

    def play(self, board: Any) -> int:
        """Trả về action index. `board` ở đây là canonical form
        (player=+1).
        """
        valids = self.game.getValidMoves(board, 1)
        valid_indices = [i for i, v in enumerate(valids) if v == 1]
        return random.choice(valid_indices)

    def reset(self):
        pass