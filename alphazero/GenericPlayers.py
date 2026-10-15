"""
GenericPlayers.py
=================
Generic player interfaces:
  - RandomPlayer: chọn action ngẫu nhiên trong tập hợp lệ.
  - HumanPlayer: cho người chơi nhập từ bàn phím.
  - NNPlayer: bọc một NeuralNet + MCTS.
"""
from __future__ import annotations
import random
import sys
from typing import Optional

import numpy as np

from alphazero.MCTS import MCTS


class RandomPlayer:
    """Chọn action ngẫu nhiên trong valid moves."""

    def __init__(self, game):
        self.game = game

    def play(self, board, player: int) -> int:
        valids = self.game.getValidMoves(board, player)
        valid_indices = [i for i, v in enumerate(valids) if v == 1]
        return random.choice(valid_indices)


class HumanPlayer:
    """Cho người chơi nhập nước đi."""

    def __init__(self, game, input_fn=input):
        self.game = game
        self.input = input_fn

    def play(self, board, player: int) -> int:
        from xiangqi.XiangqiConstants import action_to_coord
        valids = self.game.getValidMoves(board, player)
        valid_indices = set(i for i, v in enumerate(valids) if v == 1)
        while True:
            print(f"Player {'RED' if player == 1 else 'BLACK'} turn.")
            raw = self.input("Enter move as 'r1 c1 r2 c2' (0-indexed): ").strip()
            try:
                parts = raw.split()
                if len(parts) != 4:
                    raise ValueError
                r1, c1, r2, c2 = map(int, parts)
                from xiangqi.XiangqiConstants import coord_to_action
                action = coord_to_action(r1, c1, r2, c2)
                if action not in valid_indices:
                    print("  INVALID move — not in valid set.")
                    continue
                return action
            except Exception:
                print("  Parse error. Format: r1 c1 r2 c2 (e.g., '0 0 1 0')")


class NNPlayer:
    """Neural net + MCTS player."""

    def __init__(self, game, nnet, args, mcts_cls=None):
        self.game = game
        self.nnet = nnet
        self.args = args
        mcts_cls = mcts_cls or MCTS
        self.mcts = mcts_cls(game, nnet, args)

    def play(self, board, player: int) -> int:
        canonical_board, _ = self.game.getCanonicalForm(board, player)
        probs = self.mcts.getActionProb(canonical_board, temp=0)
        return int(np.argmax(probs))

    def reset(self):
        """Reset MCTS tree cache (cho ván mới)."""
        self.mcts = type(self.mcts)(self.game, self.nnet, self.args)