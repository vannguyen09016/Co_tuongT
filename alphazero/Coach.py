"""
Coach.py
========
Vòng lặp chính của AlphaZero:
  - Self-play (MCTS + neural net) thu thập examples.
  - Train neural net.
  - Arena: so tài với model cũ.
  - Save best model.
"""
from __future__ import annotations
import os
import sys
import time
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from alphazero.MCTS import MCTS
from alphazero.Arena import Arena
from utils import dotdict


class Coach:
    """Vòng lặp self-play -> train -> arena."""

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(game, nnet, args)
        self.train_examples_history: List = []

    def executeEpisode(self) -> List[Tuple]:
        """Chạy một ván self-play hoàn chỉnh.
        Mỗi example = (canonical_board, pi, result).
        result = +1 nếu player của example thắng, -1 nếu thua.
        """
        examples = []
        board = self.game.getInitBoard()
        player = 1  # RED đi trước
        episode_step = 0
        temp_threshold = int(self.args.get("tempThreshold", 15))
        max_len = int(self.args.get("max_len", 200))

        while episode_step < max_len:
            episode_step += 1
            # Canonical form: từ góc nhìn của player hiện tại
            canonical_board, _ = self.game.getCanonicalForm(board, player)
            # Temperature: cao ở đầu game (khám phá), thấp ở cuối game (khai thác)
            temp_t = 1.0 if episode_step < temp_threshold else 0.0
            pi = self.mcts.getActionProb(canonical_board, temp=temp_t)
            # Data augmentation (symmetries)
            syms = self.game.getSymmetries(canonical_board, pi)
            for sym_board, sym_pi in syms:
                examples.append((sym_board, sym_pi, None))
            # Chọn action theo phân phối pi
            pi_sum = pi.sum()
            if pi_sum > 0:
                pi = pi / pi_sum
            else:
                # Fallback: uniform
                pi = np.ones(self.game.getActionSize()) / self.game.getActionSize()
            action = np.random.choice(self.game.getActionSize(), p=pi)
            board, next_player = self.game.getNextState(board, player, action)
            # Kiểm tra kết thúc game
            r = self.game.getGameEnded(board, next_player)
            if r != 0:
                # r = +1 if next_player won, -1 if lost
                for i in range(len(examples)):
                    stored_player = 1 if i % 2 == 0 else -1
                    examples[i] = (examples[i][0], examples[i][1],
                                  r if stored_player == next_player else -r)
                return examples
            player = next_player
        # Timeout -> hoà
        for i in range(len(examples)):
            examples[i] = (examples[i][0], examples[i][1], 1e-4)
        return examples

    def learn(self):
        """Vòng lap chính với streaming training."""
        from alphazero.MCTS import MCTS
        import gc
        
        # Auto-save interval (0 = off)
        save_every = int(self.args.get("save_every", 0))
        # Train sau moi N episodes de tranh tràn RAM (0 = train cuoi iteration)
        train_every = int(self.args.get("train_every", 10))
        total_eps = int(self.args.num_eps)
        
        for i in range(1, self.args.num_iters + 1):
            print(f"\n----- Iter {i}/{self.args.num_iters} -----")
            # Reset tree cache cho moi iteration
            self.mcts = MCTS(self.game, self.nnet, self.args)
            gc.collect()
            examples: List = []
            
            for ep in tqdm(range(total_eps), desc="SelfPlay"):
                ep_examples = self.executeEpisode()
                examples.extend(ep_examples)
                
                # Train ngay neu du N episodes
                if train_every > 0 and (ep + 1) % train_every == 0:
                    print(f"\n  Training on {len(examples)} examples...")
                    self.nnet.train(examples)
                    # Xoa examples cu de giai phong RAM
                    del examples
                    examples = []
                    # Clear MCTS cache sau training de giai phong RAM
                    self.mcts = MCTS(self.game, self.nnet, self.args)
                    gc.collect()
                
                # Auto-save checkpoint sau moi N van
                if save_every > 0 and (ep + 1) % save_every == 0:
                    print(f"\n  Auto-saving after {ep + 1}/{total_eps} episodes...")
                    self.nnet.save_checkpoint(
                        folder=self.args.checkpoint, 
                        filename=f"auto_iter{i}_ep{ep+1}.pth.tar"
                    )
                    gc.collect()
            
            # Train phan con lai neu co
            if len(examples) > 0:
                print(f"  Training on {len(examples)} remaining examples...")
                self.nnet.train(examples)
                del examples
            
            print(f"Iter {i} completed")
            gc.collect()
            
            # Save checkpoint cuoi iteration
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename=f"iter{i}.pth.tar")
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename="best.pth.tar")
            print(f"Saved checkpoint iter{i}.pth.tar")
