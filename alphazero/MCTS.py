"""
MCTS.py
=======
Monte Carlo Tree Search cho AlphaZero với Dirichlet noise ở gốc.
Tương thích với mọi Game kế thừa alpha-zero-general.
"""
from __future__ import annotations
import sys
import math
from typing import List, Tuple

import numpy as np

# Tăng recursion limit nếu cần
if sys.getrecursionlimit() < 10000:
    sys.setrecursionlimit(10000)


class MCTS:
    """MCTS lưu trử tree state bang dict-of-dicts."""

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa: dict = {}      # Q values
        self.Nsa: dict = {}      # edge visit counts
        self.Ns: dict = {}       # state visit counts
        self.Ps: dict = {}       # policy prve (cached neural net policy)
        self.Vs: dict = {}       # cached game ended value (1/-1)
        # Endgame state cache (game ended == True) - value of state.
        self.Es: dict = {}       # ended flag
        # Gioi han de quy de tránh stack overflow
        self.max_search_depth = 1000
        # Gioi han cache de tránh memory leak
        self.max_cache_size = 3000
    
    def _cleanup_if_needed(self):
        """Don rac neu cache qua lon."""
        total = len(self.Ps)
        if total > self.max_cache_size:
            # Xoa 50% cache cu nhat
            keys = list(self.Ps.keys())
            for key in keys[:total // 2]:
                self.Ps.pop(key, None)
                self.Qsa.pop(key, None)
                self.Ns.pop(key, None)
                self.Es.pop(key, None)
                self.Vs.pop(key, None)
                # Xoa Nsa lien quan
                for a in range(self.game.getActionSize()):
                    self.Nsa.pop((key, a), None)

    def getActionProb(self, canonical_board, temp: float = 1.0) -> np.ndarray:
        """Tra ve policy vector (action_size,) tu MCTS.
        
        temp: temperature. temp=0 -> greedy, temp=1 -> proportional to N^(1/temp).
        """
        # Don rac neu can
        self._cleanup_if_needed()
        
        for _ in range(self.args.get("num_mcts_sims", 50)):
            self.search(canonical_board, depth=0)

        s = self.game.stringRepresentation(canonical_board)
        counts = np.zeros(self.game.getActionSize(), dtype=np.float32)
        for a in range(self.game.getActionSize()):
            if (s, a) in self.Nsa:
                counts[a] = self.Nsa[(s, a)]

        if temp == 0:
            best_as = np.array_equal(counts, np.zeros_like(counts))
            if best_as:
                # Không có nước đi (rare).
                probs = counts
            else:
                best_a = int(np.argmax(counts))
                probs = np.zeros_like(counts)
                probs[best_a] = 1.0
            return probs
        counts = counts ** (1.0 / max(1e-8, temp))
        total = counts.sum()
        if total > 0:
            return counts / total
        # uniform fallback
        return np.ones_like(counts) / self.game.getActionSize()

    def search(self, canonical_board, depth=0) -> float:
        """Một lần simulation bắt đầu từ canonical_board.
        Trả về leaf value v.
        """
        # Giới hạn độ sâu đệ quy
        depth += 1
        if depth > self.max_search_depth:
            # Force endgame check - coi như game tiếp tục với giá trị 0
            return 0.0
        
        s = self.game.stringRepresentation(canonical_board)

        # Endgame: trả về kết quả.
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonical_board, 1)
        if self.Es[s] != 0:
            return self.Es[s]

        # Expand: chưa visit -> evaluate bằng NN, lưu Ps.
        if s not in self.Ps:
            # Neural network policy + value
            self.Ps[s], v = self.nnet.predict(canonical_board)
            # Dam bao dtype = float32 de tiet kiem RAM
            self.Ps[s] = self.Ps[s].astype(np.float32)
            valids = self.game.getValidMoves(canonical_board, 1)
            # Dam bao valids la numpy array
            if not isinstance(valids, np.ndarray):
                valids = np.array(valids, dtype=np.float32)
            else:
                valids = valids.astype(np.float32)
            self.Ps[s] = self.Ps[s] * valids
            sum_Ps = np.sum(self.Ps[s])
            if sum_Ps > 0:
                self.Ps[s] /= sum_Ps
            else:
                # fallback: uniform over valids
                self.Ps[s] = np.array(valids, dtype=np.float32)
                self.Ps[s] /= np.sum(self.Ps[s])
            self.Vs[s] = valids
            self.Ns[s] = 0
            return v

        # Select: PUCT
        cur_best = -float('inf')
        best_act = -1
        valid = self.Vs[s]
        # Dirichlet noise at root
        for a in range(self.game.getActionSize()):
            if valid[a]:
                # Dam bao Nsa co key truoc khi truy cap
                if (s, a) in self.Qsa and (s, a) in self.Nsa and (s, a) in self.Nsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * math.sqrt(self.Ns[s]) \
                        * self.Ps[s][a] / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * math.sqrt(self.Ns[s]) \
                        * self.Ps[s][a]
                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_board, _ = self.game.getNextState(canonical_board, 1, a)
        next_board_canon, _ = self.game.getCanonicalForm(next_board, -1)
        v = self.search(next_board_canon, depth=depth)

        # Backpropagate
        if (s, a) in self.Qsa and (s, a) in self.Nsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        return -v


class MCTSWithDirichletNoise(MCTS):
    """MCTS với Dirichlet noise ở root (theo AlphaZero paper)."""

    def getActionProb(self, canonical_board, temp: float = 1.0) -> np.ndarray:
        # Add Dirichlet noise to the root policy once.
        s = self.game.stringRepresentation(canonical_board)
        if s not in self.Ps:
            self.search(canonical_board)  # expand root
        # Add noise
        eps = self.args.get("dirichlet_epsilon", 0.25)
        alpha = self.args.get("dirichlet_alpha", 0.3)
        valid = self.Vs.get(s, self.game.getValidMoves(canonical_board, 1))
        noise = np.random.dirichlet([alpha] * int(np.sum(valid)))
        new_ps = (1 - eps) * self.Ps[s] + eps * noise
        # Zero out invalid moves
        new_ps = new_ps * valid
        sum_new = new_ps.sum()
        if sum_new > 0:
            self.Ps[s] = new_ps / sum_new
        # Run more simulations
        for _ in range(self.args.get("numMCTSSims", 50) - 1):
            self.search(canonical_board)
        counts = np.zeros(self.game.getActionSize(), dtype=np.float32)
        for a in range(self.game.getActionSize()):
            if (s, a) in self.Nsa:
                counts[a] = self.Nsa[(s, a)]
        if temp == 0:
            best_a = int(np.argmax(counts))
            probs = np.zeros_like(counts)
            probs[best_a] = 1.0
            return probs
        counts = counts ** (1.0 / max(1e-8, temp))
        total = counts.sum()
        if total > 0:
            return counts / total
        return np.ones_like(counts) / self.game.getActionSize()