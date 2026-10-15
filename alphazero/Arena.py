"""
Arena.py
========
Đấu giữa 2 player (thường là 2 neural net khác nhau hoặc neural net vs random).
"""
from __future__ import annotations
from typing import Callable


class Arena:
    """Đấu 2 player để đánh giá mạng."""

    def __init__(self, player1, player2, game, display: Callable = None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display or (lambda b: None)

    def playGame(self, verbose: bool = False) -> int:
        """Chơi 1 ván. Trả về 1 nếu player1 thắng, -1 nếu player2 thắng, ~0 nếu hoà."""
        board = self.game.getInitBoard()
        player = 1
        while True:
            if verbose:
                self.display(board)
            if player == 1:
                action = self.player1.play(board, 1)
            else:
                action = self.player2.play(board, -1)
            valids = self.game.getValidMoves(board, player)
            assert valids[action] == 1, f"Invalid action {action}"
            board, player = self.game.getNextState(board, player, action)
            r = self.game.getGameEnded(board, player)
            if r != 0:
                return r if player == 1 else -r

    def playGames(self, num: int, verbose: bool = False) -> tuple:
        """Chơi nhiều ván, đổi bên mỗi ván.
        Trả về (wins_p1, wins_p2, draws).
        """
        wins_p1 = wins_p2 = draws = 0
        for i in range(num):
            r = self.playGame(verbose=verbose)
            if r == 1:
                wins_p1 += 1
            elif r == -1:
                wins_p2 += 1
            else:
                draws += 1
        return wins_p1, wins_p2, draws
