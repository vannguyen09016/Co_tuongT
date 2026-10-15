"""
XiangqiGame.py
==============
Game class cho Cờ Tướng (Xiangqi) tương thích alpha-zero-general.

Kế thừa `Game` (alpha-zero-general interface). Đáp ứng yêu cầu:
  * Bàn cờ 9x10.
  * Luật đầy đủ (Tượng không qua sông, Sĩ/Tướng không rời cung, Mã cản chân,
    Pháo nhảy qua 1 quân, Tốt qua sông mới đi ngang, chống Tướng).
  * Chiếu bí (checkmate), stalemate (bên hết nướng bị thua), lặp bàn cờ 3 lần
    (draw/penalty).
  * State representation: tensor PyTorch dạng (C, 10, 9) — tương thích CNN.

Quy ước bàn cờ (nhất quán với XiangqiConstants):
  * board[r][c], r ∈ [0..9], c ∈ [0..8]
  * Quân đỏ ở row 0..4 (top half); Quân đen ở row 5..9.
  * RED (player=+1) đi trước.
  * board là list-of-lists int. Quân cờ dương = đỏ, âm = đen, 0 = trống.

Action encoding (cố định cho toàn dự án):
  action = r1 * (BOARD_COLS * BOARD_ROWS * BOARD_COLS)
         + c1 * (BOARD_ROWS * BOARD_COLS)
         + r2 * (BOARD_COLS)
         + c2
  với (r1, c1) là vị trí nguồn, (r2, c2) là vị trí đích.
  Phạm vi: 0..8099 (8100 actions tổng).

Neural network input tensor (channels, 10, 9):
  * Channel 0..6: các loại quân đỏ (General, Advisor, Elephant, Horse, Chariot,
                   Cannon, Soldier).
  * Channel 7..13: các loại quân đen (cùng thứ tự).
  * Channel 14: lượt đi (1 = RED, 0 = BLACK) — thực tế mỗi lớp thể hiện lượt
                 của player hiện tại; khi gặp mạng ở góc nhìn RED thì channel
                 này luôn là 1 (xem getCanonicalForm).
  Tổng cộng 15 channels.

Lưu ý: Khi dùng cho self-play, action list được mask qua getValidMoves().
"""
from __future__ import annotations
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from Game import Game
from xiangqi.XiangqiConstants import (
    BOARD_ROWS, BOARD_COLS, ACTION_SIZE,
    EMPTY, RED, BLACK,
    PIECE_TYPE_GENERAL, PIECE_TYPE_ADVISOR, PIECE_TYPE_ELEPHANT,
    PIECE_TYPE_HORSE, PIECE_TYPE_CHARIOT, PIECE_TYPE_CANNON,
    PIECE_TYPE_SOLDIER, NUM_PIECE_TYPES,
    INITIAL_BOARD,
    piece_color, piece_type_val,
    action_to_coord, coord_to_action,
)
from xiangqi.XiangqiLogic import (
    get_legal_moves,
    is_legal_move,
    apply_move,
    find_general,
    is_in_check,
)


# ---------------------------------------------------------------------------
# Bàn cờ dưới dạng tuple-of-tuples (immutable) để PyTorch/Python không cảnh báo
# khi share giữa nhiều processes trong multiprocessing data loader.
# ---------------------------------------------------------------------------
def _deep_tuple(b: tuple) -> tuple:
    """Đệ quy chuyển list-of-lists -> tuple-of-tuples."""
    return tuple(_deep_tuple(row) if isinstance(row, (list, tuple)) else row
                 for row in b)


def _list_board(b) -> list[list[int]]:
    """Chuyển tuple-of-tuples -> list-of-lists để dễ thao tác."""
    return [list(row) for row in b]


# ---------------------------------------------------------------------------
# State representation (tensor)
# ---------------------------------------------------------------------------
NUM_PLANES_PER_SIDE = NUM_PIECE_TYPES  # 7 quân / phe
# 7 kênh đỏ + 7 kênh đen + 1 kênh lượt đi = 15 kênh.
NUM_STATE_PLANES = NUM_PLANES_PER_SIDE * 2 + 1  # 15


def board_to_tensor(board, player: int) -> np.ndarray:
    """Chuyển bàn cờ thành tensor numpy (15, 10, 9).
    Mỗi channel là một lớp nhị phân.

    Lưu ý: đây là chế độ góc nhìn thuần. Việc lật bàn cờ để nhìn từ phía
    player hiện tại được thực hiện trong `getCanonicalForm`.
    """
    tensor = np.zeros((NUM_STATE_PLANES, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            piece = board[r][c]
            if piece == EMPTY:
                continue
            color = piece_color(piece)
            ptype = piece_type_val(piece)  # 1..7
            if color == RED:
                ch = ptype - 1  # 0..6
            else:
                ch = NUM_PLANES_PER_SIDE + (ptype - 1)  # 7..13
            tensor[ch, r, c] = 1.0
    # Channel lượt đi
    turn_plane = NUM_STATE_PLANES - 1
    if player == RED:
        tensor[turn_plane, :, :] = 1.0
    return tensor


def flip_board_vertically(board) -> tuple:
    """Lật bàn cờ theo trục dọc (đảo row): đổi góc nhìn từ "đỏ đi trước, row 0 là
    phía đỏ" sang "BLACK đi trước, row 0 là phía BLACK".
    Đồng thời đảo dấu quân (đỏ ↔ đen).
    """
    new_board = tuple(tuple(-board[BOARD_ROWS - 1 - r][c] for c in range(BOARD_COLS))
                      for r in range(BOARD_ROWS))
    return new_board


# ===========================================================================
# Main Game class
# ===========================================================================
class XiangqiGame(Game):
    """Game class cho Cờ Tướng theo alpha-zero-general interface.

    Board là tuple-of-tuples (immutable) để tránh side-effect giữa các process.
    """

    # ----------------------------------------------------------------------
    # Required overrides
    # ----------------------------------------------------------------------
    def getInitBoard(self) -> tuple:
        """Trả về bàn cờ khởi đầu (tuple-of-tuples)."""
        return _deep_tuple(INITIAL_BOARD)

    def getBoardSize(self) -> Tuple[int, int]:
        return (BOARD_ROWS, BOARD_COLS)

    def getActionSize(self) -> int:
        return ACTION_SIZE

    def getNextState(self, board, player: int, action: int):
        """Áp dụng action, trả về (board_mới, player_mới).

        Lưu ý quan trọng: KHÔNG mutate board cũ. Trả về tuple mới.
        """
        r1, c1, r2, c2 = action_to_coord(action)
        b_list = _list_board(board)
        b_new = apply_move(b_list, r1, c1, r2, c2)
        return _deep_tuple(b_new), -player

    def getValidMoves(self, board, player: int) -> List[int]:
        """Trả về mask action hợp lệ dạng list of 0/1 (độ dài = action_size).

        Chú ý: trả về list[int] (không numpy) để tương thích với code MCTS trong
        alpha-zero-general.
        """
        b_list = _list_board(board)
        moves = get_legal_moves(b_list, player)
        valid = [0] * ACTION_SIZE
        for r1, c1, r2, c2 in moves:
            valid[coord_to_action(r1, c1, r2, c2)] = 1
        return valid

    def getGameEnded(self, board, player: int) -> int:
        """Trả về trạng thái kết thúc game.
        player là player HIỆN TẠI (sắp đi).
          +1: player thắng (đối thủ bị chiếu bí)
          -1: player thua (bị chiếu bí)
          ~1e-4: hoà
          0: chưa kết thúc
        """
        b_list = _list_board(board)
        moves = get_legal_moves(b_list, player)
        
        # Checkmate: đang bị chiếu và không có nước thoát
        if len(moves) == 0:
            # Nếu đang bị chiếu -> chiếu bí -> thua
            if is_in_check(b_list, player):
                return -1  # player sắp đi thua
            else:
                # Stalemate (hết nước nhưng không bị chiếu) -> hoà
                return 1e-4
        return 0

    def getCanonicalForm(self, board, player: int) -> Tuple[Any, int]:
        """Trả về (canonical_board, canonical_player).

        - Nếu player == RED, trả về (board, +1).
        - Nếu player == BLACK, lật bàn cờ dọc theo row (và đổi dấu quân) sao
          cho góc nhìn từ phía player giống như RED. Canonical_player luôn là +1.
        """
        if player == RED:
            return board, 1
        return flip_board_vertically(board), 1

    def getSymmetries(self, board, pi) -> List[Tuple[Tuple, np.ndarray]]:
        """Trả về các phép đối xứng để data augmentation.

        Vì Cờ Tướng có tính chất bất đối xứng (cung đỏ vs cung đen ở 2 đầu),
        ta chỉ áp dụng DUY NHẤT phép đối xứng lật trái-phải (fliplr) trên
        canonical board. Đây cũng là thông lệ của các dự án cchess-zero.

        Lật trái-phải:
            (r, c) -> (r, 8 - c)
        Channel lượt đi (channel 14) không đổi.
        """
        # Lật trái phải: chuyển (r, c) -> (r, 8 - c)
        sym_board = tuple(tuple(board[r][BOARD_COLS - 1 - c] for c in range(BOARD_COLS))
                          for r in range(BOARD_ROWS))
        # pi shape: (action_size,) hoặc (action_size,). Lật theo col đích/và col nguồn.
        pi_arr = np.asarray(pi)
        sym_p = np.zeros_like(pi_arr)
        for a in range(ACTION_SIZE):
            r1, c1, r2, c2 = action_to_coord(a)
            new_c1 = BOARD_COLS - 1 - c1
            new_c2 = BOARD_COLS - 1 - c2
            new_a = coord_to_action(r1, new_c1, r2, new_c2)
            sym_p[new_a] = pi_arr[a]
        return [(sym_board, sym_p)]

    # ----------------------------------------------------------------------
    # Optional overrides
    # ----------------------------------------------------------------------
    def getScore(self, board, player: int) -> float:
        """Heuristic score cho arena/evaluation. Trả về material balance."""
        b_list = _list_board(board)
        val = {PIECE_TYPE_GENERAL: 10000, PIECE_TYPE_ADVISOR: 20,
               PIECE_TYPE_ELEPHANT: 20, PIECE_TYPE_HORSE: 40,
               PIECE_TYPE_CHARIOT: 90, PIECE_TYPE_CANNON: 45,
               PIECE_TYPE_SOLDIER: 10}
        score = 0
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                p = b_list[r][c]
                if p == EMPTY:
                    continue
                s = val[piece_type_val(p)]
                if piece_color(p) == player:
                    score += s
                else:
                    score -= s
        return score

    def getEndingTurn(self) -> int:
        """Số turn tối đa cho mỗi ván. Cờ Tướng có thể kéo dài nhưng thường
        giới hạn ~ 200..300 turn. Trả về 0 nghĩa là không giới hạn.
        """
        return 200

    def stringRepresentation(self, board) -> str:
        """Dùng cho cache NeuralNetwork predictions.
        Trả về string tuple-of-tuples — gọn nhẹ và hashable.
        """
        return str(board)

    # ----------------------------------------------------------------------
    # Helpers (dùng trong neural network pipeline)
    # ----------------------------------------------------------------------
    def getStateTensor(self, board, player: int) -> torch.Tensor:
        """Trả về torch.Tensor (1, C, 10, 9) cho canonical state.
        Dùng để truyền vào neural network.
        """
        canonical_board, _ = self.getCanonicalForm(board, player)
        arr = board_to_tensor(canonical_board, RED)
        return torch.from_numpy(arr).unsqueeze(0)


# ===========================================================================
# Convenience aliases để dùng ngoài
# ===========================================================================
display = None  # tránh circular


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    g = XiangqiGame()
    b = g.getInitBoard()
    p = RED
    print("Board size:", g.getBoardSize())
    print("Action size:", g.getActionSize())
    print("Initial board (player=RED):")
    g.display(b)
    valid = g.getValidMoves(b, p)
    print(f"\nRED legal moves count: {sum(valid)}")
    # Test apply one move
    legal_idx = valid.index(1)
    b2, _ = g.getNextState(b, p, legal_idx)
    print(f"\nAfter action {legal_idx}, board:")
    for row in b2:
        print(" ", row[:5], "...")  # truncated
    # Test canonical form
    cb, cp = g.getCanonicalForm(b, RED)
    print(f"\nCanonical form (player=RED): canonical_player = {cp}")
    cb2, cp2 = g.getCanonicalForm(b, BLACK)
    print(f"Canonical form (player=BLACK): canonical_player = {cp2}")
    # Test state tensor
    t = g.getStateTensor(b, RED)
    print(f"\nState tensor shape: {tuple(t.shape)}")