"""
XiangqiLogic.py
==============
Bộ sinh nước đi (move generator) và xác thực tính hợp lệ theo luật Cờ Tướng.

Hỗ trợ:
    - get_legal_moves(board, player): list (r1, c1, r2, c2)
    - is_legal_move(board, player, r1, c1, r2, c2): bool
    - find_general(board, color): (row, col) | None
    - is_in_check(board, color): bool
    - apply_move(board, r1, c1, r2, c2): board mới (deep copy)
"""
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple

from xiangqi.XiangqiConstants import (
    BOARD_ROWS,
    BOARD_COLS,
    EMPTY,
    RED,
    BLACK,
    _PIECE_OFFSET,
    PIECE_TYPE_GENERAL,
    PIECE_TYPE_ADVISOR,
    PIECE_TYPE_ELEPHANT,
    PIECE_TYPE_HORSE,
    PIECE_TYPE_CHARIOT,
    PIECE_TYPE_CANNON,
    PIECE_TYPE_SOLDIER,
    R_GENERAL, R_ADVISOR, R_ELEPHANT, R_HORSE,
    R_CHARIOT, R_CANNON, R_SOLDIER,
    B_GENERAL, B_ADVISOR, B_ELEPHANT, B_HORSE,
    B_CHARIOT, B_CANNON, B_SOLDIER,
    RED_PALACE_COLS,
    RED_PALACE_ROWS,
    BLACK_PALACE_COLS,
    BLACK_PALACE_ROWS,
    RIVER_ROW_RED_SIDE_END,
    RIVER_ROW_BLACK_SIDE_START,
    is_opponent,
    piece_color,
    piece_type_val,
)


Board = List[List[int]]
Move = Tuple[int, int, int, int]  # (r1, c1, r2, c2)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def in_board(r: int, c: int) -> bool:
    return 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS


def in_palace(color: int, r: int, c: int) -> bool:
    if color == RED:
        return r in RED_PALACE_ROWS and c in RED_PALACE_COLS
    else:
        return r in BLACK_PALACE_ROWS and c in BLACK_PALACE_COLS


def on_own_side(color: int, r: int) -> bool:
    """True nếu hàng r thuộc phần sân của color."""
    if color == RED:
        return r <= RIVER_ROW_RED_SIDE_END
    else:
        return r >= RIVER_ROW_BLACK_SIDE_START


def cross_river(color: int, r: int) -> bool:
    """True nếu r thuộc phần sân ĐỐI PHƯƠNG (đã qua sông)."""
    if color == RED:
        return r >= RIVER_ROW_BLACK_SIDE_START
    else:
        return r <= RIVER_ROW_RED_SIDE_END


def find_general(board: Board, color: int) -> Optional[Tuple[int, int]]:
    """Tìm vị trí Tướng của color."""
    gen_val = R_GENERAL if color == RED else B_GENERAL
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if board[r][c] == gen_val:
                return r, c
    return None


# ---------------------------------------------------------------------------
# Move generators
# ---------------------------------------------------------------------------
def _moves_general(board: Board, color: int, r: int, c: int) -> Iterable[Move]:
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if not in_palace(color, nr, nc):
            continue
        target = board[nr][nc]
        if target == EMPTY or piece_color(target) != color:
            yield (r, c, nr, nc)


def _moves_advisor(board: Board, color: int, r: int, c: int) -> Iterable[Move]:
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        nr, nc = r + dr, c + dc
        if not in_palace(color, nr, nc):
            continue
        target = board[nr][nc]
        if target == EMPTY or piece_color(target) != color:
            yield (r, c, nr, nc)


def _moves_elephant(board: Board, color: int, r: int, c: int) -> Iterable[Move]:
    """Tượng đi chéo 2 ô, không qua sông, MẮT TƯỢNG (ô giữa) trống."""
    for dr, dc in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
        nr, nc = r + dr, c + dc
        if not in_board(nr, nc):
            continue
        if not on_own_side(color, nr):
            continue
        mr, mc = r + dr // 2, c + dc // 2
        if board[mr][mc] != EMPTY:
            continue
        target = board[nr][nc]
        if target == EMPTY or piece_color(target) != color:
            yield (r, c, nr, nc)


def _moves_horse(board: Board, color: int, r: int, c: int) -> Iterable[Move]:
    """Mã đi chữ L: (1,2) hoặc (2,1). CHÂN MÃ (ô kề) phải trống."""
    candidates = (
        ((-2, -1), (-1, 0)), ((-2,  1), (-1, 0)),
        (( 2, -1), ( 1, 0)), (( 2,  1), ( 1, 0)),
        ((-1, -2), ( 0, -1)), (( 1, -2), ( 0, -1)),
        ((-1,  2), ( 0,  1)), (( 1,  2), ( 0,  1)),
    )
    for (dr, dc), (lr, lc) in candidates:
        nr, nc = r + dr, c + dc
        if not in_board(nr, nc):
            continue
        if board[r + lr][c + lc] != EMPTY:
            continue
        target = board[nr][nc]
        if target == EMPTY or piece_color(target) != color:
            yield (r, c, nr, nc)


def _moves_chariot(board: Board, color: int, r: int, c: int) -> Iterable[Move]:
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        while in_board(nr, nc):
            target = board[nr][nc]
            if target == EMPTY:
                yield (r, c, nr, nc)
            elif piece_color(target) != color:
                yield (r, c, nr, nc)
                break
            else:
                break
            nr += dr
            nc += dc


def _moves_cannon(board: Board, color: int, r: int, c: int) -> Iterable[Move]:
    """Pháo: đi đường thẳng (không ăn) như Xe; để ăn phải NHẢY qua ĐÚNG 1 quân.
    Sau khi nhảy, pháo có thể trượt tiếp qua các ô trống cho tới khi gặp quân
    tiếp theo - quân thứ hai này mới có thể bị ăn (đã nhảy qua đúng 1 quân).
    """
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        # Giai đoạn 1: trượt không ăn tới khi gặp quân
        while in_board(nr, nc) and board[nr][nc] == EMPTY:
            yield (r, c, nr, nc)
            nr += dr
            nc += dc
        if not in_board(nr, nc):
            continue
        # Gặp bàn đạp đầu tiên -> nhảy qua
        nr += dr
        nc += dc
        # Sau khi nhảy: trượt tiếp qua ô trống
        while in_board(nr, nc) and board[nr][nc] == EMPTY:
            yield (r, c, nr, nc)
            nr += dr
            nc += dc
        # Gặp quân thứ hai: nếu là quân địch thì ăn được
        if not in_board(nr, nc):
            continue
        target = board[nr][nc]
        if piece_color(target) != color:
            yield (r, c, nr, nc)


def _moves_soldier(board: Board, color: int, r: int, c: int) -> Iterable[Move]:
    """Tốt: trước khi qua sông chỉ đi thẳng. Qua sông được đi ngang; không lùi.
    RED ở row 0..4 -> đi về phía row lớn hơn (toward BLACK).
    BLACK ở row 5..9 -> đi về phía row nhỏ hơn (toward RED).
    """
    fwd = 1 if color == RED else -1
    # Đi thẳng về phía trước
    nr, nc = r + fwd, c
    if in_board(nr, nc):
        target = board[nr][nc]
        if target == EMPTY or piece_color(target) != color:
            yield (r, c, nr, nc)
    # Qua sông -> được đi ngang
    if cross_river(color, r):
        for dc in (-1, 1):
            nc2 = c + dc
            if not in_board(r, nc2):
                continue
            target = board[r][nc2]
            if target == EMPTY or piece_color(target) != color:
                yield (r, c, r, nc2)


# ---------------------------------------------------------------------------
# Move generators dict (key = piece value in board)
# ---------------------------------------------------------------------------
_PIECE_GENERATORS = {
    R_GENERAL: _moves_general, B_GENERAL: _moves_general,
    R_ADVISOR: _moves_advisor, B_ADVISOR: _moves_advisor,
    R_ELEPHANT: _moves_elephant, B_ELEPHANT: _moves_elephant,
    R_HORSE: _moves_horse, B_HORSE: _moves_horse,
    R_CHARIOT: _moves_chariot, B_CHARIOT: _moves_chariot,
    R_CANNON: _moves_cannon, B_CANNON: _moves_cannon,
    R_SOLDIER: _moves_soldier, B_SOLDIER: _moves_soldier,
}


# ---------------------------------------------------------------------------
# Kiểm tra chiếu / chống Tướng
# ---------------------------------------------------------------------------
def is_facing_generals(board: Board) -> bool:
    """Hai Tướng ở cùng cột dọc và không có quân chắn giữa."""
    red = find_general(board, RED)
    black = find_general(board, BLACK)
    if red is None or black is None:
        return False
    if red[1] != black[1]:
        return False
    r1, r2 = sorted((red[0], black[0]))
    for r in range(r1 + 1, r2):
        if board[r][red[1]] != EMPTY:
            return False
    return True


def is_in_check(board: Board, color: int) -> bool:
    """Tướng color đang bị chiếu.
    Lưu ý: KHÔNG xử lý trường hợp Tướng bị ăn ở đây.
    getGameEnded sẽ kiểm tra riêng.
    """
    gen_pos = find_general(board, color)
    if gen_pos is None:
        return False  # Tướng đã bị ăn; game đã kết thúc
    if is_facing_generals(board):
        return True
    gr, gc = gen_pos
    opp = -color
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            piece = board[r][c]
            if piece == EMPTY or piece_color(piece) != opp:
                continue
            gen_fn = _PIECE_GENERATORS.get(piece)
            if gen_fn is None:
                continue
            for (r1, c1, r2, c2) in gen_fn(board, opp, r, c):
                if r2 == gr and c2 == gc:
                    return True
    return False


# ---------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def get_pseudo_legal_moves(board: Board, player: int) -> List[Move]:
    """Tất cả nước đi theo luật di chuyển từng quân (không kiểm tra Tướng)."""
    moves: List[Move] = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            piece = board[r][c]
            if piece == EMPTY or piece_color(piece) != player:
                continue
            gen = _PIECE_GENERATORS.get(piece)
            if gen is not None:
                moves.extend(gen(board, player, r, c))
    return moves


def get_legal_moves(board: Board, player: int) -> List[Move]:
    """Nước đi hợp lệ: không tự chiếu Tướng."""
    out: List[Move] = []
    # Nếu Tướng đang bị chiếu, kiểm tra có nước thoát chiếu không
    if is_in_check(board, player):
        for r1, c1, r2, c2 in get_pseudo_legal_moves(board, player):
            b2 = apply_move(board, r1, c1, r2, c2)
            if not is_in_check(b2, player):
                out.append((r1, c1, r2, c2))
    else:
        # Không bị chiếu: chỉ cần không tự chiếu (luôn đúng cho non-check)
        out = [(r1, c1, r2, c2) for r1, c1, r2, c2 in get_pseudo_legal_moves(board, player)
               if not is_in_check(apply_move(board, r1, c1, r2, c2), player)]
    return out


def is_legal_move(board: Board, player: int,
                  r1: int, c1: int, r2: int, c2: int) -> bool:
    """Kiểm tra nhanh một nước đi có hợp lệ không."""
    if not (in_board(r1, c1) and in_board(r2, c2)):
        return False
    piece = board[r1][c1]
    if piece == EMPTY or piece_color(piece) != player:
        return False
    if r1 == r2 and c1 == c2:
        return False
    gen = _PIECE_GENERATORS.get(piece)
    if gen is None:
        return False
    for (rr1, cc1, rr2, cc2) in gen(board, player, r1, c1):
        if (rr2, cc2) == (r2, c2):
            b2 = apply_move(board, r1, c1, r2, c2)
            if not is_in_check(b2, player):
                return True
            return False
    return False


def apply_move(board: Board, r1: int, c1: int, r2: int, c2: int) -> Board:
    """Trả về bàn cờ MỚI (deep copy) sau khi áp dụng nước đi."""
    new_board = [row[:] for row in board]
    new_board[r2][c2] = new_board[r1][c1]
    new_board[r1][c1] = EMPTY
    return new_board


def move_to_string(r1: int, c1: int, r2: int, c2: int) -> str:
    """Chuyển (r1,c1) -> (r2,c2) ra ký hiệu WXF/Chinese."""
    cols = "abcdefghi"
    dr = r2 - r1
    dc = c2 - c1
    if dc == 0:
        act = "t" if dr > 0 else "退"
    elif dr == 0:
        act = "p平" if dr == 0 else ""
    elif abs(dr) == abs(dc):
        act = "t"
    elif abs(dr) == 1 and abs(dc) == 2:
        act = "x"
    elif abs(dr) == 2 and abs(dc) == 1:
        act = "m"
    else:
        act = ""
    return f"{cols[c2]}{act}{cols[c1]}"


# ---------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from xiangqi.XiangqiConstants import INITIAL_BOARD

    b = [row[:] for row in INITIAL_BOARD]
    legal = get_legal_moves(b, RED)
    print(f"Legal moves for RED at start: {len(legal)}")
    for mv in legal[:5]:
        print(" ", mv)
    if legal:
        b2 = apply_move(b, *legal[0])
        legal2 = get_legal_moves(b2, BLACK)
        print(f"Legal moves for BLACK: {len(legal2)}")
