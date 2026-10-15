"""
XiangqiConstants.py
==================
Hằng số & bảng mã quân cho Cờ Tướng.

QUY ƯỚC MÃ QUÂN:
    - Mỗi ô trên bàn cờ lưu một số nguyên.
    - EMPTY = 0.
    - RED pieces:  11, 12, 13, 14, 15, 16, 17
    - BLACK pieces: -11, -12, -13, -14, -15, -16, -17
    Lý do dùng 11-17 thay vì 1-7: tránh xung đột với EMPTY=0 khi BLACK=-1.
      (Nếu dùng type 1-7, BLACK + GENERAL = -1 + 1 = 0 = EMPTY → bug.)

    RED = 1, BLACK = -1  (chỉ dùng làm hệ số nhân màu).
    Các hằng PIECE_TYPE_* chỉ dùng trong network input channel indexing.

Bàn cờ: 10 hàng × 9 cột (row 0..9, col 0..8).
    Row 0..4: phía đỏ (RED), Row 5..9: phía đen (BLACK).
    RED đi trước (player=1).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Kích thước bàn cờ
# ---------------------------------------------------------------------------
BOARD_ROWS = 10
BOARD_COLS = 9
ACTION_SIZE = BOARD_ROWS * BOARD_COLS * BOARD_ROWS * BOARD_COLS  # 8100

# ---------------------------------------------------------------------------
# Màu
# ---------------------------------------------------------------------------
EMPTY = 0
RED = 1
BLACK = -1

# ---------------------------------------------------------------------------
# Piece TYPEs (chỉ dùng làm channel index trong neural network).
# Thứ tự 1..7 dùng trực tiếp làm channel 0..6 trong input tensor.
# ---------------------------------------------------------------------------
PIECE_TYPE_GENERAL = 1
PIECE_TYPE_ADVISOR = 2
PIECE_TYPE_ELEPHANT = 3
PIECE_TYPE_HORSE = 4
PIECE_TYPE_CHARIOT = 5
PIECE_TYPE_CANNON = 6
PIECE_TYPE_SOLDIER = 7
NUM_PIECE_TYPES = 7

# Giá trị offset tránh xung đột: đỏ = +10, đen = -10.
# Quân đỏ: 11..17, Quân đen: -11..-17, EMPTY = 0.
_PIECE_OFFSET = 10

def make_piece(piece_type_val: int, color: int) -> int:
    """Tạo giá trị quân: color * (_PIECE_OFFSET + piece_type)."""
    return color * (_PIECE_OFFSET + piece_type_val)

# Convenience constants cho bàn cờ (dùng trong init board).
# RED piece values
R_GENERAL  = make_piece(PIECE_TYPE_GENERAL,   RED)
R_ADVISOR  = make_piece(PIECE_TYPE_ADVISOR,   RED)
R_ELEPHANT = make_piece(PIECE_TYPE_ELEPHANT,  RED)
R_HORSE    = make_piece(PIECE_TYPE_HORSE,     RED)
R_CHARIOT  = make_piece(PIECE_TYPE_CHARIOT,   RED)
R_CANNON   = make_piece(PIECE_TYPE_CANNON,    RED)
R_SOLDIER  = make_piece(PIECE_TYPE_SOLDIER,  RED)
# BLACK piece values
B_GENERAL  = make_piece(PIECE_TYPE_GENERAL,   BLACK)
B_ADVISOR  = make_piece(PIECE_TYPE_ADVISOR,   BLACK)
B_ELEPHANT = make_piece(PIECE_TYPE_ELEPHANT,  BLACK)
B_HORSE    = make_piece(PIECE_TYPE_HORSE,     BLACK)
B_CHARIOT  = make_piece(PIECE_TYPE_CHARIOT,   BLACK)
B_CANNON   = make_piece(PIECE_TYPE_CANNON,    BLACK)
B_SOLDIER  = make_piece(PIECE_TYPE_SOLDIER,   BLACK)

# ---------------------------------------------------------------------------
# Hằng số luật chơi
# ---------------------------------------------------------------------------
NO_REPEAT_COUNT = 3
MAX_GAME_LENGTH = 500
REPETITION_DRAW_PENALTY = 1e-4

# Cung điện (cung Tướng): mỗi bên 3x3 ở chính giữa.
RED_PALACE_COLS = (3, 4, 5)
RED_PALACE_ROWS = (0, 1, 2)
BLACK_PALACE_COLS = (3, 4, 5)
BLACK_PALACE_ROWS = (7, 8, 9)

RIVER_ROW_RED_SIDE_END = 4
RIVER_ROW_BLACK_SIDE_START = 5

# ---------------------------------------------------------------------------
# Bàn cờ khởi đầu
# ---------------------------------------------------------------------------
def _build_initial_board() -> list[list[int]]:
    b = [[EMPTY for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
    # RED (row 0..4)
    b[0][0] = R_CHARIOT;  b[0][1] = R_HORSE;   b[0][2] = R_ELEPHANT
    b[0][3] = R_ADVISOR;  b[0][4] = R_GENERAL
    b[0][5] = R_ADVISOR;  b[0][6] = R_ELEPHANT; b[0][7] = R_HORSE;  b[0][8] = R_CHARIOT
    b[2][1] = R_CANNON;   b[2][7] = R_CANNON
    for c in (0, 2, 4, 6, 8):
        b[3][c] = R_SOLDIER
    # BLACK (row 5..9)
    b[9][0] = B_CHARIOT;  b[9][1] = B_HORSE;   b[9][2] = B_ELEPHANT
    b[9][3] = B_ADVISOR;  b[9][4] = B_GENERAL
    b[9][5] = B_ADVISOR;  b[9][6] = B_ELEPHANT; b[9][7] = B_HORSE;  b[9][8] = B_CHARIOT
    b[7][1] = B_CANNON;   b[7][7] = B_CANNON
    for c in (0, 2, 4, 6, 8):
        b[6][c] = B_SOLDIER
    return b


INITIAL_BOARD = _build_initial_board()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def piece_type_val(piece: int) -> int:
    """Trả về loại quân (1..7). EMPTY=0."""
    if piece == EMPTY:
        return 0
    return abs(piece) - _PIECE_OFFSET


def piece_color(piece: int) -> int:
    """Trả về màu (+1, -1). EMPTY -> 0."""
    if piece == EMPTY:
        return 0
    return 1 if piece > 0 else -1


def is_opponent(a: int, b: int) -> bool:
    """a, b có khác phe và không phải EMPTY."""
    ca, cb = piece_color(a), piece_color(b)
    return ca != 0 and cb != 0 and ca != cb


# ---------------------------------------------------------------------------
# Encode / Decode action
# ---------------------------------------------------------------------------
def action_to_coord(action: int):
    """Chuyển action index -> (r1, c1, r2, c2)."""
    a = action
    c2 = a % BOARD_COLS;       a //= BOARD_COLS
    r2 = a % BOARD_ROWS;      a //= BOARD_ROWS
    c1 = a % BOARD_COLS;       a //= BOARD_COLS
    r1 = a % BOARD_ROWS
    return r1, c1, r2, c2


def coord_to_action(r1: int, c1: int, r2: int, c2: int) -> int:
    """Chuyển (r1, c1, r2, c2) -> action index."""
    return (
        r1 * (BOARD_COLS * BOARD_ROWS * BOARD_COLS)
        + c1 * (BOARD_ROWS * BOARD_COLS)
        + r2 * BOARD_COLS
        + c2
    )


def opposite_color(color: int) -> int:
    return -color
