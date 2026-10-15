"""
XiangqiDisplay.py
=================
In bàn cậu ra console dưới dạng dễ nhìn.
"""
from __future__ import annotations
import sys
from xiangqi.XiangqiConstants import (
    BOARD_ROWS, BOARD_COLS, EMPTY, RED, BLACK,
    piece_color, piece_type_val,
)


_GLYPHS_UNICODE = {
    EMPTY: " . ",
    (RED, 1):   "帥",   # General
    (RED, 2):   "仕",   # Advisor
    (RED, 3):   "相",   # Elephant
    (RED, 4):   "傌",   # Horse
    (RED, 5):   "俥",   # Chariot
    (RED, 6):   "炮",   # Cannon
    (RED, 7):   "兵",   # Soldier
    (BLACK, 1): "將",
    (BLACK, 2): "士",
    (BLACK, 3): "象",
    (BLACK, 4): "馬",
    (BLACK, 5): "車",
    (BLACK, 6): "砲",
    (BLACK, 7): "卒",
}

_GLYPHS_ASCII = {
    EMPTY: " . ",
    (RED, 1):   " R ",   (BLACK, 1): " r ",
    (RED, 2):   " A ",   (BLACK, 2): " a ",
    (RED, 3):   " E ",   (BLACK, 3): " e ",
    (RED, 4):   " H ",   (BLACK, 4): " h ",
    (RED, 5):   " C ",   (BLACK, 5): " c ",
    (RED, 6):   " N ",   (BLACK, 6): " n ",
    (RED, 7):   " S ",   (BLACK, 7): " s ",
}


def _try_utf8() -> bool:
    """Kiểm tra stdout có hỗ trợ UTF-8 không."""
    try:
        sys.stdout.write("帥")
        sys.stdout.flush()
        return True
    except (UnicodeEncodeError, ValueError):
        return False


_USE_UNICODE = True


def display_board(board, use_unicode: bool = None) -> str:
    """Trả về chuỗi biểu diễn bàn cờ."""
    if use_unicode is None:
        use_unicode = _USE_UNICODE
    glyphs = _GLYPHS_UNICODE if use_unicode else _GLYPHS_ASCII
    lines = []
    # Header cột
    lines.append("   " + " ".join("a b c d e f g h i".split()))
    for r in range(BOARD_ROWS):
        row_str = [f"{r} "]
        for c in range(BOARD_COLS):
            p = board[r][c]
            if p == EMPTY:
                row_str.append(glyphs[EMPTY])
            else:
                color = piece_color(p)
                ptype = piece_type_val(p)
                row_str.append(glyphs[(color, ptype)])
        lines.append(" ".join(row_str))
    lines.append("")
    return "\n".join(lines)


def print_board(board):
    """In bàn cờ ra console."""
    sys.stdout.reconfigure(encoding="utf-8")
    print(display_board(board))


if __name__ == "__main__":
    from xiangqi.XiangqiConstants import INITIAL_BOARD
    sys.stdout.reconfigure(encoding="utf-8")
    print_board(INITIAL_BOARD)