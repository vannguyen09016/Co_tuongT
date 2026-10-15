"""
test_xiangqi_game.py
====================
Integration test cho XiangqiGame class + state representation.
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from xiangqi.XiangqiGame import (
    XiangqiGame, board_to_tensor, NUM_STATE_PLANES,
    flip_board_vertically, _deep_tuple,
)
from xiangqi.XiangqiConstants import (
    BOARD_ROWS, BOARD_COLS, ACTION_SIZE, INITIAL_BOARD,
    RED, BLACK, EMPTY, piece_color, piece_type_val,
    action_to_coord, coord_to_action,
)
from xiangqi.XiangqiLogic import get_legal_moves, apply_move, find_general


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(f"FAIL: {msg}")
    print(f"PASS: {msg}")


def test_game_init():
    g = XiangqiGame()
    assert_true(g.getBoardSize() == (BOARD_ROWS, BOARD_COLS), "Board size 10x9")
    assert_true(g.getActionSize() == ACTION_SIZE, f"Action size {ACTION_SIZE}")
    assert_true(g.getInitBoard() is not None, "Init board not None")
    # Should be tuple-of-tuples (immutable)
    init = g.getInitBoard()
    assert_true(isinstance(init, tuple) and isinstance(init[0], tuple),
                "Init board is tuple-of-tuples")


def test_get_valid_moves():
    g = XiangqiGame()
    b = g.getInitBoard()
    valids = g.getValidMoves(b, RED)
    assert_true(len(valids) == ACTION_SIZE, "getValidMoves returns full-length mask")
    assert_true(sum(valids) >= 44, f"Initial RED legal moves >= 44 (got {sum(valids)})")
    # Each valid should be 0 or 1
    assert_true(all(v in (0, 1) for v in valids), "Valid mask is 0/1")


def test_get_next_state_immutability():
    """getNextState không mutate board cũ (quan trọng cho PyTorch)."""
    g = XiangqiGame()
    b = g.getInitBoard()
    b_list_before = [list(row) for row in b]
    valids = g.getValidMoves(b, RED)
    action = valids.index(1)
    b2, next_player = g.getNextState(b, RED, action)
    # Kiểm tra board cũ không thay đổi
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            assert_true(b[r][c] == b_list_before[r][c],
                        f"Original board immutable at ({r},{c})")
    assert_true(next_player == BLACK, "Next player is BLACK")


def test_get_canonical_form():
    g = XiangqiGame()
    b = g.getInitBoard()
    cb1, cp1 = g.getCanonicalForm(b, RED)
    assert_true(cp1 == 1, "Canonical player for RED is 1")
    cb2, cp2 = g.getCanonicalForm(b, BLACK)
    assert_true(cp2 == 1, "Canonical player for BLACK is also 1 (after flip)")
    # After flip, board should be flipped vertically
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            expected = -b[BOARD_ROWS - 1 - r][c]
            assert_true(cb2[r][c] == expected,
                        f"Flipped board at ({r},{c})")


def test_get_symmetries():
    g = XiangqiGame()
    b = g.getInitBoard()
    pi = np.ones(ACTION_SIZE) / ACTION_SIZE
    syms = g.getSymmetries(b, pi)
    # Xiangqi 1 symmetry (lật trái phải)
    assert_true(len(syms) == 1, "Xiangqi: 1 symmetry (fliplr)")
    sym_board, sym_pi = syms[0]
    # Sym board should be fliplr
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            assert_true(sym_board[r][c] == b[r][BOARD_COLS - 1 - c],
                        f"Symmetry flip ({r},{c})")


def test_state_tensor_shape():
    g = XiangqiGame()
    b = g.getInitBoard()
    t = g.getStateTensor(b, RED)
    assert_true(t.shape == (1, NUM_STATE_PLANES, BOARD_ROWS, BOARD_COLS),
               f"State tensor shape (1,{NUM_STATE_PLANES},{BOARD_ROWS},{BOARD_COLS})")
    # Should be float32 for PyTorch
    assert_true(t.dtype == torch.float32, "State tensor is float32")


def test_state_tensor_channels():
    """Mỗi channel biểu diễn 1 loại quân / phe."""
    g = XiangqiGame()
    b = g.getInitBoard()
    cb, _ = g.getCanonicalForm(b, RED)
    arr = board_to_tensor(cb, RED)
    # Channel 0 = RED general; có 1 quân ở (0,4) sau chín
    # First sample of RED general position (canonical form is RED view)
    assert_true(arr[0].sum() == 1.0, "1 RED GENERAL in channel 0")
    # Channel 14 = turn plane
    assert_true(arr[NUM_STATE_PLANES - 1].sum() == BOARD_ROWS * BOARD_COLS,
                "Turn plane is fully 1 for current player (RED)")
    # Channel 7 = BLACK general (top of flipped board)
    # After canonical flip for BLACK, BLACK general is at (0, 4)
    cb_black, _ = g.getCanonicalForm(b, BLACK)
    arr_black = board_to_tensor(cb_black, RED)
    assert_true(arr_black[7].sum() == 1.0, "1 BLACK GENERAL in channel 7 (after canonical)")
    # Total pieces on board
    assert_true(arr.sum() == 32 + BOARD_ROWS * BOARD_COLS,
                f"32 pieces + turn plane sum")


def test_get_score():
    g = XiangqiGame()
    b = g.getInitBoard()
    score_red = g.getScore(b, RED)
    score_black = g.getScore(b, BLACK)
    # Vì board khởi đầu cân bằng -> score ~ 0
    assert_true(abs(score_red - score_black) < 1e-6,
                f"Initial board balanced: {score_red} vs {score_black}")


def test_get_game_ended_stalemate():
    """Bên không có nước đi = thua (theo luật cờ tướng)."""
    g = XiangqiGame()
    # Tạo thế stalemate cho RED
    from xiangqi.XiangqiLogic import (
        R_GENERAL, R_ADVISOR, R_ELEPHANT, R_HORSE,
        R_CHARIOT, R_CANNON, R_SOLDIER,
        B_GENERAL, B_ADVISOR, B_ELEPHANT, B_HORSE,
        B_CHARIOT, B_CANNON, B_SOLDIER,
    )
    b = [[EMPTY for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
    b[0][4] = R_GENERAL
    b[0][1] = B_CHARIOT
    b[1][4] = B_CHARIOT
    b[9][4] = B_GENERAL
    b_tuple = _deep_tuple(b)
    # Stalemate (RED has no move) -> RED loses
    r = g.getGameEnded(b_tuple, RED)
    assert_true(r == -1, f"Stalemate: RED loses (returned {r})")


def test_action_consistency():
    """action_to_coord / coord_to_action round-trip."""
    g = XiangqiGame()
    b = g.getInitBoard()
    valids = g.getValidMoves(b, RED)
    valid_indices = [i for i, v in enumerate(valids) if v == 1]
    for a in valid_indices:
        r1, c1, r2, c2 = action_to_coord(a)
        a_back = coord_to_action(r1, c1, r2, c2)
        assert_true(a == a_back, f"Action {a} round-trip ok")


def test_get_next_state_count():
    """Số nước đi hợp lệ bằng số action được đánh dấu 1 trong getValidMoves."""
    g = XiangqiGame()
    b = g.getInitBoard()
    valids = g.getValidMoves(b, RED)
    expected_count = sum(valids)
    actual = get_legal_moves([list(row) for row in b], RED)
    assert_true(len(actual) == expected_count,
                f"Legal moves count: getValidMoves={expected_count}, get_legal_moves={len(actual)}")


def test_state_tensor_invariant_to_perspective():
    """Cùng một bàn cờ, state tensor phải có cùng shape & tổng pieces."""
    g = XiangqiGame()
    b = g.getInitBoard()
    t_red = g.getStateTensor(b, RED)
    t_black = g.getStateTensor(b, BLACK)
    # Same number of pieces, same turn plane value
    assert_true(t_red.shape == t_black.shape, "Tensors have same shape")
    assert_true(t_red[0, -1].sum() == BOARD_ROWS * BOARD_COLS,
                "Turn plane = 1 (RED view)")
    assert_true(t_black[0, -1].sum() == BOARD_ROWS * BOARD_COLS,
                "Turn plane = 1 (BLACK canonical view)")
    # Total pieces sum should match (16 + 16 = 32 pieces + 90 turn = 122)
    assert_true(t_red.sum() == t_black.sum(),
                "Same total piece count in both views")


def test_display():
    g = XiangqiGame()
    b = g.getInitBoard()
    # Just ensure no exception
    g.display(b)


if __name__ == "__main__":
    test_game_init()
    test_get_valid_moves()
    test_get_next_state_immutability()
    test_get_canonical_form()
    test_get_symmetries()
    test_state_tensor_shape()
    test_state_tensor_channels()
    test_get_score()
    test_get_game_ended_stalemate()
    test_action_consistency()
    test_get_next_state_count()
    test_state_tensor_invariant_to_perspective()
    test_display()
    print("\nAll XiangqiGame tests PASSED")