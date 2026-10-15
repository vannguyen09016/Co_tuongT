"""
test_xiangqi_logic.py
=====================
Kiểm thử luật Cờ Tướng:
  - Tướng không rời cung.
  - Sĩ không rời cung.
  - Tượng không qua sông + cản mắt.
  - Mã cản chân.
  - Pháo nhảy qua 1 quân để ăn.
  - Tốt qua sông mới được đi ngang; không lùi.
  - Chống Tướng (hai Tướng đối diện cùng cột).
  - Chiếu bí.
  - Không tự chiếu Tướng (pin).
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

from xiangqi.XiangqiLogic import (
    get_legal_moves, is_legal_move, is_in_check, is_facing_generals,
    apply_move, find_general,
)
from xiangqi.XiangqiConstants import (
    INITIAL_BOARD,
    RED, BLACK, EMPTY,
    R_GENERAL, R_ADVISOR, R_ELEPHANT, R_HORSE,
    R_CHARIOT, R_CANNON, R_SOLDIER,
    B_GENERAL, B_ADVISOR, B_ELEPHANT, B_HORSE,
    B_CHARIOT, B_CANNON, B_SOLDIER,
    BOARD_ROWS, BOARD_COLS,
)


def mb():
    return [[EMPTY for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(f"FAIL: {msg}")
    print(f"PASS: {msg}")


def test_initial():
    b = [row[:] for row in INITIAL_BOARD]
    legal = get_legal_moves(b, RED)
    assert_true(len(legal) >= 44, f"Initial RED legal moves >=44 (got {len(legal)})")


def test_general_stays_in_palace():
    b = mb()
    b[0][4] = R_GENERAL
    b[9][4] = B_GENERAL
    b[5][4] = R_ADVISOR  # block facing generals
    legal = get_legal_moves(b, RED)
    for r1, c1, r2, c2 in legal:
        assert_true(r2 <= 2, f"RED general can move to row {r2} (must be <=2)")


def test_elephant_cannot_cross_river():
    b = mb()
    b[0][4] = R_GENERAL
    b[9][4] = B_GENERAL
    b[5][4] = R_ADVISOR  # block facing generals
    b[0][2] = R_ELEPHANT
    legal = get_legal_moves(b, RED)
    for r1, c1, r2, c2 in legal:
        assert_true(r2 <= 4, f"ELEPHANT crosses river to row {r2}")
    b[4][2] = R_ELEPHANT
    legal2 = get_legal_moves(b, RED)
    for r1, c1, r2, c2 in legal2:
        assert_true(r2 <= 4, f"ELEPHANT at row4 crosses river to row {r2}")


def test_elephant_eye_blocked():
    """Tượng ở (2,4). Các đích:
        (0,2) eye=(1,3), (0,6) eye=(1,5), (4,2) eye=(3,3), (4,6) eye=(3,5).
    Chặn (1,3) -> (0,2) bị cấm.
    Chặn (3,5) -> (4,6) bị cấm.
    Luôn đặt Tướng đen ở (9,4) và THÊM QUÂN CHẮN ở column 4 để hai Tướng không đối mặt.
    """
    b = mb()
    b[0][4] = R_GENERAL
    b[9][4] = B_GENERAL
    b[5][4] = R_ADVISOR   # chắn chống Tướng (không ảnh hưởng tượng ở (2,4))
    b[2][4] = R_ELEPHANT
    b[1][3] = B_SOLDIER   # chặn mắt (1,3) -> (0,2) cấm
    b[3][5] = B_SOLDIER   # chặn mắt (3,5) -> (4,6) cấm
    legal = get_legal_moves(b, RED)
    targets = {(r2, c2) for r1, c1, r2, c2 in legal}
    assert_true((0, 2) not in targets, "Elephant blocked by eye (1,3): (2,4)->(0,2)")
    assert_true((4, 6) not in targets, "Elephant blocked by eye (3,5): (2,4)->(4,6)")
    assert_true((0, 6) in targets,  "Elephant free eye (1,5): (2,4)->(0,6)")
    assert_true((4, 2) in targets,  "Elephant free eye (3,3): (2,4)->(4,2)")


def test_horse_leg_blocked():
    b = mb()
    b[0][4] = R_GENERAL
    b[9][4] = B_GENERAL
    b[5][4] = R_ADVISOR  # block facing
    b[4][4] = R_HORSE
    b[3][4] = B_SOLDIER  # chân mã phía trên -> cản 4 nước đi lên
    legal = get_legal_moves(b, RED)
    targets = {(r2, c2) for r1, c1, r2, c2 in legal}
    for t in [(2, 3), (2, 5), (6, 3), (6, 5)]:
        assert_true(t not in targets, f"Horse leg blocks move {t}")


def test_cannon_capture():
    """Pháo ở (0,7). Bàn đạp ở (4,7), mục tiêu ở (7,7).
    Pháo có thể ăn (7,7) bằng cách nhảy qua (4,7).
    Sau đó nếu có thêm quân ở (8,7) thì không thể ăn thêm do chưa nhảy qua đủ 1 quân.
    """
    b = mb()
    b[0][4] = R_GENERAL
    b[9][4] = B_GENERAL
    b[5][4] = R_ADVISOR  # block facing
    b[0][7] = R_CANNON
    b[4][7] = R_SOLDIER   # bàn đạp
    b[7][7] = B_SOLDIER   # mục tiêu
    legal = get_legal_moves(b, RED)
    targets = {(r2, c2) for r1, c1, r2, c2 in legal}
    assert_true((7, 7) in targets, "Cannon can capture over 1 screen")
    # Đặt thêm quân ở (8,7) - cannon không thể nhảy qua 2 quân
    b[8][7] = B_SOLDIER
    legal2 = get_legal_moves(b, RED)
    targets2 = {(r2, c2) for r1, c1, r2, c2 in legal2}
    # Để ý: vẫn có (5,8),...,vẫn ăn (7,7) được - nhưng (8,7) thì không.
    assert_true((7, 7) in targets2, "Cannon can still capture (7,7) with 2 screens nearby")
    assert_true((8, 7) not in targets2, "Cannon cannot capture (8,7) (needs to jump 2 screens)")

    # Test THỰC SỰ: hai màn chắn liên tiếp cản pháo ăn quân đằng sau.
    # Pháo ở (0,7), screen1=(4,7), screen2=(5,7) -> target (8,7) cần nhảy 2 lần -> không được.
    b2 = mb()
    b2[0][4] = R_GENERAL
    b2[9][4] = B_GENERAL
    b2[5][4] = R_ADVISOR
    b2[0][7] = R_CANNON
    b2[4][7] = B_SOLDIER  # screen 1
    b2[5][7] = B_SOLDIER  # screen 2 (ngay sau screen 1)
    b2[8][7] = B_CHARIOT  # target
    legal3 = get_legal_moves(b2, RED)
    targets3 = {(r2, c2) for r1, c1, r2, c2 in legal3}
    assert_true((8, 7) not in targets3, "Cannon cannot capture when 2 screens between")


def test_soldier_cross_river():
    b = mb()
    b[0][4] = R_GENERAL
    b[9][4] = B_GENERAL
    b[5][4] = R_ADVISOR  # block facing
    # Soldier at (4,4), forward to (5,4) blocked by R_ADVISOR.
    # Place soldier at different spot:
    b[5][4] = EMPTY  # remove advisor (it's blocker, but also blocks soldier)
    b[4][4] = R_SOLDIER
    legal = get_legal_moves(b, RED)
    targets = {(r2, c2) for r1, c1, r2, c2 in legal}
    assert_true((5, 4) in targets, "Soldier forward (4,4)->(5,4)")
    assert_true((4, 3) not in targets and (4, 5) not in targets,
                "Soldier cannot move horizontally before crossing")
    assert_true((3, 4) not in targets, "Soldier cannot move backwards")

    b2 = mb()
    b2[0][4] = R_GENERAL
    b2[9][4] = B_GENERAL
    # Place a blocker ON column 4 between the generals:
    b2[5][0] = R_SOLDIER  # already across river at edge (col 0, not col 4)
    b2[6][4] = R_SOLDIER  # blocker ON column 4
    legal2 = get_legal_moves(b2, RED)
    targets2 = {(r2, c2) for r1, c1, r2, c2 in legal2}
    assert_true((6, 0) in targets2, "Soldier forward after crossing")
    assert_true((5, 1) in targets2, "Soldier can move horizontally after crossing")
    assert_true((4, 0) not in targets2, "Soldier cannot move backwards after crossing")


def test_facing_generals():
    b = mb()
    b[0][4] = R_GENERAL
    b[9][4] = B_GENERAL
    assert_true(is_facing_generals(b), "Generals face each other directly")
    b[5][4] = R_SOLDIER
    assert_true(not is_facing_generals(b), "Generals no longer face due to blocker")


def test_check_detection():
    b = mb()
    b[0][4] = R_GENERAL
    b[9][4] = B_GENERAL
    b[5][4] = R_ADVISOR  # block facing
    b[0][0] = B_CHARIOT
    assert_true(is_in_check(b, RED), "RED in check from black chariot")


def test_checkmate():
    b = mb()
    b[0][4] = R_GENERAL
    b[0][1] = B_CHARIOT
    b[1][4] = B_CHARIOT
    b[9][4] = B_GENERAL
    legal = get_legal_moves(b, RED)
    assert_true(len(legal) == 0, f"Checkmate: RED has 0 legal moves (got {len(legal)})")


def test_pin_no_self_check():
    b = mb()
    b[0][4] = R_GENERAL
    b[9][4] = B_GENERAL
    b[5][4] = R_ADVISOR  # block facing
    b[0][2] = R_HORSE   # bảo vệ tướng khỏi pháo đen
    b[0][8] = B_CANNON  # đe dọa
    legal = get_legal_moves(b, RED)
    for r1, c1, r2, c2 in legal:
        b2 = apply_move(b, r1, c1, r2, c2)
        assert_true(not is_in_check(b2, RED),
                    f"Move {(r1, c1, r2, c2)} should not leave RED in check")


def test_advisor_in_palace():
    """Sĩ đỏ ở (0,3) chỉ được ở cung đỏ (rows 0..2, cols 3..5)."""
    b = mb()
    b[0][4] = R_GENERAL
    b[9][4] = B_GENERAL
    b[5][4] = R_SOLDIER  # block facing/general
    b[0][3] = R_ADVISOR
    legal = get_legal_moves(b, RED)
    targets = {(r2, c2) for r1, c1, r2, c2 in legal}
    # Advisor (0,3) -> (1,4) và (1,2) -- nhưng (1,2) có col=2 nằm ngoài cung.
    assert_true((1, 4) in targets, "Advisor (0,3)->(1,4)")
    assert_true((1, 2) not in targets, "Advisor cannot exit palace (1,2) (col 2)")
    # Verify advisor không rời cung
    for r1, c1, r2, c2 in legal:
        if (r1, c1) == (0, 3):  # advisor moves
            assert_true(0 <= r2 <= 2 and 3 <= c2 <= 5,
                        f"Advisor out of palace: ({r1},{c1})->({r2},{c2})")

    # Sĩ ở (1,4) -> (0,3), (0,5), (2,3), (2,5) — tất cả trong cung
    b2 = mb()
    b2[0][4] = R_GENERAL
    b2[9][4] = B_GENERAL
    b2[5][4] = R_SOLDIER
    b2[1][4] = R_ADVISOR
    legal2 = get_legal_moves(b2, RED)
    t2 = {(r2, c2) for r1, c1, r2, c2 in legal2}
    assert_true((0, 3) in t2 and (0, 5) in t2 and (2, 3) in t2 and (2, 5) in t2,
                "Advisor (1,4) has 4 diagonal moves within palace")


def test_find_general():
    b = [row[:] for row in INITIAL_BOARD]
    assert_true(find_general(b, RED) == (0, 4), f"RED general at (0,4), got {find_general(b, RED)}")
    assert_true(find_general(b, BLACK) == (9, 4), f"BLACK general at (9,4), got {find_general(b, BLACK)}")


def test_apply_move():
    b = mb()
    b[0][4] = R_GENERAL
    b[9][4] = B_GENERAL
    b[5][4] = R_ADVISOR
    b[4][4] = R_SOLDIER
    b2 = apply_move(b, 4, 4, 5, 4)
    assert_true(b2[4][4] == EMPTY, "Source cleared after move")
    assert_true(b2[5][4] == R_SOLDIER, "Soldier moved to (5,4)")
    assert_true(b[4][4] == R_SOLDIER, "Original board not modified")


def test_is_legal_move():
    b = mb()
    b[0][4] = R_GENERAL
    b[9][4] = B_GENERAL
    # Move blocker away from column 4 to allow soldier move:
    b[5][4] = EMPTY
    b[4][4] = R_SOLDIER
    assert_true(is_legal_move(b, RED, 4, 4, 5, 4), "Soldier forward is legal")
    assert_true(not is_legal_move(b, RED, 4, 4, 4, 3),
                "Soldier horizontal before river is illegal")
    assert_true(not is_legal_move(b, RED, 0, 4, 3, 4),
                "General jumping 3 rows is illegal")
    assert_true(not is_legal_move(b, RED, 0, 4, 0, 0),
                "General (0,4)->(0,0) illegal (out of palace col)")
    assert_true(not is_legal_move(b, RED, 0, 4, 5, 4),
                "General (0,4)->(5,4) illegal (out of palace row)")


if __name__ == "__main__":
    test_initial()
    test_general_stays_in_palace()
    test_elephant_cannot_cross_river()
    test_elephant_eye_blocked()
    test_horse_leg_blocked()
    test_cannon_capture()
    test_soldier_cross_river()
    test_facing_generals()
    test_check_detection()
    test_checkmate()
    test_pin_no_self_check()
    test_advisor_in_palace()
    test_find_general()
    test_apply_move()
    test_is_legal_move()
    print("\nAll logic tests PASSED")
