"""
utils.py
========
Các tiện ích dùng chung cho framework AlphaZero-Xiangqi.
"""
from __future__ import annotations
import os
import math
import collections


class dotdict(dict):
    """dict cho phép truy cập qua thuộc tính: d.key thay vì d['key'].
    Dùng cho config argparse.
    """

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)


class AverageMeter:
    """Đo giá trị trung bình tích luỹ (dùng cho log loss, elo,...)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.avg = 0.0
        self.val = 0.0

    def update(self, value: float, n: int = 1):
        self.val = value
        self.sum += value * n
        self.n += n
        self.avg = self.sum / max(1, self.n)


class EloMeter:
    """Bảng xếp hạng Elo giữa các checkpoint."""

    K = 32.0

    def __init__(self, base: float = 1500.0):
        self.elo: dict[str, float] = collections.OrderedDict()
        self.base = base

    def add_player(self, name: str, rating: float = 1500.0) -> None:
        self.elo.setdefault(name, rating)

    def expected_score(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(self, winner: str, loser: str) -> None:
        self.add_player(winner)
        self.add_player(loser)
        ra, rb = self.elo[winner], self.elo[loser]
        ea = self.expected_score(ra, rb)
        eb = 1.0 - ea
        self.elo[winner] += self.K * (1.0 - ea)
        self.elo[loser] += self.K * (0.0 - eb)

    def rating(self, name: str) -> float:
        return self.elo.get(name, self.base)


def elo_diff(elo_a: float, elo_b: float) -> float:
    """Khác biệt Elo giữa 2 agent (chỉ để hiển thị)."""
    return elo_a - elo_b


def ensure_dir(path: str) -> None:
    """Tạo thư mục nếu chưa tồn tại."""
    os.makedirs(path, exist_ok=True)


def chinese_number(n: int) -> str:
    """Đếm số ván chơi theo cách Trung Quốc: 一, 二, ... 十, 一百.
    Chỉ dùng cho mục đích hiển thị log.
    """
    cn_digits = "零一二三四五六七八九"
    if n < 10:
        return cn_digits[n]
    if n == 10:
        return "十"
    if n < 20:
        return "十" + cn_digits[n - 10]
    if n < 100:
        tens, ones = divmod(n, 10)
        s = "十" if tens == 1 else cn_digits[tens] + "十"
        if ones:
            s += cn_digits[ones]
        return s
    return str(n)


if __name__ == "__main__":
    # Sanity check.
    cfg = dotdict({"a": 1, "b": {"c": 2}})
    cfg.b.c = cfg.d
    print("config a =", cfg.a, "d =", cfg.b.c)

    m = AverageMeter()
    for v in [0.1, 0.2, 0.3]:
        m.update(v)
    print("avg =", m.avg)

    e = EloMeter()
    e.update("A", "B")
    print("Elo A:", e.eloA if hasattr(e, "eloA") else e.elo["A"])