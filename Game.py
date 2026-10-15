"""
Game.py
=======
Định nghĩa giao diện trừu tượng `Game` cho framework AlphaZero.

Mọi game cụ thể (Xiangqi, Othello,...) phải kế thừa lớp này và override
toàn bộ phương thức trừu tượng.

Quy ước chung:
    - Board là một tuple/list 2-D. Phần tử board[r][c] đại diện cho ô (row r, col c).
    - Player = 1 (RED / đỏ) hoặc -1 (BLACK / đen).
    - Một action là một số nguyên từ 0 đến action_size - 1. Game cụ thể sẽ
      encode/giải mã từ (r1, c1, r2, c2) sang action index.
    - getGameEnded trả về:
        + 1  nếu player HIỆN TẠI (next player of board) thắng
        + -1 nếu player HIỆN TẠI thua
        + một số rất nhỏ (e.g. 1e-4) nếu hoà
        + 0  nếu ván chưa kết thúc
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class Game(ABC):
    """Lớp trừu tượng định nghĩa giao diện cho mọi game trong AlphaZero."""

    @abstractmethod
    def getInitBoard(self) -> Tuple[int, ...]:
        """Trả về trạng thái bàn cờ khởi đầu.
        Bàn cờ là tuple-of-tuples để tránh PyTorch cảnh báo việc mutate
        tensor khi sử dụng cho self-play trên nhiều process.
        """
        ...

    @abstractmethod
    def getBoardSize(self) -> Tuple[int, int]:
        """Trả về (rows, cols) của bàn cờ. Với Xiangqi: (10, 9)."""
        ...

    @abstractmethod
    def getActionSize(self) -> int:
        """Tổng số action có thể. Với Xiangqi: 9 * 10 * 9 * 10 = 8100."""
        ...

    @abstractmethod
    def getNextState(self, board: Any, player: int, action: int) -> Tuple[Any, int]:
        """Áp dụng action lên board, trả về (board_mới, player_mới).
        Lưu ý: KHÔNG được thay đổi board gốc (deep-copy hoặc trả tuple mới).
        """
        ...

    @abstractmethod
    def getValidMoves(self, board: Any, player: int) -> List[int]:
        """Trả về list các action index hợp lệ trong trạng thái hiện tại.
        Độ dài = getActionSize(). Action hợp lệ -> 1, ngược lại -> 0.
        """
        ...

    @abstractmethod
    def getGameEnded(self, board: Any, player: int) -> int:
        """Trả về trạng thái kết thúc game cho player HIỆN TẠI (next player).
            1  : player thắng (đối phương vừa chiếu bí -> player không thể đi nữa)
                (trong AlphaZero, +1 luôn là người ĐÃ ĐI nước cuối cùng)
            -1 : player thua
            1e-4 (gần 0) : hoà
            0  : chưa kết thúc
        Tuy nhiên, theo cách dùng phổ biến trong alpha-zero-general, hàm này
        được gọi với player = "player vừa đi nước gần nhất" và trả về:
            1  nếu player đó thắng
            -1 nếu player đó thua
            1e-4 nếu hoà
        Trong Xiangqi, ta sẽ theo cách thứ hai.
        """
        ...

    @abstractmethod
    def getCanonicalForm(self, board: Any, player: int) -> Tuple[Any, Any]:
        """Trả về (canonical_board, canonical_player).
        Canonical form: đổi "góc nhìn" sao cho player.RED luôn là +1.
        Thông thường trả về (board, player) nhưng với quân đen đã được lật.
        Trong Xiangqi, ta cần đảo chiều bàn cờ và lật dấu quân đen thành đỏ.
        """
        ...

    @abstractmethod
    def getSymmetries(self, board: Any, pi: Any) -> List[Tuple[Any, Any]]:
        """Trả về danh sách (board, pi) sau khi áp dụng các phép đối xứng.
        Dùng cho data augmentation trong huấn luyện.
        Với Xiangqi, số phép đối xứng tối đa là 8 (xoay + lật), nhưng do
        tính bất đối xứng của quân đỏ/đen và cung điện, ta thường chỉ giữ
        phép đối xứng theo trục dọc (lật trái-phải) để khớp với góc nhìn.
        """
        ...

    @abstractmethod
    def getScore(self, board: Any, player: int) -> float:
        """Heuristic dùng cho arena/evaluation (tuỳ chọn)."""
        ...

    @abstractmethod
    def getEndingTurn(self) -> int:
        """Số turn tối đa cho mỗi ván (tránh trận đấu kéo dài vô tận)."""
        ...

    # ----- Helpers không trừu tượng -----
    def stringRepresentation(self, board: Any) -> str:
        """Biểu diễn board dạng string dùng cho NeuralNetwork cache.
        Mặc định: dùng repr của canonical form.
        """
        return repr(board)

    def display(self, board: Any) -> None:
        """In bàn cờ ra console (mặc định no-op). Game cụ thể sẽ override."""
        print(board)