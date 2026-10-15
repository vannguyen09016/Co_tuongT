"""
AlphaZero-Xiangqi: Hệ thống AI Cờ Tướng (Xiangqi) theo kiến trúc AlphaZero.

Cấu trúc:
    alphazero/
        Coach.py          - Vòng lặp self-play + train + arena
        MCTS.py           - Monte Carlo Tree Search với Dirichlet noise
        Arena.py          - Thi đấu giữa 2 player
        GenericPlayers.py - Random + Human player
    xiangqi/
        XiangqiGame.py    - Môi trường Cờ Tướng (luật, nước đi, kết thúc)
        XiangqiPlayers.py - Người chơi ngẫu nhiên & người
        XiangqiLogic.py   - Move generator + kiểm tra luật
        XiangqiConstants.py - Bảng mã quân, kích thước bàn cờ
        XiangqiDisplay.py - In bàn cờ ra terminal (tuỳ chọn)
        pytorch/
            XiangqiNNet.py  - Kiến trúc ResNet (Policy + Value head)
            NNet.py         - Wrapper huấn luyện (loss, predict, save/load)
    Coach.py             - Điểm vào huấn luyện chính
    pit.py               - CLI thi đấu: human vs AI
    utils.py             - Tiện ích dotdict, Elo...
"""