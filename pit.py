"""
pit.py
======
CLI thi đấu: human vs AI (AlphaZero) hoặc AI vs AI.

Ví dụ:
    # Human vs AlphaZero (load checkpoint):
    python pit.py --checkpoint checkpoints/best.pth.tar --human-red

    # 2 model AI đấu nhau:
    python pit.py --checkpoint checkpoints/best.pth.tar --num-games 4

    # AI vs random:
    python pit.py --vs random --num-games 4
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import dotdict
from xiangqi.XiangqiGame import XiangqiGame
from xiangqi.pytorch.NNet import NNetWrapper
from alphazero.GenericPlayers import NNPlayer, RandomPlayer, HumanPlayer
from alphazero.Arena import Arena


def make_nn_player(game, args):
    nnet = NNetWrapper(game, args)
    try:
        nnet.load_checkpoint(folder=args.checkpoint)
        print(f"Loaded {args.checkpoint}")
    except FileNotFoundError:
        print(f"No checkpoint at {args.checkpoint} - using random init.")
    return NNPlayer(game, nnet, args)


def play_human_vs_ai(game, args):
    p1 = HumanPlayer(game) if args.human_red else make_nn_player(game, args)
    p2 = make_nn_player(game, args) if args.human_red else HumanPlayer(game)
    arena = Arena(p1, p2, game, display=lambda b: print(b))
    print(f"Human = {'RED' if args.human_red else 'BLACK'}; AI plays opposite.")
    while True:
        r = arena.playGame(verbose=False)
        if r == 1:
            print("RED wins!")
        elif r == -1:
            print("BLACK wins!")
        else:
            print("Draw.")
        again = input("Play again? (y/n): ").strip().lower()
        if again != "y":
            break


def play_ai_vs_ai(game, args):
    p1 = make_nn_player(game, args)
    p2 = make_nn_player(game, args) if args.checkpoint2 is None else (
        NNetPlayerFromPath(game, args.checkpoint2, args)
    )
    arena = Arena(p1, p2, game, display=lambda b: None)
    w1, w2, d = arena.playGames(args.num_games, verbose=False)
    print(f"AI1 wins: {w1}, AI2 wins: {w2}, draws: {d}")


def NNetPlayerFromPath(game, path, args):
    args2 = dotdict(dict(args))
    args2.checkpoint = path
    return make_nn_player(game, args2)


def play_ai_vs_random(game, args):
    p_ai = make_nn_player(game, args)
    p_rand = RandomPlayer(game)
    # AI luôn là player 1 để benchmark công bằng.
    arena = Arena(p_ai, p_rand, game, display=lambda b: None)
    w_ai, w_rand, d = arena.playGames(args.num_games, verbose=False)
    print(f"AI wins: {w_ai}, Random wins: {w_rand}, draws: {d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/")
    parser.add_argument("--checkpoint2", type=str, default=None,
                        help="Optional 2nd checkpoint for AI vs AI")
    parser.add_argument("--num-channels", type=int, default=256)
    parser.add_argument("--num-res-blocks", type=int, default=5)
    parser.add_argument("--num-mcts-sims", type=int, default=50)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--num-games", type=int, default=4)
    parser.add_argument("--vs", choices=["random", "human", "ai"], default="ai")
    parser.add_argument("--human-red", action="store_true")
    args = parser.parse_args()
    args = dotdict(vars(args))

    game = XiangqiGame()

    if args.vs == "human":
        play_human_vs_ai(game, args)
    elif args.vs == "random":
        play_ai_vs_random(game, args)
    else:
        play_ai_vs_ai(game, args)


if __name__ == "__main__":
    main()