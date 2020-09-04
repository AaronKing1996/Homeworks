#!/usr/bin/env python3
# UTF-8
import argparse
from agent import DQNAgent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, help="Enable cuda")
    parser.add_argument("--gamma", default=0.9, help="Weight of next_state")
    parser.add_argument("--lr", default=1e-4, help="Learning rate")
    parser.add_argument("--buffer_size", default=2000, help="Replay buffer size")
    parser.add_argument("--mini_batch", default=512, help="Sampling batch")
    parser.add_argument("--epsilon", default=1.0, help="Epsilon greedy")
    parser.add_argument("--epsilon_min", default=0.001, help="Minimum of epsilon")
    parser.add_argument("--epsilon_decay", default=0.999, help="Epsilon decay")
    parser.add_argument("--reward_boundary", default=250, help="Stop boundary")
    parser.add_argument("--training", default=False, help="Enable training")
    args = parser.parse_args()

    agent = DQNAgent('CartPole-v1', args)
    if args.training:
        agent.training()
    agent.test("training-best.dat")
    pass
