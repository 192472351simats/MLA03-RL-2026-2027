import chess
import random

class ChessMDP:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self.board.fen()

    def get_state(self):
        return self.board.fen()

    def get_actions(self):
        return list(self.board.legal_moves)

    def step(self, action):
        self.board.push(action)

        reward = 0
        done = False

        if self.board.is_checkmate():
            reward = 100
            done = True

        elif self.board.is_stalemate():
            reward = 0
            done = True

        elif self.board.is_insufficient_material():
            reward = 0
            done = True

        else:
            reward = -1

        return self.board.fen(), reward, done


def random_agent(env):

    state = env.reset()

    total_reward = 0

    move_count = 0

    print("Initial Board")
    print(env.board)

    while True:

        actions = env.get_actions()

        if len(actions) == 0:
            break

        action = random.choice(actions)

        print("\nMove", move_count + 1)
        print("Selected Move:", action)

        state, reward, done = env.step(action)

        print(env.board)

        total_reward += reward

        move_count += 1

        if done:
            break

    print("\nEpisode Finished")
    print("Total Moves:", move_count)
    print("Total Reward:", total_reward)


if __name__ == "__main__":

    env = ChessMDP()

    random_agent(env)