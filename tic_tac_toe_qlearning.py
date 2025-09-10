import numpy as np
import random
from collections import defaultdict

# -------------------------------
# Tic Tac Toe Environment
# -------------------------------
class TicTacToe:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros(9, dtype=int)  # 0 empty, 1 agent (X), -1 human (O)
        self.done = False
        return tuple(self.board)
    
    def check_winner(self, board):
        combos = [(0,1,2),(3,4,5),(6,7,8),
                  (0,3,6),(1,4,7),(2,5,8),
                  (0,4,8),(2,4,6)]
        for (i,j,k) in combos:
            s = board[i] + board[j] + board[k]
            if s == 3: return 1   # agent wins
            if s == -3: return -1 # opponent wins
        if not 0 in board:
            return 0   # draw
        return None     # ongoing
    
    def step(self, action, player=1):
        if self.board[action] != 0 or self.done:
            return tuple(self.board), -10, True  # illegal move penalty
        self.board[action] = player
        winner = self.check_winner(self.board)
        if winner is not None:
            self.done = True
            return tuple(self.board), winner, True
        return tuple(self.board), 0, False
    
    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def render(self):
        symbols = {1: "X", -1: "O", 0: " "}
        rows = []
        for i in range(0, 9, 3):
            row = " | ".join(symbols[self.board[j]] for j in range(i, i+3))
            rows.append(row)
        print("\n---------\n".join(rows))
        print()

# -------------------------------
# Q-learning Agent
# -------------------------------
Q = defaultdict(float)
alpha, gamma, eps = 0.1, 0.95, 0.1

def choose_action(state, available):
    if random.random() < eps:
        return random.choice(available)
    qvals = [Q[(state, a)] for a in available]
    maxq = max(qvals) if qvals else 0
    best = [a for a in available if Q[(state, a)] == maxq]
    return random.choice(best) if best else random.choice(available)

env = TicTacToe()

# -------------------------------
# Training loop
# -------------------------------
for episode in range(50000):
    state = env.reset()
    done = False
    while not done:
        # Agent move
        available = env.available_actions()
        action = choose_action(state, available)
        next_state, reward, done = env.step(action, player=1)
        
        if done:
            Q[(state, action)] += alpha * (reward - Q[(state, action)])
            break
        
        # Opponent (random)
        opp_action = random.choice(env.available_actions())
        state_after_opp, opp_reward, done = env.step(opp_action, player=-1)
        
        if done:
            Q[(state, action)] += alpha * (-1 - Q[(state, action)])  # agent lost
            break
        
        best_next = max([Q[(state_after_opp,a)] for a in env.available_actions()] or [0])
        Q[(state, action)] += alpha * (reward + gamma*best_next - Q[(state, action)])
        
        state = state_after_opp

# -------------------------------
# Play against the trained agent
# -------------------------------
def play_against_agent():
    state = env.reset()
    env.render()
    done = False
    while not done:
        # Human move
        human_action = int(input("Your move (0-8): "))
        state, reward, done = env.step(human_action, player=-1)
        env.render()
        if done:
            if reward == -1:
                print("You win!")
            elif reward == 0:
                print("It's a draw.")
            else:
                print("Agent wins.")
            break
        
        # Agent move
        available = env.available_actions()
        action = choose_action(state, available)
        state, reward, done = env.step(action, player=1)
        print(f"Agent chooses {action}")
        env.render()
        if done:
            if reward == 1:
                print("Agent wins.")
            elif reward == 0:
                print("It's a draw.")
            else:
                print("You win!")
            break

# -------------------------------
# Run game
# -------------------------------
if __name__ == "__main__":
    print("Play Tic Tac Toe against the Q-learning agent!")
    print("Board positions are numbered 0-8 like this:")
    print("0 | 1 | 2\n---------\n3 | 4 | 5\n---------\n6 | 7 | 8\n")
    play_against_agent()
