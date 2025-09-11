from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import random
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Simple Tic Tac Toe
class TicTacToe:
    def __init__(self):
        self.board = [0] * 9
        self.done = False
    
    def reset(self):
        self.board = [0] * 9
        self.done = False
        return self.board
    
    def check_winner(self):
        lines = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for line in lines:
            if self.board[line[0]] == self.board[line[1]] == self.board[line[2]] != 0:
                return self.board[line[0]]
        return 0 if 0 not in self.board else None
    
    def step(self, action, player):
        if self.board[action] != 0 or self.done:
            return self.board, -10, True
        self.board[action] = player
        winner = self.check_winner()
        if winner is not None:
            self.done = True
            return self.board, winner, True
        return self.board, 0, False
    
    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

# Simple Q-Learning Agent
class QAgent:
    def __init__(self):
        self.Q = defaultdict(float)
        self.alpha = 0.1
        self.gamma = 0.95
        self.eps = 0.1
        self.trained = False
    
    def choose_action(self, state, available):
        if not self.trained or random.random() < self.eps:
            return random.choice(available)
        qvals = [self.Q.get((state, a), 0) for a in available]
        maxq = max(qvals) if qvals else 0
        best = [a for a in available if self.Q.get((state, a), 0) == maxq]
        return random.choice(best) if best else random.choice(available)
    
    def train(self, episodes=1000):
        env = TicTacToe()
        for episode in range(episodes):
            state = tuple(env.reset())
            done = False
            while not done:
                available = env.available_actions()
                if not available:
                    break
                action = self.choose_action(state, available)
                next_state, reward, done = env.step(action, 1)
                if done:
                    self.Q[(state, action)] += self.alpha * (reward - self.Q[(state, action)])
                    break
                opp_action = random.choice(env.available_actions())
                state_after_opp, opp_reward, done = env.step(opp_action, -1)
                if done:
                    self.Q[(state, action)] += self.alpha * (-1 - self.Q[(state, action)])
                    break
                best_next = max([self.Q.get((tuple(state_after_opp), a), 0) for a in env.available_actions()] or [0])
                self.Q[(state, action)] += self.alpha * (reward + self.gamma * best_next - self.Q[(state, action)])
                state = tuple(state_after_opp)
        self.trained = True
        self.eps = 0.01

# Global state
game = TicTacToe()
agent = QAgent()
agent.train(1000)  # Pre-train

@app.route('/')
def index():
    return jsonify({
        'message': 'Q-Learning Tic Tac Toe API',
        'status': 'running',
        'board': game.board,
        'agent_trained': agent.trained
    })

@app.route('/reset', methods=['POST'])
def reset():
    game.reset()
    return jsonify({'board': game.board, 'game_over': False})

@app.route('/move', methods=['POST'])
def move():
    data = request.json
    position = data.get('position', 0)
    
    # Human move
    board, reward, done = game.step(position, -1)
    result = {'board': board, 'game_over': done, 'winner': None}
    
    if done:
        result['winner'] = 'human' if reward == -1 else 'draw' if reward == 0 else 'agent'
        return jsonify(result)
    
    # Agent move
    available = game.available_actions()
    if available:
        action = agent.choose_action(tuple(board), available)
        board, reward, done = game.step(action, 1)
        result.update({'board': board, 'game_over': done, 'agent_move': action})
        if done:
            result['winner'] = 'agent' if reward == 1 else 'draw' if reward == 0 else 'human'
    
    return jsonify(result)

@app.route('/training_status', methods=['GET'])
def training_status():
    return jsonify({
        'trained': agent.trained,
        'q_table_size': len(agent.Q),
        'eps': agent.eps
    })

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    episodes = min(data.get('episodes', 1000), 5000)
    agent.train(episodes)
    return jsonify({'status': 'completed', 'episodes': episodes})

# Vercel handler
def handler(request):
    return app(request.environ, lambda *args: None)

if __name__ == '__main__':
    app.run(debug=True)