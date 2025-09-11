from flask import Flask, render_template, request, jsonify
import numpy as np
import random
from collections import defaultdict
import json
import time

app = Flask(__name__)

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

    def get_board_state(self):
        return self.board.tolist()

# -------------------------------
# Q-learning Agent
# -------------------------------
class QLearningAgent:
    def __init__(self):
        self.Q = defaultdict(float)
        self.alpha = 0.1
        self.gamma = 0.95
        self.eps = 0.1
        self.trained = False
        self.training_stats = {
            'episodes_completed': 0,
            'total_episodes': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'win_rate': 0.0,
            'avg_reward': 0.0,
            'is_training': False
        }
    
    def set_parameters(self, alpha=None, gamma=None, eps=None):
        if alpha is not None:
            self.alpha = alpha
        if gamma is not None:
            self.gamma = gamma
        if eps is not None:
            self.eps = eps
    
    def choose_action(self, state, available):
        if random.random() < self.eps and self.trained:
            return random.choice(available)
        qvals = [self.Q[(state, a)] for a in available]
        maxq = max(qvals) if qvals else 0
        best = [a for a in available if self.Q[(state, a)] == maxq]
        return random.choice(best) if best else random.choice(available)
    
    def train_batch(self, episodes=1000):
        """Train the agent for a batch of episodes (Vercel-compatible)"""
        self.training_stats['is_training'] = True
        self.training_stats['total_episodes'] += episodes
        self.training_stats['episodes_completed'] = 0
        self.training_stats['wins'] = 0
        self.training_stats['losses'] = 0
        self.training_stats['draws'] = 0
        
        env = TicTacToe()
        total_reward = 0
        
        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Agent move
                available = env.available_actions()
                action = self.choose_action(state, available)
                next_state, reward, done = env.step(action, player=1)
                episode_reward += reward
                
                if done:
                    self.Q[(state, action)] += self.alpha * (reward - self.Q[(state, action)])
                    if reward == 1:
                        self.training_stats['wins'] += 1
                    elif reward == -1:
                        self.training_stats['losses'] += 1
                    else:
                        self.training_stats['draws'] += 1
                    break
                
                # Opponent (random)
                opp_action = random.choice(env.available_actions())
                state_after_opp, opp_reward, done = env.step(opp_action, player=-1)
                episode_reward += opp_reward
                
                if done:
                    self.Q[(state, action)] += self.alpha * (-1 - self.Q[(state, action)])
                    self.training_stats['losses'] += 1
                    break
                
                best_next = max([self.Q[(state_after_opp,a)] for a in env.available_actions()] or [0])
                self.Q[(state, action)] += self.alpha * (reward + self.gamma*best_next - self.Q[(state, action)])
                
                state = state_after_opp
            
            total_reward += episode_reward
            self.training_stats['episodes_completed'] += 1
            self.training_stats['win_rate'] = self.training_stats['wins'] / self.training_stats['episodes_completed']
            self.training_stats['avg_reward'] = total_reward / self.training_stats['episodes_completed']
        
        self.trained = True
        self.eps = 0.05  # Keep some exploration for Vercel
        self.training_stats['is_training'] = False
        
        return self.training_stats.copy()
    
    def get_training_stats(self):
        return self.training_stats.copy()

# Global game state
game_env = TicTacToe()
agent = QLearningAgent()

# Train the agent on startup (quick training)
print("Training Q-learning agent...")
agent.train_batch(5000)  # Reduced for faster startup
print("Training completed!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset_game():
    global game_env
    game_env.reset()
    return jsonify({
        'board': game_env.get_board_state(),
        'game_over': False,
        'winner': None
    })

@app.route('/move', methods=['POST'])
def make_move():
    global game_env, agent
    
    data = request.json
    position = data.get('position')
    
    # Human move
    state, reward, done = game_env.step(position, player=-1)
    
    result = {
        'board': game_env.get_board_state(),
        'game_over': done,
        'winner': None,
        'message': ''
    }
    
    if done:
        if reward == -1:
            result['winner'] = 'human'
            result['message'] = 'You win!'
        elif reward == 0:
            result['winner'] = 'draw'
            result['message'] = "It's a draw!"
        else:
            result['winner'] = 'agent'
            result['message'] = 'Agent wins!'
        return jsonify(result)
    
    # Agent move
    available = game_env.available_actions()
    if available:  # Check if game is still ongoing
        action = agent.choose_action(tuple(game_env.board), available)
        state, reward, done = game_env.step(action, player=1)
        
        result['board'] = game_env.get_board_state()
        result['agent_move'] = action
        result['game_over'] = done
        
        if done:
            if reward == 1:
                result['winner'] = 'agent'
                result['message'] = 'Agent wins!'
            elif reward == 0:
                result['winner'] = 'draw'
                result['message'] = "It's a draw!"
            else:
                result['winner'] = 'human'
                result['message'] = 'You win!'
    
    return jsonify(result)

@app.route('/train', methods=['POST'])
def start_training():
    global agent
    
    data = request.json
    episodes = min(data.get('episodes', 1000), 5000)  # Limit for Vercel
    alpha = data.get('alpha', 0.1)
    gamma = data.get('gamma', 0.95)
    eps = data.get('eps', 0.1)
    
    # Set parameters
    agent.set_parameters(alpha=alpha, gamma=gamma, eps=eps)
    
    # Train in batches (Vercel-compatible)
    try:
        stats = agent.train_batch(episodes)
        return jsonify({'status': 'training_completed', 'stats': stats})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/training_status', methods=['GET'])
def get_training_status():
    return jsonify(agent.get_training_stats())

@app.route('/q_values', methods=['GET'])
def get_q_values():
    # Get Q-values for the current board state
    state = tuple(game_env.board)
    q_values = {}
    
    for action in range(9):
        q_values[action] = agent.Q.get((state, action), 0.0)
    
    return jsonify(q_values)

# Vercel entry point
def handler(request):
    return app(request.environ, lambda *args: None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
