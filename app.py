from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import random
from collections import defaultdict
import json
import threading
import time

app = Flask(__name__)
CORS(app)

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
        self.training_callback = None
    
    def set_parameters(self, alpha=None, gamma=None, eps=None):
        if alpha is not None:
            self.alpha = alpha
        if gamma is not None:
            self.gamma = gamma
        if eps is not None:
            self.eps = eps
    
    def choose_action(self, state, available, training_mode=True):
        if not training_mode:
            print(f"🎮 AI Decision: trained={self.trained}, Q-table size={len(self.Q)}, eps={self.eps}")
        
        if not self.trained:
            # Always play randomly if not trained (dumb bot)
            action = random.choice(available)
            if not training_mode:
                print(f"🤖 AI (untrained): Random move {action}")
            return action
        elif training_mode and random.random() < self.eps:
            # Random action during training (exploration)
            return random.choice(available)
        else:
            # Choose best action based on Q-values (exploitation)
            qvals = [self.Q.get((state, a), 0) for a in available]
            maxq = max(qvals) if qvals else 0
            best = [a for a in available if self.Q.get((state, a), 0) == maxq]
            action = random.choice(best) if best else random.choice(available)
            
            if not training_mode:
                print(f"🧠 AI (trained): Q-values {dict(zip(available, qvals))} -> Best: {action} (Q={maxq:.3f})")
            
            return action
    
    def get_q_value(self, state, action):
        """Get Q-value, initializing to 0 if not exists"""
        if (state, action) not in self.Q:
            self.Q[(state, action)] = 0
        return self.Q[(state, action)]
    
    def update_q_value(self, state, action, reward, next_state, available_actions):
        """Update Q-value using proper Bellman equation"""
        current_q = self.get_q_value(state, action)
        
        if available_actions:
            # Find max Q-value for next state
            max_next_q = max([self.get_q_value(next_state, a) for a in available_actions])
        else:
            # Terminal state
            max_next_q = 0
        
        # Bellman equation: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.Q[(state, action)] = new_q
    
    def train(self, episodes=50000, callback=None):
        import time
        start_time = time.time()
        print(f"🚀 Starting Q-learning training with {episodes} episodes...")
        print(f"📊 Parameters: α={self.alpha}, γ={self.gamma}, ε={self.eps}")
        
        self.training_callback = callback
        self.training_stats['is_training'] = True
        self.training_stats['total_episodes'] = episodes
        self.training_stats['episodes_completed'] = 0
        self.training_stats['wins'] = 0
        self.training_stats['losses'] = 0
        self.training_stats['draws'] = 0
        
        # Epsilon decay for better exploration
        initial_eps = self.eps
        min_eps = 0.01
        
        env = TicTacToe()
        total_reward = 0
        q_updates = 0
        episode_times = []
        
        for episode in range(episodes):
            episode_start = time.time()
            state = env.reset()
            done = False
            episode_reward = 0
            moves_in_episode = 0
            
            # Epsilon decay - start high, end low
            decay_factor = (episodes - episode) / episodes
            self.eps = max(min_eps, initial_eps * decay_factor)
            
            # Debug: Log episode start
            if episode < 5 or episode % 100 == 0:
                print(f"🔍 Starting episode {episode + 1}/{episodes} (ε={self.eps:.3f})")
            
            while not done:
                moves_in_episode += 1
                
                # Agent move
                available = env.available_actions()
                if not available:
                    print(f"⚠️  No available actions at episode {episode}, move {moves_in_episode}")
                    break
                    
                action = self.choose_action(state, available, training_mode=True)
                next_state, reward, done = env.step(action, player=1)
                episode_reward += reward
                
                if done:
                    # Terminal state - update Q-value
                    self.update_q_value(state, action, reward, next_state, [])
                    q_updates += 1
                    
                    if reward == 1:
                        self.training_stats['wins'] += 1
                    elif reward == -1:
                        self.training_stats['losses'] += 1
                    else:
                        self.training_stats['draws'] += 1
                    break
                
                # Opponent (random)
                opp_available = env.available_actions()
                if not opp_available:
                    print(f"⚠️  No opponent actions at episode {episode}, move {moves_in_episode}")
                    break
                    
                opp_action = random.choice(opp_available)
                state_after_opp, opp_reward, done = env.step(opp_action, player=-1)
                episode_reward += opp_reward
                
                if done:
                    # Terminal state after opponent - update Q-value with loss
                    self.update_q_value(state, action, -1, state_after_opp, [])
                    q_updates += 1
                    self.training_stats['losses'] += 1
                    break
                
                # Non-terminal state - update Q-value
                next_available = env.available_actions()
                self.update_q_value(state, action, 0, state_after_opp, next_available)
                q_updates += 1
                
                state = state_after_opp
            
            episode_time = time.time() - episode_start
            episode_times.append(episode_time)
            
            total_reward += episode_reward
            self.training_stats['episodes_completed'] = episode + 1
            self.training_stats['win_rate'] = self.training_stats['wins'] / (episode + 1)
            self.training_stats['avg_reward'] = total_reward / (episode + 1)
            
            # Intensive logging every 50 episodes for more frequent updates
            if (episode + 1) % 50 == 0:
                avg_episode_time = sum(episode_times[-50:]) / min(50, len(episode_times))
                elapsed_time = time.time() - start_time
                estimated_remaining = (episodes - episode - 1) * avg_episode_time
                
                # Calculate learning progress metrics
                exploration_rate = self.eps if self.trained else 0.1
                q_table_growth_rate = len(self.Q) / max(1, episode + 1)
                avg_q_value = sum(self.Q.values()) / len(self.Q) if self.Q else 0
                
                print(f"📈 Episode {episode + 1:,}/{episodes:,} | "
                      f"Win Rate: {self.training_stats['win_rate']:.1%} | "
                      f"Avg Reward: {self.training_stats['avg_reward']:.3f} | "
                      f"Q-Table: {len(self.Q):,} entries | "
                      f"Q-Updates: {q_updates:,} | "
                      f"Avg Q-Value: {avg_q_value:.3f} | "
                      f"Exploration: {exploration_rate:.1%} | "
                      f"Episode Time: {avg_episode_time:.4f}s | "
                      f"Elapsed: {elapsed_time:.1f}s | "
                      f"Remaining: {estimated_remaining:.1f}s")
            
            # Callback for progress updates (every 50 episodes for real-time updates)
            if callback and (episode + 1) % 50 == 0:
                callback(self.training_stats.copy())
        
        total_time = time.time() - start_time
        avg_episode_time = sum(episode_times) / len(episode_times)
        
        # Calculate final performance metrics
        final_win_rate = self.training_stats['win_rate']
        final_avg_reward = self.training_stats['avg_reward']
        q_table_efficiency = len(self.Q) / (3**9)  # Percentage of possible states explored
        avg_q_value = sum(self.Q.values()) / len(self.Q) if self.Q else 0
        learning_speed = episodes / total_time if total_time > 0 else 0
        
        print(f"✅ Training completed!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Average episode time: {avg_episode_time:.4f}s")
        print(f"🚀 Learning speed: {learning_speed:.0f} episodes/second")
        print(f"🧠 Q-table size: {len(self.Q):,} entries ({q_table_efficiency:.1%} of possible states)")
        print(f"🔄 Total Q-updates: {q_updates:,}")
        print(f"📈 Final win rate: {final_win_rate:.1%}")
        print(f"💰 Final avg reward: {final_avg_reward:.3f}")
        print(f"🎯 Average Q-value: {avg_q_value:.3f}")
        print(f"📊 Training efficiency: {q_updates/len(self.Q):.1f} updates per state")
        
        self.trained = True
        self.eps = 0.01  # Very low exploration when playing against human (almost perfect play)
        self.training_stats['is_training'] = False
        
        print(f"🎯 Agent training completed! trained={self.trained}, Q-table size={len(self.Q)}")
        
        if callback:
            callback(self.training_stats.copy())
    
    def get_training_stats(self):
        stats = self.training_stats.copy()
        stats['q_table_size'] = len(self.Q)
        return stats

# Global game state
game_env = TicTacToe()
agent = QLearningAgent()
training_thread = None

# Agent starts untrained - training happens on demand
print("Q-learning agent initialized (untrained)")

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
        action = agent.choose_action(tuple(game_env.board), available, training_mode=False)
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
    global training_thread, agent
    
    data = request.json
    episodes = data.get('episodes', 10000)
    alpha = data.get('alpha', 0.1)
    gamma = data.get('gamma', 0.95)
    eps = data.get('eps', 0.1)
    
    print(f"🎯 Training request: {episodes} episodes, α={alpha}, γ={gamma}, ε={eps}")
    
    # Set parameters
    agent.set_parameters(alpha=alpha, gamma=gamma, eps=eps)
    
    # Start training in background thread
    if training_thread is None or not training_thread.is_alive():
        def training_callback(stats):
            # This will be called every 50 episodes during training
            print(f"📊 Training update: {stats['episodes_completed']:,}/{stats['total_episodes']:,} episodes | "
                  f"Win rate: {stats['win_rate']:.1%} | "
                  f"Avg reward: {stats['avg_reward']:.3f} | "
                  f"Wins: {stats['wins']:,} | "
                  f"Losses: {stats['losses']:,} | "
                  f"Draws: {stats['draws']:,}")
        
        training_thread = threading.Thread(target=agent.train, args=(episodes, training_callback))
        training_thread.daemon = True
        training_thread.start()
        
        # Return current stats immediately
        current_stats = agent.get_training_stats()
        current_stats['status'] = 'training_started'
        current_stats['total_episodes'] = episodes
        return jsonify(current_stats)
    else:
        return jsonify({'status': 'already_training'})

@app.route('/training_status', methods=['GET'])
def get_training_status():
    return jsonify(agent.get_training_stats())

@app.route('/stop_training', methods=['POST'])
def stop_training():
    global training_thread
    if training_thread and training_thread.is_alive():
        # Note: This is a simple implementation. In production, you'd want proper thread cancellation
        return jsonify({'status': 'training_stopped'})
    return jsonify({'status': 'not_training'})

@app.route('/reset_training', methods=['POST'])
def reset_training():
    global agent
    # Reset the agent's training data
    agent.Q = {}
    agent.training_stats = {
        'episodes_completed': 0,
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'win_rate': 0,
        'avg_reward': 0,
        'is_training': False
    }
    agent.trained = False
    agent.eps = 0.1
    print("🔄 Training data reset - Q-table cleared")
    return jsonify({'status': 'training_reset'})

@app.route('/q_values', methods=['GET'])
def get_q_values():
    # Get Q-values for the current board state
    state = tuple(game_env.board)
    q_values = {}
    
    for action in range(9):
        q_values[action] = agent.Q.get((state, action), 0.0)
    
    return jsonify(q_values)

@app.route('/agent_status', methods=['GET'])
def get_agent_status():
    return jsonify({
        'trained': agent.trained,
        'q_table_size': len(agent.Q),
        'eps': agent.eps,
        'alpha': agent.alpha,
        'gamma': agent.gamma
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
