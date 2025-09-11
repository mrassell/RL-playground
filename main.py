from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import random
from collections import defaultdict
import asyncio
import uvicorn
from typing import Dict, List, Optional
import time

app = FastAPI(title="Q-Learning Tic Tac Toe", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------------------------------
# Tic Tac Toe Environment
# -------------------------------
class TicTacToe:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros(9, dtype=int)
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
        return None
    
    def step(self, action, player=1):
        if self.board[action] != 0 or self.done:
            return tuple(self.board), -10, True
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
        self.training_task = None
    
    def set_parameters(self, alpha=None, gamma=None, eps=None):
        if alpha is not None:
            self.alpha = alpha
        if gamma is not None:
            self.gamma = gamma
        if eps is not None:
            self.eps = eps
    
    def get_q_value(self, state, action):
        return self.Q.get((state, action), 0.0)
    
    def update_q_value(self, state, action, reward, next_state, available_actions):
        if not available_actions:
            max_next_q = 0
        else:
            max_next_q = max([self.get_q_value(next_state, a) for a in available_actions])
        
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.Q[(state, action)] = new_q
    
    def choose_action(self, state, available, training_mode=True):
        if not self.trained:
            return random.choice(available)
        elif training_mode and random.random() < self.eps:
            return random.choice(available)
        else:
            qvals = [self.get_q_value(state, a) for a in available]
            maxq = max(qvals) if qvals else 0
            best = [a for a in available if self.get_q_value(state, a) == maxq]
            return random.choice(best) if best else random.choice(available)
    
    async def train_async(self, episodes=1000):
        """Async training that can run in background"""
        print(f"🚀 Starting Q-learning training for {episodes} episodes...")
        
        self.training_stats['is_training'] = True
        self.training_stats['total_episodes'] = episodes
        self.training_stats['episodes_completed'] = 0
        self.training_stats['wins'] = 0
        self.training_stats['losses'] = 0
        self.training_stats['draws'] = 0
        
        env = TicTacToe()
        total_reward = 0
        q_updates = 0
        
        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Agent move
                available = env.available_actions()
                if not available:
                    break
                    
                action = self.choose_action(state, available, training_mode=True)
                next_state, reward, done = env.step(action, player=1)
                episode_reward += reward
                
                if done:
                    self.update_q_value(state, action, reward, next_state, [])
                    q_updates += 1
                    
                    if reward == 1:
                        self.training_stats['wins'] += 1
                    elif reward == -1:
                        self.training_stats['losses'] += 1
                    else:
                        self.training_stats['draws'] += 1
                    break
                
                # Opponent move
                opp_available = env.available_actions()
                if not opp_available:
                    break
                    
                opp_action = random.choice(opp_available)
                state_after_opp, opp_reward, done = env.step(opp_action, player=-1)
                episode_reward += opp_reward
                
                if done:
                    self.update_q_value(state, action, -1, state_after_opp, [])
                    q_updates += 1
                    self.training_stats['losses'] += 1
                    break
                
                # Non-terminal state update
                next_available = env.available_actions()
                self.update_q_value(state, action, reward, state_after_opp, next_available)
                q_updates += 1
                
                state = state_after_opp
            
            total_reward += episode_reward
            self.training_stats['episodes_completed'] += 1
            self.training_stats['win_rate'] = self.training_stats['wins'] / self.training_stats['episodes_completed']
            self.training_stats['avg_reward'] = total_reward / self.training_stats['episodes_completed']
            
            # Epsilon decay
            decay_factor = (episodes - episode) / episodes
            min_eps = 0.01
            initial_eps = 0.1
            self.eps = max(min_eps, initial_eps * decay_factor)
            
            # Progress logging every 100 episodes
            if (episode + 1) % 100 == 0:
                print(f"📈 Episode {episode + 1}/{episodes} | "
                      f"Win Rate: {self.training_stats['win_rate']:.1%} | "
                      f"Q-Table: {len(self.Q)}")
            
            # Yield control to allow other tasks
            if episode % 10 == 0:
                await asyncio.sleep(0.001)
        
        # Training completed
        self.trained = True
        self.eps = 0.01
        self.training_stats['is_training'] = False
        
        print(f"🎯 Training completed! Win rate: {self.training_stats['win_rate']:.1%}")
        return self.training_stats.copy()
    
    def get_training_stats(self):
        stats = self.training_stats.copy()
        stats['q_table_size'] = len(self.Q)
        return stats

# Global state
game_env = TicTacToe()
agent = QLearningAgent()

# -------------------------------
# Pydantic Models
# -------------------------------
class MoveRequest(BaseModel):
    position: int

class TrainingRequest(BaseModel):
    episodes: int = 1000
    alpha: float = 0.1
    gamma: float = 0.95
    eps: float = 0.1

# -------------------------------
# API Routes
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post("/reset")
async def reset_game():
    game_env.reset()
    return {
        "board": game_env.get_board_state(),
        "game_over": False,
        "winner": None
    }

@app.post("/move")
async def make_move(request: MoveRequest):
    position = request.position
    
    # Human move
    state, reward, done = game_env.step(position, player=-1)
    
    result = {
        "board": game_env.get_board_state(),
        "game_over": done,
        "winner": None,
        "message": ""
    }
    
    if done:
        if reward == -1:
            result["winner"] = "human"
            result["message"] = "You win!"
        elif reward == 0:
            result["winner"] = "draw"
            result["message"] = "It's a draw!"
        else:
            result["winner"] = "agent"
            result["message"] = "Agent wins!"
        return result
    
    # Agent move
    available = game_env.available_actions()
    if available:
        action = agent.choose_action(tuple(game_env.board), available, training_mode=False)
        state, reward, done = game_env.step(action, player=1)
        
        result["board"] = game_env.get_board_state()
        result["agent_move"] = action
        result["game_over"] = done
        
        if done:
            if reward == 1:
                result["winner"] = "agent"
                result["message"] = "Agent wins!"
            elif reward == 0:
                result["winner"] = "draw"
                result["message"] = "It's a draw!"
            else:
                result["winner"] = "human"
                result["message"] = "You win!"
    
    return result

@app.post("/train")
async def start_training(request: TrainingRequest):
    if agent.training_stats['is_training']:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    agent.set_parameters(alpha=request.alpha, gamma=request.gamma, eps=request.eps)
    
    # Start training in background
    agent.training_task = asyncio.create_task(
        agent.train_async(min(request.episodes, 10000))
    )
    
    return {
        "status": "training_started",
        "episodes": request.episodes,
        "current_stats": agent.get_training_stats()
    }

@app.get("/training_status")
async def get_training_status():
    return agent.get_training_stats()

@app.get("/q_values")
async def get_q_values():
    state = tuple(game_env.board)
    q_values = {}
    
    for action in range(9):
        q_values[action] = agent.Q.get((state, action), 0.0)
    
    return q_values

@app.get("/agent_status")
async def get_agent_status():
    return {
        "trained": agent.trained,
        "q_table_size": len(agent.Q),
        "eps": agent.eps,
        "alpha": agent.alpha,
        "gamma": agent.gamma
    }

@app.post("/reset_training")
async def reset_training():
    if agent.training_task and not agent.training_task.done():
        agent.training_task.cancel()
    
    agent.Q.clear()
    agent.training_stats = {
        'episodes_completed': 0,
        'total_episodes': 0,
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'win_rate': 0.0,
        'avg_reward': 0.0,
        'is_training': False
    }
    agent.trained = False
    agent.eps = 0.1
    return {"status": "training_reset"}

# -------------------------------
# Run the app
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting Q-Learning Tic Tac Toe with FastAPI!")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🎮 Game Interface: http://localhost:8000")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
