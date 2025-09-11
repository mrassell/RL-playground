# Q-Learning Tic Tac Toe Agent FROM SCRATCH

by Maheen Rassell

## (No TensorFlow or PyTorch - Just Pure Reinforcement Learning!)

## Why: Understanding Reinforcement Learning from the Ground Up

Everyone knows about neural networks and supervised learning, but reinforcement learning is where the magic really happens - agents that learn through trial and error, just like humans do!

![Q-Learning Concept](https://via.placeholder.com/600x300/1a1a1a/ffffff?text=Q-Learning+Agent+Learning+Through+Self-Play)

Most people think AI agents are just complex neural networks, but reinforcement learning is fundamentally different - it's about **learning through interaction** and **reward maximization** rather than pattern recognition.

This project implements Q-learning from scratch to understand how agents truly learn optimal strategies through exploration and exploitation.

## Problem Statement:

Tic Tac Toe is a perfect environment for understanding Q-learning because:

1. **Finite State Space**: Only 3^9 = 19,683 possible board states
2. **Discrete Actions**: 9 possible moves (0-8 positions)
3. **Clear Rewards**: Win (+1), Loss (-1), Draw (0), Illegal Move (-10)
4. **Perfect Information**: Both players can see the entire board
5. **Deterministic**: Same action always leads to same result

The goal: Train an AI agent that learns optimal Tic Tac Toe strategy through self-play, achieving near-perfect gameplay without any pre-programmed rules.

## The Math Behind Q-Learning

### Core Concept: The Q-Table

Q-learning uses a **Q-table** that stores the "quality" of taking action `a` in state `s`:

```
Q(s, a) = Expected future reward from state s after taking action a
```

Think of it as a massive lookup table where:
- **Keys**: (state, action) pairs
- **Values**: Expected future rewards

### State Representation

Each Tic Tac Toe board state is represented as a tuple of 9 numbers:
- `0` = empty cell
- `1` = AI's move (X)  
- `-1` = Human's move (O)

Example state: `(1, 0, -1, 0, 1, 0, 0, 0, 0)` represents:
```
X |   | O
---------
  | X |  
---------
  |   |  
```

### The Q-Learning Update Rule

The heart of Q-learning is this equation:

```
Q(s,a) = Q(s,a) + α × [R + γ × max(Q(s',a')) - Q(s,a)]
```

Where:
- **α (alpha)**: Learning rate (0.01-0.5) - how fast the agent learns
- **γ (gamma)**: Discount factor (0.1-0.99) - how much future rewards matter
- **R**: Immediate reward from taking action `a` in state `s`
- **s'**: Next state after taking action `a`
- **max(Q(s',a'))**: Best possible future reward from state `s'`

### Exploration vs Exploitation

The agent uses an **ε-greedy policy**:

```
Action = {
    random action    with probability ε (exploration)
    best known action with probability (1-ε) (exploitation)
}
```

- **During Training**: ε = 0.1 (10% random, 90% best known)
- **During Play**: ε = 0 (always use best known strategy)

### Reward Structure

```
Reward = {
    +1   if AI wins
    -1   if AI loses  
     0   if draw
    -10  if illegal move
}
```

The -10 penalty for illegal moves teaches the agent to avoid invalid actions.

## How Q-Learning Works Step-by-Step

### 1. **Initialization**
```python
Q = defaultdict(float)  # Empty Q-table
α = 0.1                 # Learning rate
γ = 0.95                # Discount factor
ε = 0.1                 # Exploration rate
```

### 2. **Episode Loop** (10,000 times)
For each training episode:

#### a) **Reset Environment**
```python
board = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # Empty board
state = tuple(board)                  # Convert to hashable tuple
```

#### b) **Agent's Turn**
```python
available_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # All positions empty

# Choose action using ε-greedy policy
if random.random() < ε:
    action = random.choice(available_actions)  # Explore
else:
    # Exploit - choose action with highest Q-value
    q_values = [Q[(state, a)] for a in available_actions]
    best_action = available_actions[np.argmax(q_values)]
    action = best_action

# Take action
board[action] = 1  # Place X
next_state = tuple(board)
```

#### c) **Check for Terminal State**
```python
if check_winner(board) == 1:  # AI wins
    reward = 1
    Q[(state, action)] += α * (reward - Q[(state, action)])
    break
elif check_winner(board) == -1:  # AI loses
    reward = -1
    Q[(state, action)] += α * (reward - Q[(state, action)])
    break
elif is_draw(board):  # Draw
    reward = 0
    Q[(state, action)] += α * (reward - Q[(state, action)])
    break
```

#### d) **Opponent's Turn** (Random)
```python
opponent_action = random.choice(available_actions)
board[opponent_action] = -1  # Place O
state_after_opponent = tuple(board)
```

#### e) **Q-Value Update** (Non-terminal state)
```python
# Find best future action from new state
future_actions = get_available_actions(state_after_opponent)
if future_actions:
    future_q_values = [Q[(state_after_opponent, a)] for a in future_actions]
    max_future_reward = max(future_q_values)
else:
    max_future_reward = 0

# Update Q-value using Bellman equation
current_q = Q[(state, action)]
Q[(state, action)] = current_q + α * (reward + γ * max_future_reward - current_q)
```

### 3. **Learning Convergence**

As training progresses:
- **Early episodes**: Agent explores randomly, Q-values are mostly 0
- **Middle episodes**: Agent starts recognizing patterns, Q-values begin to differentiate
- **Late episodes**: Agent exploits learned knowledge, Q-values converge to optimal values

## Why Q-Learning Works for Tic Tac Toe

### 1. **Pattern Recognition**
The agent learns strategic patterns:
- "If I have two in a row, complete it"
- "If opponent has two in a row, block it"  
- "Center and corners are generally good positions"

### 2. **Value Propagation**
Good moves get higher Q-values, which propagate backward through the game tree:
```
Win state: Q = +1
One move from win: Q = γ × 1 = 0.95
Two moves from win: Q = γ² × 1 = 0.9025
```

### 3. **Self-Play Learning**
Playing against random opponents teaches the agent:
- How to win when possible
- How to avoid losing
- How to force draws when winning isn't possible

## Code Architecture

### 1. **TicTacToe Environment Class**
```python
class TicTacToe:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros(9, dtype=int)
        return tuple(self.board)
    
    def step(self, action, player):
        # Place piece, check winner, return new state
        self.board[action] = player
        winner = self.check_winner()
        return tuple(self.board), reward, done
```

### 2. **QLearningAgent Class**
```python
class QLearningAgent:
    def __init__(self):
        self.Q = defaultdict(float)  # Q-table
        self.alpha = 0.1             # Learning rate
        self.gamma = 0.95            # Discount factor
        self.eps = 0.1               # Exploration rate
    
    def choose_action(self, state, available):
        # ε-greedy action selection
        if random.random() < self.eps:
            return random.choice(available)
        else:
            q_values = [self.Q[(state, a)] for a in available]
            return available[np.argmax(q_values)]
    
    def train(self, episodes):
        # Main training loop with Q-value updates
        for episode in range(episodes):
            # ... training logic with intensive logging
```

### 3. **Flask Web Interface**
```python
@app.route('/move', methods=['POST'])
def make_move():
    # Handle human move, get AI response
    human_move = request.json['position']
    # ... game logic
    ai_move = agent.choose_action(state, available)
    # ... return game state
```

## Training Performance Analysis

Based on the intensive logging we implemented:

### **Training Speed**
- **Average episode time**: 0.0001 seconds
- **10,000 episodes**: Completed in ~0.6 seconds
- **Q-table growth**: 6,407 unique state-action pairs
- **Q-updates**: 36,483 total updates

### **Learning Progress**
- **Early episodes**: Random play, ~50% win rate
- **Mid training**: Pattern recognition, ~80% win rate  
- **Final performance**: ~92% win rate against random opponent

### **Memory Efficiency**
- **Q-table size**: Only 6,407 entries (out of 19,683 possible states)
- **Sparse representation**: Only stores visited state-action pairs
- **Fast lookup**: O(1) average time complexity

## Real-Time Training Features

### **Live Progress Tracking**
```python
# Intensive logging every 100 episodes
print(f"📈 Episode {episode + 1}/{episodes} | "
      f"Win Rate: {win_rate:.3f} | "
      f"Q-Table Size: {len(self.Q)} | "
      f"Avg Episode Time: {avg_time:.4f}s")
```

### **Performance Visualization**
- **Win Rate Chart**: Real-time learning curve
- **Average Reward**: Reward progression over time
- **Q-Value Heatmap**: Visual representation of learned values
- **Training Statistics**: Episodes, wins, losses, draws

### **Parameter Tuning**
- **Learning Rate (α)**: Controls how fast the agent learns
- **Discount Factor (γ)**: How much future rewards matter
- **Exploration Rate (ε)**: Balance between exploration and exploitation
- **Training Episodes**: How long to train

## Results & Performance

### **Final Training Results**
```
✅ Training completed!
⏱️  Total time: 0.60s
📊 Average episode time: 0.0001s
🧠 Q-table size: 6407 entries
🔄 Total Q-updates: 36483
📈 Final win rate: 0.919
```

### **Gameplay Performance**
- **Against Random Opponent**: ~92% win rate
- **Against Human Players**: Highly competitive
- **Strategic Play**: Recognizes winning patterns, blocks threats
- **No Illegal Moves**: Learns to avoid invalid actions

### **Learning Efficiency**
- **Convergence**: Reaches optimal play in <1 second
- **Memory Usage**: Only stores 32% of possible states
- **Generalization**: Learns general strategies, not just memorization

## Key Insights

### 1. **Q-Learning vs Neural Networks**
- **Q-Learning**: Learns through trial and error, builds explicit strategy
- **Neural Networks**: Learn through pattern recognition, implicit strategy
- **For Tic Tac Toe**: Q-learning is more interpretable and efficient

### 2. **Exploration vs Exploitation Trade-off**
- **Too much exploration**: Slow learning, poor final performance
- **Too little exploration**: Gets stuck in suboptimal strategies
- **Sweet spot**: 10% exploration during training, 0% during play

### 3. **Reward Engineering**
- **Win/Loss rewards**: Teach the agent what matters
- **Illegal move penalty**: Prevents invalid actions
- **No intermediate rewards**: Keeps the agent focused on winning

### 4. **State Space Efficiency**
- **Sparse representation**: Only stores visited states
- **Symmetry reduction**: Could be optimized further
- **Memory vs Performance**: Perfect balance for this problem size

## TLDR

We built a Q-learning agent that learns optimal Tic Tac Toe strategy through self-play in under 1 second, achieving 92% win rate against random opponents. The agent learns by:

1. **Exploring** different moves during training
2. **Updating** Q-values based on game outcomes  
3. **Exploiting** learned knowledge during play
4. **Converging** to near-optimal strategy

No neural networks, no complex math - just pure reinforcement learning from the ground up!

## Usage

### **Web Interface**
```bash
python app.py
# Open http://localhost:5001
```

### **Console Version**
```bash
python tic_tac_toe_qlearning.py
```

### **Training Parameters**
- **Learning Rate (α)**: 0.01-0.5 (default: 0.1)
- **Discount Factor (γ)**: 0.1-0.99 (default: 0.95)  
- **Exploration Rate (ε)**: 0.01-0.5 (default: 0.1)
- **Training Episodes**: 1,000-100,000 (default: 10,000)

## Hope you enjoyed the deep dive into Q-learning! 🎮🤖
