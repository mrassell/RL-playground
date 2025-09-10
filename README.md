# RL-playground

A reinforcement learning project featuring a Q-learning agent that learns to play Tic Tac Toe.

## Tic Tac Toe Q-Learning Agent

This project implements a Q-learning algorithm to train an AI agent to play Tic Tac Toe. The agent learns through self-play against a random opponent and can then play against human players.

### Features

- **Q-Learning Algorithm**: Uses temporal difference learning to improve gameplay
- **Interactive Gameplay**: Play against the trained AI agent with beautiful web interface
- **Real-time Training**: Watch the AI learn with live progress visualization
- **Parameter Tuning**: Adjust learning rate, discount factor, exploration rate, and training episodes
- **Training Analytics**: Real-time charts showing win rate progression and performance metrics
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RL-playground
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

### Console Version
Run the console Tic Tac Toe game:
```bash
python tic_tac_toe_qlearning.py
```

### Web App Version
Run the web application:
```bash
python app.py
```

Then open your browser and go to: **http://localhost:5001**

The web app will:
1. Start with a pre-trained Q-learning agent (10,000 episodes)
2. Provide a beautiful, interactive web interface
3. Allow you to play against the trained agent with visual feedback
4. Enable real-time training with customizable parameters
5. Show live training progress with charts and statistics

### How to Play

- Board positions are numbered 0-8 as follows:
```
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

- Enter a number (0-8) when prompted to make your move
- The agent (X) will automatically respond with its move
- Win by getting three in a row horizontally, vertically, or diagonally

### Interactive Training Features

- **Parameter Controls**: Adjust learning rate (α), discount factor (γ), exploration rate (ε), and training episodes
- **Real-time Progress**: Watch training progress with live updates every second
- **Performance Charts**: Visualize win rate progression over training episodes
- **Training Statistics**: Monitor wins, losses, draws, and average rewards
- **Start/Stop Control**: Begin or halt training at any time

### Algorithm Details

- **Default Learning Rate (α)**: 0.1 (adjustable 0.01-0.5)
- **Default Discount Factor (γ)**: 0.95 (adjustable 0.1-0.99)
- **Default Exploration Rate (ε)**: 0.1 (adjustable 0.01-0.5)
- **Default Training Episodes**: 10,000 (adjustable 1,000-100,000)
- **Reward Structure**: +1 for win, -1 for loss, 0 for draw, -10 for illegal moves

The agent uses an epsilon-greedy policy during training and exploits learned knowledge when playing against humans.
