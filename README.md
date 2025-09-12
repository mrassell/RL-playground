# 🎮 Q-Learning Tic Tac Toe - Complete Monofile

A complete Q-learning implementation for Tic Tac Toe with Streamlit interface. Everything in one file!

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run tic_tac_toe_qlearning_complete.py
   ```

3. **Open your browser** to the URL shown (usually http://localhost:8501)

## 🎯 Features

- **Complete Q-Learning Implementation**: Full reinforcement learning algorithm
- **Interactive Game Interface**: Play against the AI in real-time
- **Real-time Training**: Watch the AI learn with live progress charts
- **Visual Q-Values**: See the AI's decision-making process
- **Adjustable Parameters**: Tune learning rate, discount factor, exploration rate
- **Training Statistics**: Track win rates, Q-table growth, and more
- **Beautiful UI**: Modern Streamlit interface with charts and metrics

## 🧠 How Q-Learning Works

1. **Exploration**: The AI tries random moves to discover strategies
2. **Learning**: It updates Q-values based on wins/losses using the Bellman equation
3. **Exploitation**: Over time, it chooses the best moves it has learned
4. **Convergence**: Eventually becomes unbeatable!

## 📊 What You'll See

- **Game Board**: Interactive 3x3 grid to play against the AI
- **Training Charts**: Win rate and Q-table size over time
- **Q-Value Heatmap**: Visual representation of the AI's decision-making
- **Statistics**: Real-time tracking of wins, losses, and draws
- **Agent Status**: Training progress and Q-table size

## 🎮 How to Use

1. **Start Training**: Click "Start Training" to make the AI learn
2. **Play Games**: Click on empty cells to make your moves
3. **Watch Learning**: See the AI get smarter with each training episode
4. **Adjust Parameters**: Tune the learning parameters in the sidebar
5. **Reset**: Start fresh with a new game or untrained agent

## 🔧 Technical Details

- **Algorithm**: Q-Learning with epsilon-greedy exploration
- **State Space**: 3^9 = 19,683 possible board states
- **Action Space**: 9 possible moves (0-8)
- **Reward Structure**: +1 for win, -1 for loss, 0 for draw
- **Learning Rate**: Adjustable (default 0.1)
- **Discount Factor**: Adjustable (default 0.95)
- **Exploration Rate**: Decays from 0.1 to 0.01

## 🎓 Perfect for Demonstrations

This is ideal for showing professors, classmates, or anyone interested in:
- Reinforcement Learning
- Q-Learning algorithms
- Machine Learning concepts
- Interactive AI demonstrations

**Just run the single file and you have a complete Q-learning demonstration!**
