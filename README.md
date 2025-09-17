# 🎮 GRPO Tic-Tac-Toe Final Boss

A real-time GRPO (Group Relative Policy Optimization) training playground with live plotting and a translucent glass UI. Train your AI agent and then face it as the "Final Boss"!

## ✨ Features

- **🔥 Real-time GRPO Training** - Watch your agent learn with live winrate visualization
- **📊 Live Plotting** - Plotly charts showing training progress in real-time
- **🎨 Glassmorphism UI** - Modern translucent glass interface with blur effects
- **⚔️ Final Boss Mode** - Face your trained agent in epic Tic-Tac-Toe battles
- **🎯 Interactive Training** - Adjustable parameters and live feedback

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Train your agent:**
   - Adjust GRPO parameters in the sidebar
   - Click "🔥 Run GRPO Warmup"
   - Watch the live winrate improvement!

4. **Face the Final Boss:**
   - Play against your trained AI
   - Try to defeat the agent you just trained!

## 🧠 GRPO Algorithm

**Group Relative Policy Optimization** is a reinforcement learning algorithm that:

- Collects episodes and ranks them by performance
- Uses the top-performing episodes to update the policy
- Computes advantages based on return differences (like DPO)
- Updates the neural network to improve performance

## 🎛️ Training Parameters

- **Episodes per Update (K)**: Number of episodes to collect per training step
- **Top Episodes (m)**: Number of best episodes to use for policy updates
- **Training Steps**: Number of training iterations
- **Learning Rate**: How fast the agent learns
- **Training Opponent**: Random or greedy opponent for training

## 🎨 UI Features

- **Glassmorphism Design**: Translucent panels with blur effects
- **Real-time Plotting**: Live winrate charts using Plotly
- **Interactive Controls**: Sliders and buttons for parameter tuning
- **Visual Feedback**: Progress bars, success messages, and animations
- **Responsive Layout**: Works on different screen sizes

## 🎯 Gameplay

1. **Training Phase**: Use GRPO to train your agent
2. **Visualization**: Watch live winrate improvement
3. **Final Boss**: Play against your trained agent
4. **Victory**: Try to defeat the AI you just created!

## 📊 Performance Metrics

- **Win Rate**: Percentage of games won against random opponent
- **Training Loss**: Policy loss during training
- **Average Return**: Mean episode return
- **Improvement**: Change in win rate over training

## 🛠️ Technical Details

- **Framework**: Streamlit for UI, PyTorch for neural networks
- **Visualization**: Plotly for real-time charts
- **Algorithm**: Custom GRPO implementation
- **Environment**: Tic-Tac-Toe with configurable opponents

## 🎮 How to Play

1. **Start Training**: Click "🔥 Run GRPO Warmup"
2. **Watch Progress**: See live winrate improvement
3. **Adjust Parameters**: Use sidebar controls to tune training
4. **Face Final Boss**: Play against your trained agent
5. **Victory**: Defeat the AI you just trained!

## 🏆 Success Criteria

- **Win Rate > 70%**: Agent is ready for Final Boss
- **No Losses**: Perfect performance against minimax
- **Live Visualization**: Real-time training progress
- **Glass UI**: Modern, translucent interface

## 🚀 Deployment

Deploy to Streamlit Cloud or Hugging Face Spaces:

1. Push to GitHub
2. Connect to Streamlit Cloud
3. Deploy with one click!

## 📝 License

MIT License - feel free to use and modify!

---

**🎮 Train, Visualize, Conquer! Face the Final Boss!** ⚔️
