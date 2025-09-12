"""
Q-Learning Tic Tac Toe - Complete Monofile
==========================================

A complete Q-learning implementation for Tic Tac Toe with Streamlit interface.
Everything in one file - just run: streamlit run tic_tac_toe_qlearning_complete.py

Author: Maheen Rassell
"""

import streamlit as st
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# TIC TAC TOE ENVIRONMENT
# =============================================================================

class TicTacToe:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros(9, dtype=int)  # 0 empty, 1 agent (X), -1 human (O)
        self.done = False
        return tuple(self.board)
    
    def check_winner(self, board):
        """Check if there's a winner on the board"""
        combos = [(0,1,2),(3,4,5),(6,7,8),  # rows
                  (0,3,6),(1,4,7),(2,5,8),  # columns
                  (0,4,8),(2,4,6)]          # diagonals
        
        for (i,j,k) in combos:
            s = board[i] + board[j] + board[k]
            if s == 3: return 1   # agent wins
            if s == -3: return -1 # opponent wins
        
        if not 0 in board:
            return 0   # draw
        return None    # ongoing
    
    def step(self, action, player=1):
        """Make a move and return new state, reward, done"""
        if self.board[action] != 0 or self.done:
            return tuple(self.board), -10, True  # illegal move penalty
        
        self.board[action] = player
        winner = self.check_winner(self.board)
        
        if winner is not None:
            self.done = True
            return tuple(self.board), winner, True
        
        return tuple(self.board), 0, False
    
    def available_actions(self):
        """Get list of available actions (empty cells)"""
        return [i for i in range(9) if self.board[i] == 0]
    
    def get_board_state(self):
        """Get current board as list"""
        return self.board.tolist()
    
    def display_board(self):
        """Display the board in a nice format"""
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        board_display = [symbols[cell] for cell in self.board]
        
        return f"""
        {board_display[0]} | {board_display[1]} | {board_display[2]}
        --|---|--
        {board_display[3]} | {board_display[4]} | {board_display[5]}
        --|---|--
        {board_display[6]} | {board_display[7]} | {board_display[8]}
        """

# =============================================================================
# Q-LEARNING AGENT
# =============================================================================

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.95, eps=0.2):
        self.Q = defaultdict(float)  # Q-table: (state, action) -> Q-value
        self.alpha = alpha           # Learning rate
        self.gamma = gamma           # Discount factor
        self.eps = eps               # Exploration rate
        self.trained = False
        
        # Training statistics
        self.training_stats = {
            'episodes_completed': 0,
            'total_episodes': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'win_rate': 0.0,
            'avg_reward': 0.0,
            'is_training': False,
            'training_history': []
        }
    
    def set_parameters(self, alpha=None, gamma=None, eps=None):
        """Update learning parameters"""
        if alpha is not None: self.alpha = alpha
        if gamma is not None: self.gamma = gamma
        if eps is not None: self.eps = eps
    
    def get_winning_moves(self, board, player):
        """Find moves that would win for the given player"""
        winning_moves = []
        combos = [(0,1,2),(3,4,5),(6,7,8),  # rows
                  (0,3,6),(1,4,7),(2,5,8),  # columns
                  (0,4,8),(2,4,6)]          # diagonals
        
        for (i,j,k) in combos:
            values = [board[i], board[j], board[k]]
            if values.count(player) == 2 and values.count(0) == 1:
                # Two of player's pieces and one empty - find the empty one
                for pos in [i,j,k]:
                    if board[pos] == 0:
                        winning_moves.append(pos)
        return winning_moves
    
    def get_blocking_moves(self, board, opponent):
        """Find moves that would block the opponent from winning"""
        return self.get_winning_moves(board, opponent)
    
    def get_corner_moves(self, board):
        """Get available corner moves (0, 2, 6, 8)"""
        corners = [0, 2, 6, 8]
        return [pos for pos in corners if board[pos] == 0]
    
    def get_center_move(self, board):
        """Get center move if available"""
        return 4 if board[4] == 0 else None
    
    def get_edge_moves(self, board):
        """Get available edge moves (1, 3, 5, 7)"""
        edges = [1, 3, 5, 7]
        return [pos for pos in edges if board[pos] == 0]
    
    def get_q_value(self, state, action):
        """Safely get Q-value, initializing to 0 if not present"""
        return self.Q.get((state, action), 0.0)
    
    def update_q_value(self, state, action, reward, next_state, available_actions):
        """Update Q-value using Bellman equation"""
        if not available_actions:
            max_next_q = 0
        else:
            max_next_q = max([self.get_q_value(next_state, a) for a in available_actions])
        
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.Q[(state, action)] = new_q
    
    def choose_action(self, state, available, training_mode=True):
        """Choose action using strategic priority + epsilon-greedy policy"""
        if not available:
            return None
        
        board = list(state)
        
        # Strategic moves (always prioritize these)
        # 1. WIN: Take winning move if available
        winning_moves = self.get_winning_moves(board, 1)  # Agent is player 1
        if winning_moves:
            return random.choice(winning_moves)
        
        # 2. BLOCK: Block opponent from winning
        blocking_moves = self.get_blocking_moves(board, -1)  # Opponent is player -1
        if blocking_moves:
            return random.choice(blocking_moves)
        
        # 3. CENTER: Take center if available
        center = self.get_center_move(board)
        if center and center in available:
            return center
        
        # 4. CORNERS: Take corner if available
        corner_moves = self.get_corner_moves(board)
        corner_moves = [m for m in corner_moves if m in available]
        if corner_moves:
            return random.choice(corner_moves)
        
        # 5. EDGES: Take edge if available
        edge_moves = self.get_edge_moves(board)
        edge_moves = [m for m in edge_moves if m in available]
        if edge_moves:
            return random.choice(edge_moves)
        
        # 6. Q-LEARNING: Use learned Q-values
        if self.trained and (not training_mode or random.random() >= self.eps):
            qvals = [self.get_q_value(state, a) for a in available]
            maxq = max(qvals) if qvals else 0
            best = [a for a in available if self.get_q_value(state, a) == maxq]
            if best:
                return random.choice(best)
        
        # 7. RANDOM: Fallback to random
        return random.choice(available)
    
    def train(self, episodes=1000, progress_callback=None):
        """Train the agent using Q-learning with improved strategy"""
        st.info(f"🚀 Starting Q-learning training for {episodes} episodes...")
        st.info(f"📊 Parameters: α={self.alpha}, γ={self.gamma}, ε={self.eps}")
        
        # Initialize training
        self.training_stats['is_training'] = True
        self.training_stats['total_episodes'] = episodes
        self.training_stats['episodes_completed'] = 0
        self.training_stats['wins'] = 0
        self.training_stats['losses'] = 0
        self.training_stats['draws'] = 0
        self.training_stats['training_history'] = []
        
        env = TicTacToe()
        total_reward = 0
        q_updates = 0
        start_time = time.time()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            move_count = 0
            
            while not done and move_count < 9:
                # Agent move
                available = env.available_actions()
                if not available:
                    break
                
                action = self.choose_action(state, available, training_mode=True)
                next_state, reward, done = env.step(action, player=1)
                episode_reward += reward
                move_count += 1
                
                if done:
                    # Terminal state - update Q-value with proper reward
                    if reward == 1:
                        self.update_q_value(state, action, 100, next_state, [])  # Huge win reward
                        self.training_stats['wins'] += 1
                    elif reward == -1:
                        self.update_q_value(state, action, -100, next_state, [])  # Huge loss penalty
                        self.training_stats['losses'] += 1
                    else:
                        self.update_q_value(state, action, 50, next_state, [])   # Good draw reward
                        self.training_stats['draws'] += 1
                    q_updates += 1
                    break
                
                # Opponent (random) move
                opp_available = env.available_actions()
                if not opp_available:
                    break
                
                opp_action = random.choice(opp_available)
                state_after_opp, opp_reward, done = env.step(opp_action, player=-1)
                episode_reward += opp_reward
                move_count += 1
                
                if done:
                    # Terminal state after opponent move
                    if opp_reward == -1:  # Opponent won
                        self.update_q_value(state, action, -100, state_after_opp, [])  # Huge loss penalty
                        self.training_stats['losses'] += 1
                    else:  # Draw
                        self.update_q_value(state, action, 50, state_after_opp, [])   # Good draw reward
                        self.training_stats['draws'] += 1
                    q_updates += 1
                    break
                
                # Non-terminal state - give strategic rewards
                strategic_reward = 0
                
                # Reward for strategic moves
                if action in self.get_winning_moves(board, 1):
                    strategic_reward = 20  # Big reward for winning move
                elif action in self.get_blocking_moves(board, -1):
                    strategic_reward = 15  # Good reward for blocking
                elif action == 4:  # Center
                    strategic_reward = 5   # Small reward for center
                elif action in [0, 2, 6, 8]:  # Corners
                    strategic_reward = 3   # Small reward for corners
                
                self.update_q_value(state, action, strategic_reward, state_after_opp, next_available)
                q_updates += 1
                
                state = state_after_opp
            
            total_reward += episode_reward
            self.training_stats['episodes_completed'] += 1
            self.training_stats['win_rate'] = self.training_stats['wins'] / self.training_stats['episodes_completed']
            self.training_stats['avg_reward'] = total_reward / self.training_stats['episodes_completed']
            
            # Better epsilon decay - slower decay for more exploration
            decay_factor = max(0.1, (episodes - episode) / episodes)
            min_eps = 0.05  # Higher minimum exploration
            initial_eps = 0.3  # Start with more exploration
            self.eps = max(min_eps, initial_eps * decay_factor)
            
            # Store training history for plotting
            if episode % 10 == 0:  # Store every 10 episodes
                self.training_stats['training_history'].append({
                    'episode': episode + 1,
                    'win_rate': self.training_stats['win_rate'],
                    'avg_reward': self.training_stats['avg_reward'],
                    'q_table_size': len(self.Q),
                    'eps': self.eps
                })
            
            # Update progress
            progress = (episode + 1) / episodes
            progress_bar.progress(progress)
            
            if (episode + 1) % 100 == 0:
                elapsed = time.time() - start_time
                status_text.text(f"📈 Episode {episode + 1}/{episodes} | "
                               f"Win Rate: {self.training_stats['win_rate']:.1%} | "
                               f"Q-Table: {len(self.Q)} | "
                               f"Time: {elapsed:.1f}s")
        
        # Training completed
        self.trained = True
        self.eps = 0.01  # Very low exploration when playing against human
        self.training_stats['is_training'] = False
        
        total_time = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.text("🎯 Training completed!")
        
        st.success(f"🎯 Training completed!")
        st.success(f"⏱️ Total time: {total_time:.1f}s")
        st.success(f"🧠 Q-table size: {len(self.Q)}")
        st.success(f"🔄 Total Q-updates: {q_updates}")
        st.success(f"🏆 Final win rate: {self.training_stats['win_rate']:.1%}")
        
        return self.training_stats.copy()
    
    def get_training_stats(self):
        """Get current training statistics"""
        stats = self.training_stats.copy()
        stats['q_table_size'] = len(self.Q)
        stats['trained'] = self.trained
        return stats

# =============================================================================
# STREAMLIT INTERFACE
# =============================================================================

def main():
    st.set_page_config(
        page_title="Q-Learning Tic Tac Toe",
        page_icon="🎮",
        layout="wide"
    )
    
    st.title("🎮 Q-Learning Tic Tac Toe")
    st.markdown("**Reinforcement Learning Agent Training Interface**")
    
    # Initialize session state
    if 'game' not in st.session_state:
        st.session_state.game = TicTacToe()
    if 'agent' not in st.session_state:
        st.session_state.agent = QLearningAgent()
    if 'game_history' not in st.session_state:
        st.session_state.game_history = {'human_wins': 0, 'ai_wins': 0, 'draws': 0}
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Training parameters
        st.subheader("Training Parameters")
        alpha = st.slider("Learning Rate (α)", 0.01, 0.5, 0.5, 0.01)
        gamma = st.slider("Discount Factor (γ)", 0.1, 0.99, 0.95, 0.01)
        eps = st.slider("Exploration Rate (ε)", 0.01, 0.5, 0.2, 0.01)
        episodes = st.slider("Training Episodes", 100, 10000, 3000, 100)
        
        # Update agent parameters
        st.session_state.agent.set_parameters(alpha=alpha, gamma=gamma, eps=eps)
        
        # Training button
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training in progress..."):
                st.session_state.agent.train(episodes)
        
        # Reset buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Game"):
                st.session_state.game.reset()
                st.rerun()
        
        with col2:
            if st.button("🧠 Reset Agent"):
                st.session_state.agent = QLearningAgent(alpha, gamma, eps)
                st.session_state.game_history = {'human_wins': 0, 'ai_wins': 0, 'draws': 0}
                st.rerun()
        
        # Agent status
        st.subheader("🤖 Agent Status")
        stats = st.session_state.agent.get_training_stats()
        st.metric("Trained", "Yes" if stats.get('trained', False) else "No")
        st.metric("Q-Table Size", stats.get('q_table_size', 0))
        st.metric("Win Rate", f"{stats.get('win_rate', 0.0):.1%}")
        st.metric("Episodes", stats.get('episodes_completed', 0))
    
    # Main game area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎮 Game Board")
        
        # Display current board
        board = st.session_state.game.get_board_state()
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        
        # Create 3x3 grid of buttons
        for i in range(3):
            cols = st.columns(3)
            for j in range(3):
                idx = i * 3 + j
                with cols[j]:
                    if st.button(
                        symbols[board[idx]], 
                        key=f"cell_{idx}",
                        disabled=board[idx] != 0 or st.session_state.game.done,
                        use_container_width=True
                    ):
                        # Human move
                        state, reward, done = st.session_state.game.step(idx, player=-1)
                        
                        if done:
                            if reward == -1:
                                st.session_state.game_history['human_wins'] += 1
                                st.success("🎉 You win!")
                            elif reward == 0:
                                st.session_state.game_history['draws'] += 1
                                st.info("🤝 It's a draw!")
                            else:
                                st.session_state.game_history['ai_wins'] += 1
                                st.error("🤖 AI wins!")
                        else:
                            # AI move
                            available = st.session_state.game.available_actions()
                            if available:
                                action = st.session_state.agent.choose_action(
                                    tuple(st.session_state.game.board), 
                                    available, 
                                    training_mode=False
                                )
                                state, reward, done = st.session_state.game.step(action, player=1)
                                
                                if done:
                                    if reward == 1:
                                        st.session_state.game_history['ai_wins'] += 1
                                        st.error("🤖 AI wins!")
                                    elif reward == 0:
                                        st.session_state.game_history['draws'] += 1
                                        st.info("🤝 It's a draw!")
                                    else:
                                        st.session_state.game_history['human_wins'] += 1
                                        st.success("🎉 You win!")
                        
                        st.rerun()
        
        # Game statistics
        st.subheader("📊 Game Statistics")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Your Wins", st.session_state.game_history['human_wins'])
        with col_b:
            st.metric("AI Wins", st.session_state.game_history['ai_wins'])
        with col_c:
            st.metric("Draws", st.session_state.game_history['draws'])
    
    with col2:
        st.subheader("📈 Training Progress")
        
        if st.session_state.agent.training_stats['training_history']:
            # Create training progress chart
            df = pd.DataFrame(st.session_state.agent.training_stats['training_history'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['episode'], 
                y=df['win_rate'],
                mode='lines+markers',
                name='Win Rate',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title="Win Rate Over Time",
                xaxis_title="Episode",
                yaxis_title="Win Rate",
                yaxis=dict(range=[0, 1]),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Q-table size chart
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df['episode'], 
                y=df['q_table_size'],
                mode='lines+markers',
                name='Q-Table Size',
                line=dict(color='#ff7f0e', width=2)
            ))
            
            fig2.update_layout(
                title="Q-Table Size Growth",
                xaxis_title="Episode",
                yaxis_title="Q-Table Size",
                height=200
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Start training to see progress charts!")
        
        # Q-values heatmap
        st.subheader("🧠 Q-Values Heatmap")
        if st.session_state.agent.trained:
            state = tuple(st.session_state.game.board)
            q_values = []
            for i in range(9):
                q_values.append(st.session_state.agent.Q.get((state, i), 0.0))
            
            # Reshape to 3x3
            q_matrix = np.array(q_values).reshape(3, 3)
            
            fig3 = go.Figure(data=go.Heatmap(
                z=q_matrix,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Q-Value")
            ))
            
            fig3.update_layout(
                title="Current Board Q-Values",
                height=300
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Train the agent to see Q-values!")
    
    # Footer
    st.markdown("---")
    st.markdown("**Q-Learning Algorithm**: The agent learns optimal strategies through trial and error, updating its Q-table based on rewards and future expected rewards.")
    st.markdown("**How it works**: The agent explores different moves (ε-greedy), learns from wins/losses, and gradually becomes unbeatable!")

if __name__ == "__main__":
    main()
