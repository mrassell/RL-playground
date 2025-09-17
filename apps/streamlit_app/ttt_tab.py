"""
Tic-Tac-Toe Tab for Streamlit App
"""

import os
import sys

# CRITICAL: Add project root to Python path BEFORE any other imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add current working directory
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

from envs.ttt_env import TicTacToeEnv, OpponentType
from models.policy_ttt import create_ttt_policy
from grpo.grpo_core import create_grpo_trainer
from apps.streamlit_app.components.ui_utils import (
    create_ttt_reward_config, create_ttt_board_ui, create_metrics_display,
    show_spinner, show_status_message, create_action_buttons
)
from apps.streamlit_app.components.charts import create_line_chart, save_plot


def initialize_ttt_session_state():
    """Initialize Tic-Tac-Toe session state."""
    if 'ttt_policy' not in st.session_state:
        st.session_state.ttt_policy = create_ttt_policy()
    
    if 'ttt_trainer' not in st.session_state:
        st.session_state.ttt_trainer = create_grpo_trainer(st.session_state.ttt_policy)
    
    if 'ttt_env' not in st.session_state:
        st.session_state.ttt_env = TicTacToeEnv()
    
    if 'ttt_metrics_history' not in st.session_state:
        st.session_state.ttt_metrics_history = []
    
    if 'ttt_board_state' not in st.session_state:
        st.session_state.ttt_board_state = np.zeros(9, dtype=np.int8)
    
    if 'ttt_game_over' not in st.session_state:
        st.session_state.ttt_game_over = False
    
    if 'ttt_current_player' not in st.session_state:
        st.session_state.ttt_current_player = 1  # Human starts first


def update_ttt_rewards(config: Dict[str, float]):
    """Update Tic-Tac-Toe environment rewards."""
    st.session_state.ttt_env.win_reward = config['win_reward']
    st.session_state.ttt_env.draw_reward = config['draw_reward']
    st.session_state.ttt_env.loss_penalty = config['loss_penalty']
    st.session_state.ttt_env.center_bonus = config['center_bonus']
    st.session_state.ttt_env.fork_bonus = config['fork_bonus']
    st.session_state.ttt_env.illegal_move_penalty = config['illegal_move_penalty']


def train_ttt_policy(num_updates: int, num_episodes: int, top_m: int, learning_rate: float):
    """Train Tic-Tac-Toe policy."""
    # Update trainer learning rate
    for param_group in st.session_state.ttt_trainer.optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    # Training loop
    for update in range(num_updates):
        with st.spinner(f"Training update {update + 1}/{num_updates}"):
            metrics = st.session_state.ttt_trainer.train_update(
                st.session_state.ttt_env,
                num_episodes=num_episodes,
                top_m=top_m,
                max_steps=9  # Max 9 moves in Tic-Tac-Toe
            )
            
            st.session_state.ttt_metrics_history.append(metrics)
        
        # Show progress
        progress = (update + 1) / num_updates
        st.progress(progress, text=f"Training Progress: {update + 1}/{num_updates}")
        
        # Show current metrics
        if metrics:
            st.write(f"**Update {update + 1} Metrics:**")
            for name, value in metrics.items():
                st.write(f"- {name}: {value:.4f}")


def evaluate_ttt_policy(num_games: int = 100) -> Dict[str, float]:
    """Evaluate Tic-Tac-Toe policy."""
    with st.spinner(f"Evaluating policy over {num_games} games"):
        metrics = st.session_state.ttt_env.evaluate_vs_opponent(
            st.session_state.ttt_policy,
            num_games=num_games,
            opponent_type=OpponentType.RANDOM
        )
    
    return metrics


def play_ttt_game():
    """Play a game of Tic-Tac-Toe against the agent."""
    # Ensure environment is initialized
    if 'ttt_env' not in st.session_state:
        st.session_state.ttt_env = TicTacToeEnv()
    
    # Reset game state
    st.session_state.ttt_board_state = st.session_state.ttt_env.reset()
    st.session_state.ttt_game_over = False
    st.session_state.ttt_current_player = 1  # Human starts first
    
    st.success("New game started! You are X, agent is O. Click a position to make your move.")
    st.rerun()


def handle_ttt_move(position: int):
    """Handle Tic-Tac-Toe move."""
    if st.session_state.ttt_game_over:
        return
    
    # Ensure environment is initialized
    if 'ttt_env' not in st.session_state:
        st.session_state.ttt_env = TicTacToeEnv()
    
    if st.session_state.ttt_board_state is None:
        st.session_state.ttt_board_state = np.zeros(9, dtype=np.int8)
    
    # Human move
    if position < len(st.session_state.ttt_board_state) and st.session_state.ttt_board_state[position] == 0:
        st.session_state.ttt_board_state[position] = 1
        
        # Check for human win
        if st.session_state.ttt_env._check_winner_on_board(st.session_state.ttt_board_state) == 1:
            st.session_state.ttt_game_over = True
            st.success("🎉 You won!")
            st.rerun()
            return
        
        # Check for draw
        if np.all(st.session_state.ttt_board_state != 0):
            st.session_state.ttt_game_over = True
            st.info("🤝 It's a draw!")
            st.rerun()
            return
        
        # Agent move
        action_mask = st.session_state.ttt_env.get_action_mask()
        agent_action, _ = st.session_state.ttt_policy.act(st.session_state.ttt_board_state, action_mask)
        
        if st.session_state.ttt_board_state[agent_action] == 0:
            st.session_state.ttt_board_state[agent_action] = -1
            
            # Check for agent win
            if st.session_state.ttt_env._check_winner_on_board(st.session_state.ttt_board_state) == -1:
                st.session_state.ttt_game_over = True
                st.error("🤖 Agent won!")
                st.rerun()
                return
            
            # Check for draw
            if np.all(st.session_state.ttt_board_state != 0):
                st.session_state.ttt_game_over = True
                st.info("🤝 It's a draw!")
                st.rerun()
                return
        
        # Update the board to show both moves
        st.rerun()


def render_ttt_tab():
    """Render Tic-Tac-Toe tab."""
    try:
        # Initialize session state
        initialize_ttt_session_state()
        
        # Get configuration from sidebar
        config = st.session_state.get('config', {})
        
        # Ensure all required session state exists
        if 'ttt_board_state' not in st.session_state:
            st.session_state.ttt_board_state = np.zeros(9, dtype=np.int8)
        if 'ttt_game_over' not in st.session_state:
            st.session_state.ttt_game_over = False
        if 'ttt_current_player' not in st.session_state:
            st.session_state.ttt_current_player = 1
        
        # Training section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Train Bot", type="primary"):
                with st.spinner("Training bot..."):
                    train_ttt_policy(
                        num_updates=config.get('num_updates', 10),
                        num_episodes=config.get('num_episodes', 64),
                        top_m=config.get('top_m', 16),
                        learning_rate=config.get('learning_rate', 1e-3)
                    )
                
                # Show training results
                if st.session_state.ttt_metrics_history:
                    latest_metrics = st.session_state.ttt_metrics_history[-1]
                    mean_return = latest_metrics.get('mean_return', 0)
                    st.success(f"Bot trained! Mean return: {mean_return:.2f}")
                else:
                    st.success("Bot trained!")
        
        with col2:
            if st.button("📊 Test Bot"):
                metrics = evaluate_ttt_policy(100)
                win_rate = metrics['win_rate']
                if win_rate > 0.7:
                    st.success(f"Bot is strong! Win rate: {win_rate:.1%}")
                elif win_rate > 0.4:
                    st.info(f"Bot is learning. Win rate: {win_rate:.1%}")
                else:
                    st.warning(f"Bot needs more training. Win rate: {win_rate:.1%}")
        
        with col3:
            training_count = len(st.session_state.ttt_metrics_history)
            if training_count > 0:
                st.metric("Training Sessions", training_count)
            else:
                st.metric("Training Sessions", 0)
        
        # Reward configuration
        reward_config = create_ttt_reward_config()
        update_ttt_rewards(reward_config)
        
        # Game section
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎮 New Game", type="primary"):
                play_ttt_game()
        
        with col2:
            if st.session_state.ttt_game_over:
                if st.session_state.ttt_current_player == 1:
                    st.success("You won! 🎉")
                elif st.session_state.ttt_current_player == -1:
                    st.error("Bot won! 🤖")
                else:
                    st.info("Draw! 🤝")
        
        # Game board
        st.markdown("### 🎯 Play Against Bot")
        
        # Render board
        try:
            if st.session_state.ttt_board_state is None:
                st.session_state.ttt_board_state = np.zeros(9, dtype=np.int8)
            
            board_list = st.session_state.ttt_board_state.tolist()
            
            if not st.session_state.ttt_game_over:
                clicked_positions = create_ttt_board_ui(
                    board_list,
                    on_click=handle_ttt_move
                )
            else:
                create_ttt_board_ui(board_list)
                
        except Exception as board_error:
            st.error(f"Board error: {board_error}")
            # Reset board state
            st.session_state.ttt_board_state = np.zeros(9, dtype=np.int8)
            st.session_state.ttt_game_over = False
            st.session_state.ttt_current_player = 1
        
    except Exception as e:
        st.error(f"Error: {e}")
        # Reset everything
        initialize_ttt_session_state()
