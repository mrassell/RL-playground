"""
Edit-Agent Tab for Streamlit App
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import difflib
import logging

logger = logging.getLogger(__name__)

from envs.edit_env import EditAgentEnv
from models.policy_edit import create_edit_policy
from grpo.grpo_core import create_grpo_trainer
from apps.streamlit_app.components.ui_utils import (
    create_edit_preview, create_diff_viewer, create_metrics_display,
    show_spinner, show_status_message, create_action_buttons,
    create_text_input, create_file_uploader
)
from apps.streamlit_app.components.charts import create_line_chart, save_plot


def initialize_edit_session_state():
    """Initialize Edit-Agent session state."""
    if 'edit_env' not in st.session_state:
        st.session_state.edit_env = EditAgentEnv()
    
    if 'edit_policy' not in st.session_state:
        st.session_state.edit_policy = create_edit_policy(
            payload_vocab_size=len(st.session_state.edit_env.payload_vocab)
        )
    
    if 'edit_trainer' not in st.session_state:
        st.session_state.edit_trainer = create_grpo_trainer(st.session_state.edit_policy)
    
    if 'edit_metrics_history' not in st.session_state:
        st.session_state.edit_metrics_history = []
    
    if 'edit_data' not in st.session_state:
        st.session_state.edit_data = st.session_state.edit_env.data
    
    if 'edit_heldout_data' not in st.session_state:
        try:
            st.session_state.edit_heldout_data = st.session_state.edit_env.load_heldout_data()
        except:
            # Create split if it doesn't exist
            train_data, test_data = st.session_state.edit_env.split_data()
            st.session_state.edit_env.save_split_data(train_data, test_data)
            st.session_state.edit_heldout_data = test_data


def train_edit_policy(num_updates: int, num_episodes: int, top_m: int, learning_rate: float):
    """Train Edit-Agent policy."""
    # Update trainer learning rate
    for param_group in st.session_state.edit_trainer.optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    # Training loop
    for update in range(num_updates):
        with st.spinner(f"Training update {update + 1}/{num_updates}"):
            # Reset environment with random task
            state = st.session_state.edit_env.reset()
            
            metrics = st.session_state.edit_trainer.train_update(
                st.session_state.edit_env,
                num_episodes=num_episodes,
                top_m=top_m,
                max_steps=20  # Max 20 edit steps
            )
            
            st.session_state.edit_metrics_history.append(metrics)
        
        # Show progress
        progress = (update + 1) / num_updates
        st.progress(progress, text=f"Training Progress: {update + 1}/{num_updates}")
        
        # Show current metrics
        if metrics:
            st.write(f"**Update {update + 1} Metrics:**")
            for name, value in metrics.items():
                st.write(f"- {name}: {value:.4f}")


def evaluate_edit_policy(num_episodes: int = 50) -> Dict[str, float]:
    """Evaluate Edit-Agent policy on held-out data."""
    with st.spinner(f"Evaluating policy on {num_episodes} held-out tasks"):
        total_exact_matches = 0
        total_distance_improvement = 0.0
        total_episodes = 0
        
        for i in range(min(num_episodes, len(st.session_state.edit_heldout_data))):
            task_id = st.session_state.edit_heldout_data.iloc[i]['id']
            state = st.session_state.edit_env.reset(task_id)
            
            # Run episode
            done = False
            step_count = 0
            initial_distance = st.session_state.edit_env.previous_distance
            
            while not done and step_count < 20:
                action_mask = st.session_state.edit_env.get_flat_action_mask()
                action, _ = st.session_state.edit_policy.act(state, action_mask)
                
                # Convert flat action to 3D
                op, loc, payload = st.session_state.edit_env.unflatten_action(action)
                
                state, reward, done, info = st.session_state.edit_env.step((op, loc, payload))
                step_count += 1
            
            # Calculate metrics
            final_distance = st.session_state.edit_env.previous_distance
            if final_distance == 0.0:
                total_exact_matches += 1
            
            total_distance_improvement += (initial_distance - final_distance)
            total_episodes += 1
        
        metrics = {
            "exact_match_rate": total_exact_matches / total_episodes if total_episodes > 0 else 0.0,
            "mean_distance_improvement": total_distance_improvement / total_episodes if total_episodes > 0 else 0.0,
            "total_episodes": total_episodes
        }
    
    return metrics


def verify_edit_outcome(task_id: int) -> Dict[str, Any]:
    """Verify edit outcome for a specific task."""
    state = st.session_state.edit_env.reset(task_id)
    
    # Store initial state
    initial_text = state['current_text']
    target_text = state['target_text']
    instruction = state['instruction']
    
    # Run agent
    done = False
    step_count = 0
    actions_taken = []
    
    while not done and step_count < 20:
        action_mask = st.session_state.edit_env.get_flat_action_mask()
        action, _ = st.session_state.edit_policy.act(state, action_mask)
        
        # Convert flat action to 3D
        op, loc, payload = st.session_state.edit_env.unflatten_action(action)
        actions_taken.append((op, loc, payload))
        
        state, reward, done, info = st.session_state.edit_env.step((op, loc, payload))
        step_count += 1
    
    final_text = state['current_text']
    exact_match = final_text == target_text
    
    return {
        "task_id": task_id,
        "instruction": instruction,
        "initial_text": initial_text,
        "target_text": target_text,
        "final_text": final_text,
        "exact_match": exact_match,
        "actions_taken": actions_taken,
        "step_count": step_count
    }


def render_edit_tab():
    """Render Edit-Agent tab."""
    try:
        logger.info("Rendering Edit-Agent tab...")
        st.header("✏️ Edit-Agent")
        
        # Initialize session state
        logger.info("Initializing Edit-Agent session state...")
        initialize_edit_session_state()
        logger.info("✅ Edit-Agent session state initialized")
        
        # Get configuration from sidebar
        logger.info("Getting configuration from sidebar...")
        config = st.session_state.get('config', {})
        logger.info(f"Configuration: {config}")
        
        # Training controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Train", type="primary"):
                with st.spinner("Training..."):
                    train_edit_policy(
                        num_updates=config.get('num_updates', 10),
                        num_episodes=config.get('num_episodes', 64),
                        top_m=config.get('top_m', 16),
                        learning_rate=config.get('learning_rate', 1e-3)
                    )
                st.success("Training complete!")
        
        with col2:
            if st.button("Evaluate"):
                metrics = evaluate_edit_policy()
                st.success(f"Exact match: {metrics['exact_match_rate']:.1%}")
        
        with col3:
            if st.button("Reset"):
                st.session_state.edit_policy = create_edit_policy(
                    payload_vocab_size=len(st.session_state.edit_env.payload_vocab)
                )
                st.session_state.edit_trainer = create_grpo_trainer(st.session_state.edit_policy)
                st.session_state.edit_metrics_history = []
                st.success("Reset complete!")
        
        # Action buttons
        logger.info("Creating action buttons...")
        buttons = create_action_buttons()
        logger.info("✅ Action buttons created")
    
        # Training section
        if buttons['train']:
            with st.spinner("Training Edit-Agent policy..."):
                train_edit_policy(
                    num_updates=config.get('num_updates', 10),
                    num_episodes=config.get('num_episodes', 64),
                    top_m=config.get('top_m', 16),
                    learning_rate=config.get('learning_rate', 1e-3)
                )
            
            show_status_message("Training completed!", "success")
        
        # Evaluation section
        if buttons['evaluate']:
            metrics = evaluate_edit_policy()
            create_metrics_display(metrics, "Evaluation Results")
            
            # Save evaluation results
            os.makedirs("figs", exist_ok=True)
            pd.DataFrame([metrics]).to_csv("figs/edit_evaluation.csv", index=False)
        
        # Reset button
        if buttons['reset']:
            st.session_state.edit_policy = create_edit_policy(
                payload_vocab_size=len(st.session_state.edit_env.payload_vocab)
            )
            st.session_state.edit_trainer = create_grpo_trainer(st.session_state.edit_policy)
            st.session_state.edit_metrics_history = []
            show_status_message("Policy reset!", "success")
    
        # Training metrics display
        if st.session_state.edit_metrics_history:
            st.subheader("📊 Training Metrics")
            
            # Extract metrics
            returns = [m.get('mean_return', 0) for m in st.session_state.edit_metrics_history]
            losses = [m.get('loss', 0) for m in st.session_state.edit_metrics_history]
            
            # Create charts
            col1, col2 = st.columns(2)
            
            with col1:
                if returns:
                    fig1 = create_line_chart(returns, "Mean Return", color="green")
                    st.pyplot(fig1)
                    save_plot(fig1, "figs/edit_return.png")
            
            with col2:
                if losses:
                    fig2 = create_line_chart(losses, "Training Loss", color="red")
                    st.pyplot(fig2)
                    save_plot(fig2, "figs/edit_loss.png")
    
        # Test section
        st.subheader("Test Agent")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Test Random", type="primary"):
                # Select random task from held-out data
                random_task = st.session_state.edit_heldout_data.sample(1).iloc[0]
                task_id = random_task['id']
                
                with st.spinner("Testing..."):
                    result = verify_edit_outcome(task_id)
                
                # Display results
                st.write(f"**Instruction:** {result['instruction']}")
                
                # Show diff
                create_diff_viewer(
                    result['initial_text'],
                    result['final_text'],
                    "Result"
                )
                
                # Show target
                st.write("**Target:**")
                st.code(result['target_text'], language='text')
                
                # Show metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Match", "✅" if result['exact_match'] else "❌")
                with col2:
                    st.metric("Steps", result['step_count'])
                with col3:
                    st.metric("Actions", len(result['actions_taken']))
        
        with col2:
            training_updates = len(st.session_state.edit_metrics_history)
            if training_updates > 0:
                st.info(f"Agent trained {training_updates} times")
            else:
                st.warning("Train agent first")
    
        # Custom edit section
        st.subheader("📝 Custom Edit Task")
        
        col1, col2 = st.columns(2)
        
        with col1:
            custom_text = create_text_input("Input Text", height=150)
            custom_instruction = create_text_input("Instruction", height=100)
        
        with col2:
            if st.button("Run Agent on Custom Task"):
                if custom_text and custom_instruction:
                    # Create temporary task
                    temp_task = {
                        'id': 999,
                        'before_text': custom_text,
                        'instruction': custom_instruction,
                        'after_text': custom_text  # Placeholder, agent will try to edit
                    }
                    
                    # Run agent (simplified version)
                    st.info("Custom edit functionality would be implemented here.")
                    st.write("**Input:**", custom_text)
                    st.write("**Instruction:**", custom_instruction)
                else:
                    st.warning("Please provide both text and instruction.")
        
        # Data preview
        st.subheader("📋 Training Data Preview")
        create_edit_preview(st.session_state.edit_data, num_rows=3)
        
        # Policy information
        st.subheader("🧠 Policy Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Policy Architecture:**")
            st.write("- Input: 128-dimensional state features")
            st.write("- Hidden: 64 units, 2 layers")
            st.write("- Output: Joint logits over (op, loc, payload)")
            st.write("- Actions: INSERT, REPLACE, DELETE, STOP")
        
        with col2:
            st.write("**Training Configuration:**")
            st.write(f"- Episodes per update: {config.get('num_episodes', 64)}")
            st.write(f"- Top episodes (m): {config.get('top_m', 16)}")
            st.write(f"- Learning rate: {config.get('learning_rate', 1e-3)}")
            st.write(f"- Updates completed: {len(st.session_state.edit_metrics_history)}")
    
        # Save metrics
        if st.session_state.edit_metrics_history:
            logger.info("Saving Edit-Agent metrics...")
            os.makedirs("figs", exist_ok=True)
            metrics_df = pd.DataFrame(st.session_state.edit_metrics_history)
            metrics_df.to_csv("figs/edit_metrics.csv", index=False)
            st.success("Metrics saved to figs/edit_metrics.csv")
            logger.info("✅ Edit-Agent metrics saved")
        
        logger.info("✅ Edit-Agent tab rendered successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error rendering Edit-Agent tab: {e}")
        st.error(f"Error in Edit-Agent tab: {e}")
        st.exception(e)
