"""
GRPO (Group Relative Policy Optimization) Core Implementation

A minimal implementation of GRPO-style post-training that works for both
Tic-Tac-Toe and Edit-Agent environments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class Episode:
    """Represents a complete episode with states, actions, rewards, and log probabilities."""
    states: List[Any]
    actions: List[int]
    rewards: List[float]
    log_probs: List[float]
    total_return: float
    episode_length: int
    metadata: Dict[str, Any] = None


class GRPOTrainer:
    """
    GRPO Trainer that implements group relative policy optimization.
    
    Key components:
    1. Collect K episodes per update
    2. Score episodes by total return
    3. Rank episodes and select top-m as preferred
    4. Compute advantages from rankings
    5. Update policy using advantage-weighted loss
    """
    
    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 1e-3,
        l2_reg: float = 1e-4,
        device: str = "cpu"
    ):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        self.l2_reg = l2_reg
        self.device = device
        self.policy.to(device)
        
        # Training metrics
        self.update_count = 0
        self.metrics_history = []
    
    def seed_everything(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def rollout_episodes(
        self,
        env,
        num_episodes: int,
        max_steps: int = 100
    ) -> List[Episode]:
        """
        Roll out episodes using the current policy.
        
        Args:
            env: Environment instance
            num_episodes: Number of episodes to collect
            max_steps: Maximum steps per episode
            
        Returns:
            List of Episode objects
        """
        episodes = []
        
        for _ in range(num_episodes):
            episode = self._rollout_single_episode(env, max_steps)
            episodes.append(episode)
        
        return episodes
    
    def _rollout_single_episode(self, env, max_steps: int) -> Episode:
        """Roll out a single episode."""
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        state = env.reset()
        done = False
        step = 0
        
        while not done and step < max_steps:
            states.append(state)
            
            # Get action mask
            action_mask = env.get_action_mask()
            
            # Sample action from policy
            action, log_prob = self.policy.act(state, action_mask)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = next_state
            step += 1
        
        # Calculate total return
        total_return = sum(rewards)
        
        return Episode(
            states=states,
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            total_return=total_return,
            episode_length=len(actions),
            metadata=info if 'info' in locals() else {}
        )
    
    def compute_advantages(self, episodes: List[Episode], top_m: int) -> List[float]:
        """
        Compute advantages using group relative ranking.
        
        Args:
            episodes: List of episodes
            top_m: Number of top episodes to consider as preferred
            
        Returns:
            List of advantages for each episode
        """
        # Sort episodes by total return (descending)
        sorted_episodes = sorted(episodes, key=lambda ep: ep.total_return, reverse=True)
        
        # Select top-m episodes as preferred
        preferred_episodes = sorted_episodes[:top_m]
        
        # Compute advantages based on ranking
        advantages = []
        preferred_returns = [ep.total_return for ep in preferred_episodes]
        
        for episode in episodes:
            if episode.total_return in preferred_returns:
                # Rank-based advantage: higher rank = higher advantage
                rank = preferred_returns.index(episode.total_return)
                advantage = (top_m - rank) / top_m  # Normalize to [0, 1]
            else:
                advantage = 0.0  # Non-preferred episodes get 0 advantage
            
            advantages.append(advantage)
        
        # Normalize advantages across the batch
        if len(advantages) > 0:
            mean_adv = np.mean(advantages)
            std_adv = np.std(advantages)
            if std_adv > 0:
                advantages = [(adv - mean_adv) / std_adv for adv in advantages]
        
        return advantages
    
    def update_policy(
        self,
        episodes: List[Episode],
        top_m: int
    ) -> Dict[str, float]:
        """
        Update policy using GRPO loss.
        
        Args:
            episodes: List of episodes from rollout
            top_m: Number of top episodes to use for update
            
        Returns:
            Dictionary of training metrics
        """
        if len(episodes) == 0:
            return {"loss": 0.0, "mean_return": 0.0, "std_return": 0.0}
        
        # Compute advantages
        advantages = self.compute_advantages(episodes, top_m)
        
        # Collect all states, actions, and advantages
        all_states = []
        all_actions = []
        all_advantages = []
        
        for episode, advantage in zip(episodes, advantages):
            if advantage > 0:  # Only use preferred episodes
                all_states.extend(episode.states)
                all_actions.extend(episode.actions)
                all_advantages.extend([advantage] * len(episode.actions))
        
        if len(all_states) == 0:
            return {"loss": 0.0, "mean_return": 0.0, "std_return": 0.0}
        
        # Convert to tensors
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32, device=self.device)
        
        # Compute policy loss
        total_loss = 0.0
        for state, action in zip(all_states, all_actions):
            # Get action mask for this state
            action_mask = self.policy.get_action_mask(state)
            
            # Compute log probability of the action
            log_prob = self.policy.log_prob(state, action, action_mask)
            
            # Add to loss (negative because we want to maximize)
            total_loss += -log_prob
        
        # Average loss
        policy_loss = total_loss / len(all_states)
        
        # Add L2 regularization
        l2_loss = 0.0
        for param in self.policy.parameters():
            l2_loss += torch.sum(param ** 2)
        
        total_loss = policy_loss + self.l2_reg * l2_loss
        
        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        returns = [ep.total_return for ep in episodes]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        metrics = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item() if hasattr(policy_loss, 'item') else float(policy_loss),
            "l2_loss": l2_loss.item(),
            "mean_return": mean_return,
            "std_return": std_return,
            "num_preferred": sum(1 for adv in advantages if adv > 0),
            "total_episodes": len(episodes)
        }
        
        self.update_count += 1
        self.metrics_history.append(metrics)
        
        return metrics
    
    def train_update(
        self,
        env,
        num_episodes: int = 64,
        top_m: int = 16,
        max_steps: int = 100
    ) -> Dict[str, float]:
        """
        Perform one GRPO training update.
        
        Args:
            env: Environment instance
            num_episodes: Number of episodes to collect (K)
            top_m: Number of top episodes to use for update (m)
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary of training metrics
        """
        # Rollout episodes
        episodes = self.rollout_episodes(env, num_episodes, max_steps)
        
        # Update policy
        metrics = self.update_policy(episodes, top_m)
        
        return metrics
    
    def evaluate_policy(
        self,
        env,
        num_episodes: int = 100,
        max_steps: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate policy without training.
        
        Args:
            env: Environment instance
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary of evaluation metrics
        """
        episodes = self.rollout_episodes(env, num_episodes, max_steps)
        
        returns = [ep.total_return for ep in episodes]
        episode_lengths = [ep.episode_length for ep in episodes]
        
        metrics = {
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "num_episodes": len(episodes)
        }
        
        # Add environment-specific metrics
        if hasattr(env, 'get_evaluation_metrics'):
            env_metrics = env.get_evaluation_metrics(episodes)
            metrics.update(env_metrics)
        
        return metrics


def create_grpo_trainer(
    policy: nn.Module,
    learning_rate: float = 1e-3,
    l2_reg: float = 1e-4,
    device: str = "cpu"
) -> GRPOTrainer:
    """Factory function to create a GRPO trainer."""
    return GRPOTrainer(policy, learning_rate, l2_reg, device)
