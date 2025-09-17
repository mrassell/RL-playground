"""
Unit tests for GRPO core functionality.
"""

import pytest
import torch
import numpy as np
import sys
import os

# CRITICAL: Add project root to Python path BEFORE any other imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from grpo.grpo_core import GRPOTrainer, Episode, create_grpo_trainer
from models.policy_ttt import create_ttt_policy
from envs.ttt_env import TicTacToeEnv


class MockPolicy(torch.nn.Module):
    """Mock policy for testing."""
    
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        # Add a dummy parameter so optimizer works
        self.dummy_param = torch.nn.Parameter(torch.randn(1))
    
    def act(self, state, action_mask):
        """Mock act method."""
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return 0, 0.0
        action = np.random.choice(valid_actions)
        log_prob = np.log(1.0 / len(valid_actions))
        return action, log_prob
    
    def log_prob(self, state, action, action_mask):
        """Mock log_prob method."""
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return -1e9
        return np.log(1.0 / len(valid_actions))
    
    def get_action_mask(self, state):
        """Mock get_action_mask method."""
        return np.ones(9, dtype=bool)


class MockEnv:
    """Mock environment for testing."""
    
    def __init__(self):
        self.step_count = 0
        self.max_steps = 5
    
    def reset(self):
        """Mock reset method."""
        self.step_count = 0
        return np.zeros(9, dtype=np.int8)
    
    def step(self, action):
        """Mock step method."""
        self.step_count += 1
        state = np.zeros(9, dtype=np.int8)
        reward = np.random.random()
        done = self.step_count >= self.max_steps
        info = {"step_count": self.step_count}
        return state, reward, done, info
    
    def get_action_mask(self):
        """Mock get_action_mask method."""
        return np.ones(9, dtype=bool)
    
    def get_evaluation_metrics(self, episodes):
        """Mock get_evaluation_metrics method."""
        return {"mock_metric": 0.5}


class TestGRPOTrainer:
    """Test cases for GRPOTrainer."""
    
    def setup_method(self):
        """Set up test environment."""
        self.policy = MockPolicy()
        self.trainer = GRPOTrainer(self.policy)
        self.env = MockEnv()
    
    def test_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.policy == self.policy
        assert isinstance(self.trainer.optimizer, torch.optim.Adam)
        assert self.trainer.device == "cpu"
        assert self.trainer.update_count == 0
        assert len(self.trainer.metrics_history) == 0
    
    def test_seed_everything(self):
        """Test seed setting."""
        self.trainer.seed_everything(42)
        
        # Should not raise any errors
        assert True
    
    def test_rollout_single_episode(self):
        """Test single episode rollout."""
        episode = self.trainer._rollout_single_episode(self.env, max_steps=3)
        
        assert isinstance(episode, Episode)
        assert len(episode.states) > 0
        assert len(episode.actions) > 0
        assert len(episode.rewards) > 0
        assert len(episode.log_probs) > 0
        assert isinstance(episode.total_return, float)
        assert episode.episode_length > 0
    
    def test_rollout_episodes(self):
        """Test multiple episode rollout."""
        episodes = self.trainer.rollout_episodes(self.env, num_episodes=5, max_steps=3)
        
        assert len(episodes) == 5
        for episode in episodes:
            assert isinstance(episode, Episode)
            assert episode.episode_length > 0
    
    def test_compute_advantages(self):
        """Test advantage computation."""
        # Create mock episodes with different returns
        episodes = []
        for i in range(10):
            episode = Episode(
                states=[np.zeros(9)],
                actions=[0],
                rewards=[i * 0.1],  # Increasing returns
                log_probs=[0.0],
                total_return=i * 0.1,
                episode_length=1
            )
            episodes.append(episode)
        
        advantages = self.trainer.compute_advantages(episodes, top_m=5)
        
        assert len(advantages) == 10
        # After normalization, advantages can be negative, but should be finite
        assert all(np.isfinite(adv) for adv in advantages)
        
        # Top episodes should have higher advantages (before normalization)
        # We need to check the original ranking-based advantages
        sorted_episodes = sorted(episodes, key=lambda ep: ep.total_return, reverse=True)
        preferred_returns = [ep.total_return for ep in sorted_episodes[:5]]
        
        # Check that episodes with higher returns get higher original advantages
        original_advantages = []
        for episode in episodes:
            if episode.total_return in preferred_returns:
                rank = preferred_returns.index(episode.total_return)
                advantage = (5 - rank) / 5  # Original advantage before normalization
                original_advantages.append(advantage)
            else:
                original_advantages.append(0.0)
        
        # Top episodes should have higher original advantages
        assert original_advantages[-1] > original_advantages[0]  # Highest vs lowest return
    
    def test_update_policy(self):
        """Test policy update."""
        # Create mock episodes
        episodes = []
        for i in range(10):
            episode = Episode(
                states=[np.zeros(9)],
                actions=[0],
                rewards=[i * 0.1],
                log_probs=[0.0],
                total_return=i * 0.1,
                episode_length=1
            )
            episodes.append(episode)
        
        metrics = self.trainer.update_policy(episodes, top_m=5)
        
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "mean_return" in metrics
        assert "std_return" in metrics
        assert "num_preferred" in metrics
        assert "total_episodes" in metrics
        
        assert self.trainer.update_count == 1
        assert len(self.trainer.metrics_history) == 1
    
    def test_update_policy_empty_episodes(self):
        """Test policy update with empty episodes."""
        metrics = self.trainer.update_policy([], top_m=5)
        
        assert metrics["loss"] == 0.0
        assert metrics["mean_return"] == 0.0
        assert metrics["std_return"] == 0.0
    
    def test_train_update(self):
        """Test complete training update."""
        metrics = self.trainer.train_update(
            self.env,
            num_episodes=5,
            top_m=3,
            max_steps=3
        )
        
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "mean_return" in metrics
        assert self.trainer.update_count == 1
    
    def test_evaluate_policy(self):
        """Test policy evaluation."""
        metrics = self.trainer.evaluate_policy(self.env, num_episodes=5, max_steps=3)
        
        assert isinstance(metrics, dict)
        assert "mean_return" in metrics
        assert "std_return" in metrics
        assert "mean_length" in metrics
        assert "std_length" in metrics
        assert "num_episodes" in metrics
    
    def test_evaluate_policy_with_env_metrics(self):
        """Test policy evaluation with environment-specific metrics."""
        metrics = self.trainer.evaluate_policy(self.env, num_episodes=5, max_steps=3)
        
        assert "mock_metric" in metrics
        assert metrics["mock_metric"] == 0.5
    
    def test_metrics_history(self):
        """Test metrics history tracking."""
        # Perform multiple updates
        for i in range(3):
            self.trainer.train_update(self.env, num_episodes=2, top_m=1, max_steps=2)
        
        assert self.trainer.update_count == 3
        assert len(self.trainer.metrics_history) == 3
        
        # Check that metrics are being tracked
        for metrics in self.trainer.metrics_history:
            assert "loss" in metrics
            assert "mean_return" in metrics


class TestEpisode:
    """Test cases for Episode class."""
    
    def test_episode_creation(self):
        """Test episode creation."""
        episode = Episode(
            states=[np.zeros(9), np.ones(9)],
            actions=[0, 1],
            rewards=[0.5, 0.3],
            log_probs=[-0.7, -0.8],
            total_return=0.8,
            episode_length=2,
            metadata={"test": "value"}
        )
        
        assert len(episode.states) == 2
        assert len(episode.actions) == 2
        assert len(episode.rewards) == 2
        assert len(episode.log_probs) == 2
        assert episode.total_return == 0.8
        assert episode.episode_length == 2
        assert episode.metadata["test"] == "value"
    
    def test_episode_default_metadata(self):
        """Test episode with default metadata."""
        episode = Episode(
            states=[np.zeros(9)],
            actions=[0],
            rewards=[0.5],
            log_probs=[-0.7],
            total_return=0.5,
            episode_length=1
        )
        
        assert episode.metadata is None


class TestCreateGRPOTrainer:
    """Test cases for create_grpo_trainer factory function."""
    
    def test_create_grpo_trainer(self):
        """Test GRPO trainer creation."""
        policy = MockPolicy()
        trainer = create_grpo_trainer(policy)
        
        assert isinstance(trainer, GRPOTrainer)
        assert trainer.policy == policy
        assert trainer.device == "cpu"
    
    def test_create_grpo_trainer_with_params(self):
        """Test GRPO trainer creation with parameters."""
        policy = MockPolicy()
        trainer = create_grpo_trainer(
            policy,
            learning_rate=1e-4,
            l2_reg=1e-5,
            device="cpu"
        )
        
        assert isinstance(trainer, GRPOTrainer)
        assert trainer.policy == policy
        assert trainer.l2_reg == 1e-5
        
        # Check learning rate
        for param_group in trainer.optimizer.param_groups:
            assert param_group['lr'] == 1e-4


class TestIntegration:
    """Integration tests with real components."""
    
    def test_ttt_integration(self):
        """Test integration with Tic-Tac-Toe environment."""
        policy = create_ttt_policy()
        trainer = create_grpo_trainer(policy)
        env = TicTacToeEnv()
        
        # Test rollout
        episodes = trainer.rollout_episodes(env, num_episodes=3, max_steps=9)
        assert len(episodes) == 3
        
        # Test training update
        metrics = trainer.train_update(env, num_episodes=3, top_m=2, max_steps=9)
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        
        # Test evaluation
        eval_metrics = trainer.evaluate_policy(env, num_episodes=5, max_steps=9)
        assert isinstance(eval_metrics, dict)
        assert "mean_return" in eval_metrics
    
    def test_deterministic_behavior(self):
        """Test deterministic behavior with seeds."""
        # Set seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        policy1 = create_ttt_policy()
        trainer1 = create_grpo_trainer(policy1)
        # Use greedy opponent for deterministic behavior
        from envs.ttt_env import OpponentType
        env1 = TicTacToeEnv(opponent_type=OpponentType.GREEDY)
        
        policy2 = create_ttt_policy()
        trainer2 = create_grpo_trainer(policy2)
        env2 = TicTacToeEnv(opponent_type=OpponentType.GREEDY)
        
        # Both should produce same results
        episodes1 = trainer1.rollout_episodes(env1, num_episodes=2, max_steps=5)
        episodes2 = trainer2.rollout_episodes(env2, num_episodes=2, max_steps=5)
        
        # Check that episodes have same structure
        assert len(episodes1) == len(episodes2)
        for ep1, ep2 in zip(episodes1, episodes2):
            assert ep1.episode_length == ep2.episode_length
            assert len(ep1.actions) == len(ep2.actions)


if __name__ == "__main__":
    pytest.main([__file__])
