"""
Unit tests for Tic-Tac-Toe environment.
"""

import pytest
import numpy as np
import sys
import os

# CRITICAL: Add project root to Python path BEFORE any other imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.ttt_env import TicTacToeEnv, OpponentType


class TestTicTacToeEnv:
    """Test cases for TicTacToeEnv."""
    
    def setup_method(self):
        """Set up test environment."""
        self.env = TicTacToeEnv()
    
    def test_reset(self):
        """Test environment reset."""
        state = self.env.reset()
        
        # Check initial state
        assert isinstance(state, np.ndarray)
        assert state.shape == (9,)
        assert np.all(state == 0)  # All positions empty
        assert not self.env.game_over
        assert self.env.winner is None
        assert self.env.step_count == 0
    
    def test_valid_action(self):
        """Test valid action detection."""
        self.env.reset()
        
        # All positions should be valid initially
        for i in range(9):
            assert self.env._is_valid_action(i)
        
        # Make a move
        self.env.board[0] = 1
        assert not self.env._is_valid_action(0)  # Occupied position
        assert self.env._is_valid_action(1)  # Empty position
    
    def test_action_mask(self):
        """Test action mask generation."""
        self.env.reset()
        
        mask = self.env.get_action_mask()
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (9,)
        assert np.all(mask)  # All positions should be valid initially
        
        # Make a move
        self.env.board[0] = 1
        mask = self.env.get_action_mask()
        assert not mask[0]  # Position 0 should be invalid
        assert mask[1]  # Position 1 should be valid
    
    def test_winner_detection(self):
        """Test winner detection."""
        self.env.reset()
        
        # No winner initially
        assert self.env._check_winner() is None
        
        # Test row win
        self.env.board = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        assert self.env._check_winner() == 1
        
        # Test column win
        self.env.board = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=np.int8)
        assert self.env._check_winner() == 1
        
        # Test diagonal win
        self.env.board = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.int8)
        assert self.env._check_winner() == 1
        
        # Test opponent win
        self.env.board = np.array([-1, -1, -1, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        assert self.env._check_winner() == -1
    
    def test_board_full(self):
        """Test board full detection."""
        self.env.reset()
        
        # Empty board
        assert not self.env._is_board_full()
        
        # Full board
        self.env.board = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1], dtype=np.int8)
        assert self.env._is_board_full()
    
    def test_step_valid_move(self):
        """Test valid move step."""
        self.env.reset()
        
        state, reward, done, info = self.env.step(0)
        
        # Check state
        assert isinstance(state, np.ndarray)
        assert state[0] == 1  # Agent's move
        assert not done  # Game should continue
        
        # Check reward
        assert isinstance(reward, float)
        
        # Check info
        assert isinstance(info, dict)
        assert "step_count" in info
    
    def test_step_illegal_move(self):
        """Test illegal move step."""
        self.env.reset()
        
        # Make first move
        self.env.board[0] = 1
        
        # Try to make illegal move
        state, reward, done, info = self.env.step(0)
        
        # Should get penalty
        assert reward == self.env.illegal_move_penalty
        assert not done
        assert "error" in info
    
    def test_game_completion(self):
        """Test game completion scenarios."""
        self.env.reset()
        
        # Test win scenario
        self.env.board = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        state, reward, done, info = self.env.step(2)  # Complete row
        
        assert done
        assert self.env.winner == 1
        assert reward == self.env.win_reward
        assert "winner" in info
    
    def test_reward_shaping(self):
        """Test reward shaping features."""
        self.env.reset()
        
        # Test center bonus
        state, reward, done, info = self.env.step(4)  # Center position
        assert reward >= self.env.center_bonus
    
    def test_opponent_types(self):
        """Test different opponent types."""
        # Test random opponent
        env_random = TicTacToeEnv(opponent_type=OpponentType.RANDOM)
        env_random.reset()
        
        # Should be able to get opponent action
        action = env_random._get_opponent_action()
        assert action is not None or env_random._is_board_full()
        
        # Test greedy opponent
        env_greedy = TicTacToeEnv(opponent_type=OpponentType.GREEDY)
        env_greedy.reset()
        
        action = env_greedy._get_opponent_action()
        assert action is not None or env_greedy._is_board_full()
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        self.env.reset()
        
        # Create mock episodes
        episodes = []
        for i in range(10):
            episode = type('Episode', (), {
                'metadata': {'winner': 1 if i < 6 else -1 if i < 8 else 0}
            })()
            episodes.append(episode)
        
        metrics = self.env.get_evaluation_metrics(episodes)
        
        assert "win_rate" in metrics
        assert "loss_rate" in metrics
        assert "draw_rate" in metrics
        assert metrics["win_rate"] == 0.6  # 6 wins out of 10
        assert metrics["loss_rate"] == 0.2  # 2 losses out of 10
        assert metrics["draw_rate"] == 0.2  # 2 draws out of 10
    
    def test_render_state(self):
        """Test state rendering."""
        self.env.reset()
        
        # Empty board
        rendered = self.env.render_state()
        assert isinstance(rendered, str)
        assert " " in rendered  # Empty positions
        
        # Make some moves
        self.env.board = np.array([1, -1, 0, 0, 1, 0, 0, 0, -1], dtype=np.int8)
        rendered = self.env.render_state()
        assert "X" in rendered  # Agent's moves
        assert "O" in rendered  # Opponent's moves
    
    def test_deterministic_seeds(self):
        """Test deterministic behavior with seeds."""
        # Use greedy opponent for deterministic behavior
        env1 = TicTacToeEnv(opponent_type=OpponentType.GREEDY)
        env2 = TicTacToeEnv(opponent_type=OpponentType.GREEDY)
        
        # Both should start the same
        state1 = env1.reset()
        state2 = env2.reset()
        assert np.array_equal(state1, state2)
        
        # Same moves should produce same results with greedy opponent
        state1, reward1, done1, info1 = env1.step(0)
        state2, reward2, done2, info2 = env2.step(0)
        
        assert np.array_equal(state1, state2)
        assert reward1 == reward2
        assert done1 == done2


if __name__ == "__main__":
    pytest.main([__file__])
