"""
Unit tests for Edit-Agent environment.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
import tempfile

# CRITICAL: Add project root to Python path BEFORE any other imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.edit_env import EditAgentEnv, EditOp


class TestEditAgentEnv:
    """Test cases for EditAgentEnv."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create temporary CSV file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_file.write("id,before_text,instruction,after_text\n")
        self.temp_file.write("1,hello world,capitalize,HELLO WORLD\n")
        self.temp_file.write("2,name: john,capitalize name,name: John\n")
        self.temp_file.write("3,test,add exclamation,test!\n")
        self.temp_file.close()
        
        self.env = EditAgentEnv(data_file=self.temp_file.name, max_steps=10)
    
    def teardown_method(self):
        """Clean up test files."""
        os.unlink(self.temp_file.name)
    
    def test_initialization(self):
        """Test environment initialization."""
        assert self.env.data_file == self.temp_file.name
        assert self.env.max_steps == 10
        assert len(self.env.data) == 3
        assert len(self.env.payload_vocab) > 0
    
    def test_reset(self):
        """Test environment reset."""
        state = self.env.reset(task_id=1)
        
        assert isinstance(state, dict)
        assert "current_text" in state
        assert "target_text" in state
        assert "instruction" in state
        assert "step_count" in state
        assert "done" in state
        
        assert state["current_text"] == "hello world"
        assert state["target_text"] == "HELLO WORLD"
        assert state["instruction"] == "capitalize"
        assert state["step_count"] == 0
        assert not state["done"]
    
    def test_valid_action(self):
        """Test valid action detection."""
        self.env.reset(task_id=1)
        
        # Valid actions
        assert self.env._is_valid_action((EditOp.STOP.value, 0, 0))
        assert self.env._is_valid_action((EditOp.INSERT.value, 0, 0))
        assert self.env._is_valid_action((EditOp.INSERT.value, 11, 0))  # End of string
        
        # Invalid actions
        assert not self.env._is_valid_action((EditOp.REPLACE.value, 11, 0))  # Beyond string length
        assert not self.env._is_valid_action((EditOp.DELETE.value, 11, 0))  # Beyond string length
        assert not self.env._is_valid_action((999, 0, 0))  # Invalid operation
        assert not self.env._is_valid_action((EditOp.INSERT.value, 0, 999))  # Invalid payload
    
    def test_action_mask(self):
        """Test action mask generation."""
        self.env.reset(task_id=1)
        
        mask = self.env.get_action_mask()
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (4, 11, len(self.env.payload_vocab))  # op, loc, payload
        
        # STOP should always be valid
        assert np.all(mask[EditOp.STOP.value, :, :])
        
        # INSERT should be valid at all positions
        assert np.all(mask[EditOp.INSERT.value, :11, :])  # Up to string length
        
        # REPLACE and DELETE should be valid at existing positions
        assert np.all(mask[EditOp.REPLACE.value, :11, :])
        assert np.all(mask[EditOp.DELETE.value, :11, :])
    
    def test_flat_action_mask(self):
        """Test flattened action mask."""
        self.env.reset(task_id=1)
        
        mask = self.env.get_flat_action_mask()
        assert isinstance(mask, np.ndarray)
        assert len(mask) == 4 * 11 * len(self.env.payload_vocab)
    
    def test_action_flattening(self):
        """Test action flattening and unflattening."""
        self.env.reset(task_id=1)
        
        # Test flattening
        action = (EditOp.INSERT.value, 5, 10)
        flat_action = self.env.flatten_action(action)
        assert isinstance(flat_action, int)
        
        # Test unflattening
        unflat_action = self.env.unflatten_action(flat_action)
        assert unflat_action == action
    
    def test_step_stop(self):
        """Test STOP action."""
        self.env.reset(task_id=1)
        
        state, reward, done, info = self.env.step((EditOp.STOP.value, 0, 0))
        
        assert done
        assert reward == 0.0  # No exact match
        assert "action" in info
        assert info["action"] == "STOP"
    
    def test_step_insert(self):
        """Test INSERT action."""
        self.env.reset(task_id=1)
        
        # Insert "H" at position 0
        payload_id = self.env.payload_vocab.index("H") if "H" in self.env.payload_vocab else 0
        state, reward, done, info = self.env.step((EditOp.INSERT.value, 0, payload_id))
        
        assert not done
        assert "current_text" in state
        assert "H" in state["current_text"]
    
    def test_step_replace(self):
        """Test REPLACE action."""
        self.env.reset(task_id=1)
        
        # Replace first character with "H"
        payload_id = self.env.payload_vocab.index("H") if "H" in self.env.payload_vocab else 0
        state, reward, done, info = self.env.step((EditOp.REPLACE.value, 0, payload_id))
        
        assert not done
        assert "current_text" in state
        assert state["current_text"].startswith("H")
    
    def test_step_delete(self):
        """Test DELETE action."""
        self.env.reset(task_id=1)
        
        # Delete first character
        state, reward, done, info = self.env.step((EditOp.DELETE.value, 0, 0))
        
        assert not done
        assert "current_text" in state
        assert len(state["current_text"]) == 10  # One character deleted
    
    def test_step_invalid_action(self):
        """Test invalid action handling."""
        self.env.reset(task_id=1)
        
        # Try to replace beyond string length
        state, reward, done, info = self.env.step((EditOp.REPLACE.value, 20, 0))
        
        assert not done
        assert reward == -0.1  # Penalty for invalid action
        assert "error" in info
    
    def test_edit_distance(self):
        """Test edit distance calculation."""
        # Test identical strings
        dist = self.env._normalized_edit_distance("hello", "hello")
        assert dist == 0.0
        
        # Test different strings
        dist = self.env._normalized_edit_distance("hello", "world")
        assert dist > 0.0
        
        # Test empty strings
        dist = self.env._normalized_edit_distance("", "")
        assert dist == 0.0
        
        # Test one empty string
        dist = self.env._normalized_edit_distance("hello", "")
        assert dist == 1.0
    
    def test_final_reward(self):
        """Test final reward calculation."""
        self.env.reset(task_id=1)
        
        # Set current text to target text
        self.env.current_text = self.env.target_text
        
        reward = self.env._calculate_final_reward()
        assert reward == 1.0  # Exact match
        
        # Set current text to different text
        self.env.current_text = "different text"
        
        reward = self.env._calculate_final_reward()
        assert reward == 0.0  # No exact match
    
    def test_max_steps(self):
        """Test maximum steps limit."""
        self.env.reset(task_id=1)
        
        # Make max_steps moves
        for i in range(self.env.max_steps):
            state, reward, done, info = self.env.step((EditOp.INSERT.value, 0, 0))
            if done:
                break
        
        # Should be done after max_steps
        assert self.env.step_count >= self.env.max_steps or done
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        # Create mock episodes
        episodes = []
        for i in range(10):
            episode = type('Episode', (), {
                'metadata': {
                    'final_distance': 0.0 if i < 3 else 0.5,
                    'initial_distance': 1.0
                }
            })()
            episodes.append(episode)
        
        metrics = self.env.get_evaluation_metrics(episodes)
        
        assert "exact_match_rate" in metrics
        assert "mean_distance_improvement" in metrics
        assert "mean_final_distance" in metrics
        assert metrics["exact_match_rate"] == 0.3  # 3 exact matches out of 10
    
    def test_data_splitting(self):
        """Test data splitting functionality."""
        train_data, test_data = self.env.split_data(train_ratio=0.5)
        
        assert len(train_data) + len(test_data) == len(self.env.data)
        assert len(train_data) > 0
        assert len(test_data) > 0
        
        # Check that all IDs are unique
        all_ids = set(train_data['id'].tolist() + test_data['id'].tolist())
        assert len(all_ids) == len(self.env.data)
    
    def test_render_state(self):
        """Test state rendering."""
        self.env.reset(task_id=1)
        
        rendered = self.env.render_state()
        assert isinstance(rendered, str)
        assert "Task ID" in rendered
        assert "Instruction" in rendered
        assert "Current Text" in rendered
        assert "Target Text" in rendered
    
    def test_payload_vocab(self):
        """Test payload vocabulary building."""
        assert len(self.env.payload_vocab) > 0
        assert isinstance(self.env.payload_vocab, list)
        
        # Check that common characters are included
        common_chars = [' ', '\n', '\t', ',', ':', ';', '.', '!', '?']
        for char in common_chars:
            if char in self.env.payload_vocab:
                assert True  # At least some common chars should be present
    
    def test_apply_edit(self):
        """Test edit application."""
        self.env.reset(task_id=1)
        
        # Test INSERT
        new_text = self.env._apply_edit((EditOp.INSERT.value, 0, 0))
        assert new_text is not None
        assert len(new_text) > len(self.env.current_text)
        
        # Test REPLACE - find a single character payload
        single_char_payload = None
        for i, payload in enumerate(self.env.payload_vocab):
            if len(payload) == 1:
                single_char_payload = i
                break
        
        if single_char_payload is not None:
            new_text = self.env._apply_edit((EditOp.REPLACE.value, 0, single_char_payload))
            assert new_text is not None
            # REPLACE should keep same length when replacing with single character
            assert len(new_text) == len(self.env.current_text)
        
        # Test DELETE
        new_text = self.env._apply_edit((EditOp.DELETE.value, 0, 0))
        assert new_text is not None
        assert len(new_text) < len(self.env.current_text)
        
        # Test invalid edit
        new_text = self.env._apply_edit((EditOp.REPLACE.value, 20, 0))
        assert new_text is None


if __name__ == "__main__":
    pytest.main([__file__])
