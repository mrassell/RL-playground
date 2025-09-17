"""
Edit-Agent Environment for GRPO Training

Implements a text editing environment where an agent learns to transform
text according to instructions using discrete edit operations.
"""

import numpy as np
import pandas as pd
import Levenshtein
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
import os
import random


class EditOp(Enum):
    INSERT = 0
    REPLACE = 1
    DELETE = 2
    STOP = 3


class EditAgentEnv:
    """
    Edit-Agent environment for learning text transformations.
    
    Action space:
    - op: Edit operation (INSERT, REPLACE, DELETE, STOP)
    - loc: Location index in the text
    - payload: Token/payload for the operation
    
    Reward structure:
    - Step reward: improvement in normalized edit distance
    - Final reward: +1 for exact match, 0 otherwise
    """
    
    def __init__(
        self,
        data_file: str = "data/edits.csv",
        max_steps: int = 20,
        payload_vocab_size: int = 100
    ):
        self.data_file = data_file
        self.max_steps = max_steps
        self.payload_vocab_size = payload_vocab_size
        
        # Load data
        self.data = self._load_data()
        self.payload_vocab = self._build_payload_vocab()
        
        # Current episode state
        self.current_task = None
        self.current_text = ""
        self.target_text = ""
        self.instruction = ""
        self.step_count = 0
        self.done = False
        self.previous_distance = 0.0
        
        # Statistics
        self.total_episodes = 0
        self.exact_matches = 0
        self.total_distance_improvement = 0.0
    
    def _load_data(self) -> pd.DataFrame:
        """Load edit data from CSV file."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file {self.data_file} not found")
        
        df = pd.read_csv(self.data_file)
        required_cols = ['id', 'before_text', 'instruction', 'after_text']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        return df
    
    def _build_payload_vocab(self) -> List[str]:
        """Build vocabulary of common tokens from the data."""
        vocab = set()
        
        # Extract tokens from all text fields
        for _, row in self.data.iterrows():
            # Split by common delimiters
            tokens = []
            for text in [row['before_text'], row['after_text']]:
                tokens.extend(text.split())
                tokens.extend(text.split('\n'))
                tokens.extend(text.split('\t'))
                tokens.extend(text.split(','))
                tokens.extend(text.split(':'))
                tokens.extend(text.split(';'))
            
            # Add individual characters
            for text in [row['before_text'], row['after_text']]:
                vocab.update(list(text))
            
            vocab.update(tokens)
        
        # Convert to list and limit size
        vocab_list = list(vocab)
        vocab_list = [token for token in vocab_list if len(token.strip()) > 0]
        
        # Sort by frequency (simple heuristic: length)
        vocab_list.sort(key=lambda x: len(x), reverse=True)
        
        # Limit to vocab_size
        vocab_list = vocab_list[:self.payload_vocab_size]
        
        # Add some common tokens if not present
        common_tokens = [' ', '\n', '\t', ',', ':', ';', '.', '!', '?', '(', ')', '[', ']', '{', '}']
        for token in common_tokens:
            if token not in vocab_list:
                vocab_list.append(token)
        
        return vocab_list[:self.payload_vocab_size]
    
    def reset(self, task_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset environment with a new task.
        
        Args:
            task_id: Specific task ID to use (random if None)
            
        Returns:
            Initial state dictionary
        """
        if task_id is None:
            task_id = random.choice(self.data['id'].tolist())
        
        task_row = self.data[self.data['id'] == task_id].iloc[0]
        
        self.current_task = task_id
        self.current_text = task_row['before_text']
        self.target_text = task_row['after_text']
        self.instruction = task_row['instruction']
        self.step_count = 0
        self.done = False
        
        # Calculate initial distance
        self.previous_distance = self._normalized_edit_distance(
            self.current_text, self.target_text
        )
        
        self.total_episodes += 1
        
        return self._get_state()
    
    def step(self, action: Tuple[int, int, int]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Tuple of (op, loc, payload_id)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.done:
            return self._get_state(), 0.0, True, {"error": "Episode already done"}
        
        op_id, loc, payload_id = action
        
        # Validate action
        if not self._is_valid_action(action):
            reward = -0.1  # Small penalty for invalid action
            info = {"error": "Invalid action", "action": action}
            return self._get_state(), reward, False, info
        
        # Apply action
        if op_id == EditOp.STOP.value:
            self.done = True
            reward = self._calculate_final_reward()
            info = {"action": "STOP", "final_distance": self.previous_distance}
            return self._get_state(), reward, True, info
        
        # Apply edit operation
        new_text = self._apply_edit(action)
        
        if new_text is None:
            reward = -0.1  # Penalty for failed edit
            info = {"error": "Edit failed", "action": action}
            return self._get_state(), reward, False, info
        
        # Calculate reward based on distance improvement
        old_distance = self.previous_distance
        new_distance = self._normalized_edit_distance(new_text, self.target_text)
        
        reward = old_distance - new_distance  # Positive if distance decreased
        
        # Update state
        self.current_text = new_text
        self.previous_distance = new_distance
        self.step_count += 1
        
        # Check if done (max steps reached)
        if self.step_count >= self.max_steps:
            self.done = True
            reward += self._calculate_final_reward()
        
        info = {
            "action": action,
            "step_count": self.step_count,
            "distance_improvement": old_distance - new_distance,
            "current_distance": new_distance
        }
        
        return self._get_state(), reward, self.done, info
    
    def _is_valid_action(self, action: Tuple[int, int, int]) -> bool:
        """Check if action is valid."""
        op_id, loc, payload_id = action
        
        # Check operation validity
        if op_id not in [op.value for op in EditOp]:
            return False
        
        # Check payload validity
        if payload_id < 0 or payload_id >= len(self.payload_vocab):
            return False
        
        # Check location validity
        if op_id == EditOp.STOP.value:
            return True  # STOP is always valid
        
        text_length = len(self.current_text)
        
        if op_id == EditOp.INSERT.value:
            return 0 <= loc <= text_length
        
        elif op_id == EditOp.REPLACE.value:
            return 0 <= loc < text_length
        
        elif op_id == EditOp.DELETE.value:
            return 0 <= loc < text_length
        
        return False
    
    def _apply_edit(self, action: Tuple[int, int, int]) -> Optional[str]:
        """Apply edit operation to current text."""
        op_id, loc, payload_id = action
        
        try:
            payload = self.payload_vocab[payload_id]
            text = self.current_text
            
            if op_id == EditOp.INSERT.value:
                return text[:loc] + payload + text[loc:]
            
            elif op_id == EditOp.REPLACE.value:
                if loc >= len(text):
                    return None
                return text[:loc] + payload + text[loc+1:]
            
            elif op_id == EditOp.DELETE.value:
                if loc >= len(text):
                    return None
                return text[:loc] + text[loc+1:]
            
            else:
                return None
        
        except Exception:
            return None
    
    def _normalized_edit_distance(self, text1: str, text2: str) -> float:
        """Calculate normalized Levenshtein distance."""
        if len(text1) == 0 and len(text2) == 0:
            return 0.0
        
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 0.0
        
        distance = Levenshtein.distance(text1, text2)
        return distance / max_len
    
    def _calculate_final_reward(self) -> float:
        """Calculate final reward based on exact match."""
        if self.current_text == self.target_text:
            self.exact_matches += 1
            return 1.0
        else:
            return 0.0
    
    def _get_state(self) -> Dict[str, Any]:
        """Get current state representation."""
        return {
            "current_text": self.current_text,
            "target_text": self.target_text,
            "instruction": self.instruction,
            "step_count": self.step_count,
            "done": self.done,
            "task_id": self.current_task,
            "text_length": len(self.current_text),
            "distance": self.previous_distance
        }
    
    def get_action_mask(self) -> np.ndarray:
        """Get action mask for valid actions."""
        # Action space: (op, loc, payload)
        # op: 4 values (INSERT, REPLACE, DELETE, STOP)
        # loc: max text_length + 1 (for INSERT)
        # payload: vocab_size values
        
        text_length = len(self.current_text)
        max_loc = max(text_length, 1)  # At least 1 for empty text
        
        # Create mask for all possible actions
        mask = np.zeros((4, max_loc, len(self.payload_vocab)), dtype=bool)
        
        # STOP is always valid
        mask[EditOp.STOP.value, :, :] = True
        
        # INSERT: valid at any position 0 to text_length
        for loc in range(min(text_length + 1, max_loc)):
            mask[EditOp.INSERT.value, loc, :] = True
        
        # REPLACE: valid at positions 0 to text_length-1
        for loc in range(min(text_length, max_loc)):
            mask[EditOp.REPLACE.value, loc, :] = True
        
        # DELETE: valid at positions 0 to text_length-1
        for loc in range(min(text_length, max_loc)):
            mask[EditOp.DELETE.value, loc, :] = True
        
        return mask
    
    def get_flat_action_mask(self) -> np.ndarray:
        """Get flattened action mask for easier use with policies."""
        mask = self.get_action_mask()
        return mask.flatten()
    
    def get_action_space_size(self) -> Tuple[int, int, int]:
        """Get action space dimensions."""
        text_length = len(self.current_text)
        max_loc = max(text_length, 1)
        return (4, max_loc, len(self.payload_vocab))
    
    def get_flat_action_space_size(self) -> int:
        """Get flattened action space size."""
        op_size, loc_size, payload_size = self.get_action_space_size()
        return op_size * loc_size * payload_size
    
    def flatten_action(self, action: Tuple[int, int, int]) -> int:
        """Flatten 3D action to 1D index."""
        op_id, loc, payload_id = action
        _, loc_size, payload_size = self.get_action_space_size()
        return op_id * (loc_size * payload_size) + loc * payload_size + payload_id
    
    def unflatten_action(self, flat_action: int) -> Tuple[int, int, int]:
        """Unflatten 1D action to 3D."""
        _, loc_size, payload_size = self.get_action_space_size()
        op_id = flat_action // (loc_size * payload_size)
        remaining = flat_action % (loc_size * payload_size)
        loc = remaining // payload_size
        payload_id = remaining % payload_size
        return (op_id, loc, payload_id)
    
    def render_state(self) -> str:
        """Render current state as string."""
        return f"""
Task ID: {self.current_task}
Instruction: {self.instruction}
Current Text: {repr(self.current_text)}
Target Text: {repr(self.target_text)}
Step: {self.step_count}/{self.max_steps}
Distance: {self.previous_distance:.3f}
Done: {self.done}
"""
    
    def get_evaluation_metrics(self, episodes: List) -> Dict[str, float]:
        """Get environment-specific evaluation metrics."""
        exact_matches = 0
        total_distance_improvement = 0.0
        total_episodes = len(episodes)
        
        for episode in episodes:
            if episode.metadata and "final_distance" in episode.metadata:
                final_distance = episode.metadata["final_distance"]
                if final_distance == 0.0:  # Exact match
                    exact_matches += 1
                
                # Calculate distance improvement (assuming we started with some distance)
                initial_distance = episode.metadata.get("initial_distance", 1.0)
                improvement = initial_distance - final_distance
                total_distance_improvement += improvement
        
        if total_episodes == 0:
            return {
                "exact_match_rate": 0.0,
                "mean_distance_improvement": 0.0,
                "mean_final_distance": 1.0
            }
        
        return {
            "exact_match_rate": exact_matches / total_episodes,
            "mean_distance_improvement": total_distance_improvement / total_episodes,
            "mean_final_distance": sum(ep.metadata.get("final_distance", 1.0) for ep in episodes) / total_episodes
        }
    
    def split_data(self, train_ratio: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        # Shuffle data
        shuffled_data = self.data.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        # Split
        split_idx = int(len(shuffled_data) * train_ratio)
        train_data = shuffled_data[:split_idx]
        test_data = shuffled_data[split_idx:]
        
        return train_data, test_data
    
    def save_split_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Save train/test split to files."""
        train_data.to_csv("data/edits.csv", index=False)
        test_data.to_csv("data/edits_heldout.csv", index=False)
    
    def load_heldout_data(self) -> pd.DataFrame:
        """Load held-out test data."""
        heldout_file = "data/edits_heldout.csv"
        if os.path.exists(heldout_file):
            return pd.read_csv(heldout_file)
        else:
            # Create split if it doesn't exist
            train_data, test_data = self.split_data()
            self.save_split_data(train_data, test_data)
            return test_data
