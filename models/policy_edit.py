"""
Edit-Agent Policy Network

Policy network for text editing with discrete actions (op, loc, payload).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, Dict, Any
import hashlib


class EditAgentPolicy(nn.Module):
    """
    Policy network for edit-agent environment.
    
    Input: State features (instruction + current text features)
    Output: Joint logits over (op, loc, payload) actions
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_text_length: int = 100,
        payload_vocab_size: int = 100
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_text_length = max_text_length
        self.payload_vocab_size = payload_vocab_size
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Output heads for each action component
        self.op_head = nn.Linear(hidden_size, 4)  # INSERT, REPLACE, DELETE, STOP
        self.loc_head = nn.Linear(hidden_size, max_text_length)
        self.payload_head = nn.Linear(hidden_size, payload_vocab_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def encode_state(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Encode state into feature vector.
        
        Args:
            state: State dictionary from environment
            
        Returns:
            Encoded state features
        """
        features = []
        
        # Text length feature
        text_length = state.get('text_length', 0)
        features.append(text_length / self.max_text_length)  # Normalize
        
        # Distance feature
        distance = state.get('distance', 1.0)
        features.append(distance)
        
        # Step count feature
        step_count = state.get('step_count', 0)
        features.append(step_count / 20.0)  # Normalize by max steps
        
        # Text features (bag of characters)
        current_text = state.get('current_text', '')
        target_text = state.get('target_text', '')
        
        # Character frequency features
        char_features = self._get_char_features(current_text, target_text)
        features.extend(char_features)
        
        # Instruction features (simple hash-based)
        instruction = state.get('instruction', '')
        inst_features = self._get_instruction_features(instruction)
        features.extend(inst_features)
        
        # Pad or truncate to state_dim
        while len(features) < self.state_dim:
            features.append(0.0)
        
        features = features[:self.state_dim]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _get_char_features(self, current_text: str, target_text: str) -> list:
        """Extract character-based features."""
        features = []
        
        # Character frequency differences
        current_chars = {}
        target_chars = {}
        
        for char in current_text:
            current_chars[char] = current_chars.get(char, 0) + 1
        
        for char in target_text:
            target_chars[char] = target_chars.get(char, 0) + 1
        
        # Common characters
        common_chars = set(current_chars.keys()) & set(target_chars.keys())
        features.append(len(common_chars) / max(len(current_chars), len(target_chars), 1))
        
        # Length difference
        length_diff = abs(len(current_text) - len(target_text))
        features.append(length_diff / max(len(current_text), len(target_text), 1))
        
        # Common prefixes/suffixes
        prefix_len = 0
        for i in range(min(len(current_text), len(target_text))):
            if current_text[i] == target_text[i]:
                prefix_len += 1
            else:
                break
        
        suffix_len = 0
        for i in range(min(len(current_text), len(target_text))):
            if current_text[-(i+1)] == target_text[-(i+1)]:
                suffix_len += 1
            else:
                break
        
        features.append(prefix_len / max(len(current_text), len(target_text), 1))
        features.append(suffix_len / max(len(current_text), len(target_text), 1))
        
        return features
    
    def _get_instruction_features(self, instruction: str) -> list:
        """Extract instruction-based features."""
        features = []
        
        # Instruction length
        features.append(len(instruction) / 100.0)  # Normalize
        
        # Keyword features
        keywords = ['replace', 'add', 'remove', 'delete', 'insert', 'change', 'capitalize', 'format']
        keyword_counts = [instruction.lower().count(keyword) for keyword in keywords]
        features.extend([count / len(instruction) if len(instruction) > 0 else 0 for count in keyword_counts])
        
        # Hash-based features for instruction similarity
        inst_hash = hashlib.md5(instruction.encode()).hexdigest()
        hash_features = [int(inst_hash[i:i+2], 16) / 255.0 for i in range(0, min(32, len(inst_hash)), 2)]
        features.extend(hash_features[:8])  # Take first 8 hash features
        
        return features
    
    def forward(self, state: Union[Dict[str, Any], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: State dictionary or encoded state tensor
            
        Returns:
            Tuple of (op_logits, loc_logits, payload_logits)
        """
        if isinstance(state, dict):
            state_features = self.encode_state(state)
        else:
            state_features = state
        
        # Ensure state is 2D (batch_size, state_dim)
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)
        
        # Forward pass
        x = self.state_encoder(state_features)
        
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        
        # Get logits for each action component
        op_logits = self.op_head(x)
        loc_logits = self.loc_head(x)
        payload_logits = self.payload_head(x)
        
        # Remove batch dimension if input was 1D
        if state_features.dim() == 1:
            op_logits = op_logits.squeeze(0)
            loc_logits = loc_logits.squeeze(0)
            payload_logits = payload_logits.squeeze(0)
        
        return op_logits, loc_logits, payload_logits
    
    def get_action_mask(self, state: Union[Dict[str, Any], torch.Tensor]) -> np.ndarray:
        """
        Get action mask for valid actions.
        
        Args:
            state: State dictionary or encoded state tensor
            
        Returns:
            Boolean mask for valid actions
        """
        if isinstance(state, dict):
            text_length = state.get('text_length', 0)
        else:
            # Assume state is encoded and we need to infer text length
            text_length = int(state[0].item() * self.max_text_length)
        
        # Create mask for all possible actions
        mask = np.zeros((4, self.max_text_length, self.payload_vocab_size), dtype=bool)
        
        # STOP is always valid
        mask[3, :, :] = True  # STOP (op=3)
        
        # INSERT: valid at any position 0 to text_length
        for loc in range(min(text_length + 1, self.max_text_length)):
            mask[0, loc, :] = True  # INSERT (op=0)
        
        # REPLACE: valid at positions 0 to text_length-1
        for loc in range(min(text_length, self.max_text_length)):
            mask[1, loc, :] = True  # REPLACE (op=1)
        
        # DELETE: valid at positions 0 to text_length-1
        for loc in range(min(text_length, self.max_text_length)):
            mask[2, loc, :] = True  # DELETE (op=2)
        
        return mask
    
    def get_flat_action_mask(self, state: Union[Dict[str, Any], torch.Tensor]) -> np.ndarray:
        """Get flattened action mask."""
        mask = self.get_action_mask(state)
        return mask.flatten()
    
    def act(
        self,
        state: Union[Dict[str, Any], torch.Tensor],
        action_mask: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Tuple[int, int, int], float]:
        """
        Sample action from policy.
        
        Args:
            state: State dictionary or encoded state tensor
            action_mask: Boolean mask for valid actions
            
        Returns:
            Tuple of ((op, loc, payload), log_prob)
        """
        # Get logits
        op_logits, loc_logits, payload_logits = self.forward(state)
        
        # Apply action mask
        if isinstance(action_mask, np.ndarray):
            action_mask = torch.tensor(action_mask, dtype=torch.bool, device=op_logits.device)
        
        # Reshape mask to match logits
        mask_reshaped = action_mask.view(4, self.max_text_length, self.payload_vocab_size)
        
        # Sample each component
        op_probs = F.softmax(op_logits, dim=-1)
        op = torch.multinomial(op_probs, 1).item()
        
        # Sample location
        loc_probs = F.softmax(loc_logits, dim=-1)
        loc = torch.multinomial(loc_probs, 1).item()
        
        # Sample payload
        payload_probs = F.softmax(payload_logits, dim=-1)
        payload = torch.multinomial(payload_probs, 1).item()
        
        # Check if action is valid
        if not mask_reshaped[op, loc, payload]:
            # Find a valid action
            valid_actions = torch.nonzero(mask_reshaped, as_tuple=False)
            if len(valid_actions) > 0:
                idx = torch.randint(0, len(valid_actions), (1,)).item()
                op, loc, payload = valid_actions[idx].tolist()
            else:
                # Fallback to STOP
                op, loc, payload = 3, 0, 0
        
        # Calculate log probability
        log_prob = (
            F.log_softmax(op_logits, dim=-1)[op] +
            F.log_softmax(loc_logits, dim=-1)[loc] +
            F.log_softmax(payload_logits, dim=-1)[payload]
        ).item()
        
        return (op, loc, payload), log_prob
    
    def log_prob(
        self,
        state: Union[Dict[str, Any], torch.Tensor],
        action: Tuple[int, int, int],
        action_mask: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate log probability of action.
        
        Args:
            state: State dictionary or encoded state tensor
            action: Action taken (op, loc, payload)
            action_mask: Boolean mask for valid actions
            
        Returns:
            Log probability of the action
        """
        op, loc, payload = action
        
        # Get logits
        op_logits, loc_logits, payload_logits = self.forward(state)
        
        # Calculate log probability
        log_prob = (
            F.log_softmax(op_logits, dim=-1)[op] +
            F.log_softmax(loc_logits, dim=-1)[loc] +
            F.log_softmax(payload_logits, dim=-1)[payload]
        ).item()
        
        return log_prob
    
    def save(self, filepath: str):
        """Save policy to file."""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath: str):
        """Load policy from file."""
        self.load_state_dict(torch.load(filepath, map_location=next(self.parameters()).device))


def create_edit_policy(
    state_dim: int = 128,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    max_text_length: int = 100,
    payload_vocab_size: int = 100
) -> EditAgentPolicy:
    """Factory function to create an edit-agent policy."""
    return EditAgentPolicy(state_dim, hidden_size, num_layers, dropout, max_text_length, payload_vocab_size)
