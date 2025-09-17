"""
Experience buffers for GRPO training.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque


class ExperienceBuffer:
    """Simple experience buffer for storing episodes."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add_episode(self, episode: Dict[str, Any]):
        """Add an episode to the buffer."""
        self.buffer.append(episode)
    
    def sample_episodes(self, num_episodes: int) -> List[Dict[str, Any]]:
        """Sample random episodes from the buffer."""
        if len(self.buffer) == 0:
            return []
        
        indices = np.random.choice(len(self.buffer), size=min(num_episodes, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def get_all_episodes(self) -> List[Dict[str, Any]]:
        """Get all episodes in the buffer."""
        return list(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)


class PrioritizedBuffer:
    """Prioritized experience buffer based on episode returns."""
    
    def __init__(self, max_size: int = 10000, alpha: float = 0.6):
        self.max_size = max_size
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
    
    def add_episode(self, episode: Dict[str, Any], priority: Optional[float] = None):
        """Add an episode with priority."""
        if priority is None:
            priority = abs(episode.get('total_return', 0.0))
        
        self.buffer.append(episode)
        self.priorities.append(priority ** self.alpha)
        
        if len(self.buffer) > self.max_size:
            # Remove lowest priority episode
            min_idx = np.argmin(self.priorities)
            del self.buffer[min_idx]
            del self.priorities[min_idx]
    
    def sample_episodes(self, num_episodes: int) -> List[Dict[str, Any]]:
        """Sample episodes based on priority."""
        if len(self.buffer) == 0:
            return []
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / np.sum(priorities)
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), size=min(num_episodes, len(self.buffer)), 
                                 replace=False, p=probabilities)
        
        return [self.buffer[i] for i in indices]
    
    def get_all_episodes(self) -> List[Dict[str, Any]]:
        """Get all episodes in the buffer."""
        return self.buffer.copy()
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.priorities.clear()
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
