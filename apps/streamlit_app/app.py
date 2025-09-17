"""
Edit-Agent Playground - GRPO Post-Training Demo

Main Streamlit application with Tic-Tac-Toe and Edit-Agent tabs.
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
import logging
from typing import Dict, Any

# Set up intense logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting Edit-Agent Playground application...")
logger.info(f"Python path: {sys.path}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Script location: {__file__}")
logger.info(f"Project root: {project_root}")

try:
    logger.info("Importing components...")
    from apps.streamlit_app.components.ui_utils import create_sidebar_config
    logger.info("✅ ui_utils imported successfully")
    
    from apps.streamlit_app.ttt_tab import render_ttt_tab
    logger.info("✅ ttt_tab imported successfully")
    
    from apps.streamlit_app.edit_tab import render_edit_tab
    logger.info("✅ edit_tab imported successfully")
    
    logger.info("✅ All imports successful!")
    
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    st.error(f"Import error: {e}")
    st.stop()
except Exception as e:
    logger.error(f"❌ Unexpected error during imports: {e}")
    st.error(f"Unexpected error: {e}")
    st.stop()


def initialize_app():
    """Initialize the application."""
    # Page config
    st.set_page_config(
        page_title="Edit-Agent Playground",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .tab-container {
        margin-top: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)


def create_header():
    """Create application header."""
    st.markdown('<h1 class="main-header">🎮 Tic-Tac-Toe Bot</h1>', unsafe_allow_html=True)


def create_sidebar():
    """Create sidebar configuration."""
    st.sidebar.title("⚙️ Training Parameters")
    
    # Training parameters
    episodes = st.sidebar.slider("Episodes per Update", 10, 200, 64)
    top_episodes = st.sidebar.slider("Top Episodes to Learn From", 5, 50, 16)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
    updates = st.sidebar.slider("Training Updates", 1, 50, 10)
    
    config = {
        'num_episodes': episodes,
        'top_m': top_episodes,
        'learning_rate': learning_rate,
        'num_updates': updates,
        'seed': 1337
    }
    
    st.session_state.config = config
    
    # Reset button
    if st.sidebar.button("🔄 Reset Bot"):
        for key in list(st.session_state.keys()):
            if key.startswith('ttt_'):
                del st.session_state[key]
        st.sidebar.success("Bot reset!")


def create_tabs():
    """Create main application tabs."""
    render_ttt_tab()


def create_footer():
    """Create application footer."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p><strong>Edit-Agent Playground</strong> - GRPO Post-Training Demo</p>
        <p>Built with Streamlit, PyTorch, and ❤️</p>
        <p>Check out the <a href="https://github.com/your-repo/edit-agent-playground">GitHub repository</a> for source code and documentation.</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application function."""
    try:
        logger.info("Initializing application...")
        # Initialize app
        initialize_app()
        logger.info("✅ App initialized")
        
        # Create header
        logger.info("Creating header...")
        create_header()
        logger.info("✅ Header created")
        
        # Create sidebar
        logger.info("Creating sidebar...")
        create_sidebar()
        logger.info("✅ Sidebar created")
        
        # Create main content tabs
        logger.info("Creating tabs...")
        create_tabs()
        logger.info("✅ Tabs created")
        
        # Create footer
        logger.info("Creating footer...")
        create_footer()
        logger.info("✅ Footer created")
        
        # Add some helpful information
        logger.info("Adding technical details...")
        with st.expander("🔧 Technical Details"):
            st.markdown("""
            **Architecture:**
            - **GRPO Core**: Group relative policy optimization with episode ranking
            - **Tic-Tac-Toe**: 3x3 board with configurable reward shaping
            - **Edit-Agent**: Text transformation with discrete edit operations
            - **Policies**: Small MLPs with action masking
            
            **Training Process:**
            1. Collect K episodes using current policy
            2. Score episodes by total return
            3. Rank episodes and select top-m as preferred
            4. Compute advantages from rankings
            5. Update policy using advantage-weighted loss
            
            **Evaluation:**
            - Tic-Tac-Toe: Win rate vs random/greedy opponents
            - Edit-Agent: Exact match rate and edit distance improvement
            - Real-time metrics and saved plots
            """)
        logger.info("✅ Technical details added")
        
        logger.info("🎉 Application loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in main function: {e}")
        st.error(f"Application error: {e}")
        st.exception(e)


if __name__ == "__main__":
    logger.info("🚀 Starting Edit-Agent Playground...")
    main()
