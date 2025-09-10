# Tic Tac Toe Q-Learning AI - Vercel Deployment

This is a Flask-based web application that implements a Tic Tac Toe game with a Q-learning AI agent, optimized for deployment on Vercel.

## Features

- **Interactive Tic Tac Toe Game**: Play against an AI trained with reinforcement learning
- **Q-Learning Training**: Train the AI with customizable parameters
- **Real-time Visualizations**: Charts showing training progress and Q-values
- **Professional UI**: Clean, data-focused interface
- **Vercel Compatible**: Optimized for serverless deployment

## Vercel Deployment

### Prerequisites

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Deployment Steps

1. **Login to Vercel**:
```bash
vercel login
```

2. **Deploy the application**:
```bash
vercel
```

3. **Follow the prompts**:
   - Set up and deploy? `Y`
   - Which scope? Choose your account
   - Link to existing project? `N`
   - What's your project's name? `tic-tac-toe-ai`
   - In which directory is your code located? `./`
   - Want to override the settings? `N`

4. **Set environment variables** (if needed):
```bash
vercel env add
```

### File Structure for Vercel

```
├── app_vercel.py          # Main Flask app (Vercel-compatible)
├── vercel.json            # Vercel configuration
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # HTML template
├── static/
│   ├── css/
│   │   └── style.css      # Styles
│   └── js/
│       └── game.js        # Frontend logic
└── README_VERCEL.md       # This file
```

### Key Vercel Optimizations

1. **Batch Training**: Training is done in batches instead of long-running threads
2. **Serverless Compatible**: No background processes or persistent state
3. **Memory Efficient**: Limited training episodes per request
4. **Fast Response**: Optimized for Vercel's execution limits

### Usage

1. **Play the Game**: Click on cells to make your moves
2. **Train the AI**: Use the training controls to improve the AI
3. **Monitor Progress**: View charts and statistics
4. **Customize Parameters**: Adjust learning rate, discount factor, etc.

### Technical Details

- **Framework**: Flask with Vercel Python runtime
- **AI Algorithm**: Q-learning with epsilon-greedy exploration
- **Frontend**: Vanilla JavaScript with Canvas API for charts
- **Styling**: Professional CSS with responsive design
- **Deployment**: Vercel serverless functions

### Limitations

- Training is limited to 5000 episodes per request (Vercel timeout limits)
- No persistent state between requests
- Q-values are reset on each deployment

### Development

To run locally:
```bash
python app_vercel.py
```

The app will be available at `http://localhost:5001`

### Troubleshooting

1. **Import Errors**: Ensure all dependencies are in `requirements.txt`
2. **Timeout Issues**: Reduce training episodes in the frontend
3. **Memory Issues**: Vercel has memory limits for serverless functions
4. **Chart Issues**: Check browser console for JavaScript errors

### Support

For issues with Vercel deployment, check:
- Vercel documentation
- Function logs in Vercel dashboard
- Browser console for frontend errors
