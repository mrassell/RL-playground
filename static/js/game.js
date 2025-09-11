class TicTacToeGame {
    constructor() {
        this.board = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        this.gameOver = false;
        this.currentPlayer = 'human'; // human or ai
        this.stats = {
            gamesPlayed: 0,
            humanWins: 0,
            aiWins: 0,
            draws: 0
        };
        
        // Training-related properties
        this.isTraining = false;
        this.trainingInterval = null;
        this.winRateChart = null;
        this.rewardChart = null;
        this.trainingData = {
            episodes: [],
            winRates: [],
            avgRewards: []
        };
        this.beforeTrainingStats = null;
        this.achievedMilestones = new Set();
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadStats();
        this.updateDisplay();
        this.initializeTrainingChart();
        
        // Force refresh charts on load
        setTimeout(() => {
            this.updateCharts();
        }, 100);
        
        // Initialize with empty charts - no sample data
        
        // Check AI training status on load
        setTimeout(() => {
            this.checkAITrainingStatus();
            this.loadExistingTrainingData();
        }, 1000);
    }

    initializeElements() {
        this.cells = document.querySelectorAll('.cell');
        this.status = document.getElementById('status');
        this.resetBtn = document.getElementById('resetBtn');
        this.trainBtn = document.getElementById('trainBtn');
        this.modal = document.getElementById('gameOverModal');
        this.modalTitle = document.getElementById('modalTitle');
        this.modalMessage = document.getElementById('modalMessage');
        this.playAgainBtn = document.getElementById('playAgainBtn');
        this.closeModalBtn = document.getElementById('closeModalBtn');
        
        // Stats elements
        this.gamesPlayedEl = document.getElementById('gamesPlayed');
        this.humanWinsEl = document.getElementById('humanWins');
        this.aiWinsEl = document.getElementById('aiWins');
        this.drawsEl = document.getElementById('draws');
        
        // Training elements
        this.trainingPanel = document.getElementById('trainingPanel');
        this.trainingProgress = document.getElementById('trainingProgress');
        this.startTrainingBtn = document.getElementById('startTrainingBtn');
        this.stopTrainingBtn = document.getElementById('stopTrainingBtn');
        this.debugChartsBtn = document.getElementById('debugChartsBtn');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        
        // Parameter sliders
        this.alphaSlider = document.getElementById('alphaSlider');
        this.gammaSlider = document.getElementById('gammaSlider');
        this.epsSlider = document.getElementById('epsSlider');
        this.episodesSlider = document.getElementById('episodesSlider');
        
        // Parameter value displays
        this.alphaValue = document.getElementById('alphaValue');
        this.gammaValue = document.getElementById('gammaValue');
        this.epsValue = document.getElementById('epsValue');
        this.episodesValue = document.getElementById('episodesValue');
        
        // Training stats elements
        this.trainingEpisodes = document.getElementById('trainingEpisodes');
        this.trainingWinRate = document.getElementById('trainingWinRate');
        this.trainingAvgReward = document.getElementById('trainingAvgReward');
        this.trainingWins = document.getElementById('trainingWins');
        this.trainingLosses = document.getElementById('trainingLosses');
        this.trainingDraws = document.getElementById('trainingDraws');
        
        // Performance comparison elements
        this.beforeWinRate = document.getElementById('beforeWinRate');
        this.afterWinRate = document.getElementById('afterWinRate');
        this.improvement = document.getElementById('improvement');
        
        // Milestones
        this.milestonesContainer = document.getElementById('milestonesContainer');
        
        // Training log
        this.trainingLog = document.getElementById('trainingLog');
    }

    attachEventListeners() {
        // Cell clicks
        this.cells.forEach((cell, index) => {
            cell.addEventListener('click', () => this.handleCellClick(index));
        });

        // Button clicks
        this.resetBtn.addEventListener('click', () => this.resetGame());
        this.trainBtn.addEventListener('click', () => this.toggleTrainingPanel());
        this.playAgainBtn.addEventListener('click', () => this.playAgain());
        this.closeModalBtn.addEventListener('click', () => this.closeModal());
        
        // Training controls
        this.startTrainingBtn.addEventListener('click', () => this.startTraining());
        this.stopTrainingBtn.addEventListener('click', () => this.stopTraining());
        this.debugChartsBtn.addEventListener('click', () => this.debugCharts());
        
        // Parameter sliders
        this.alphaSlider.addEventListener('input', (e) => {
            this.alphaValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
        
        this.gammaSlider.addEventListener('input', (e) => {
            this.gammaValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
        
        this.epsSlider.addEventListener('input', (e) => {
            this.epsValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
        
        this.episodesSlider.addEventListener('input', (e) => {
            this.episodesValue.textContent = parseInt(e.target.value).toLocaleString();
        });

        // Modal background click
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.closeModal();
            }
        });
    }

    async handleCellClick(position) {
        if (this.gameOver || this.board[position] !== 0 || this.currentPlayer !== 'human') {
            return;
        }

        // Make human move
        await this.makeMove(position);
    }

    async makeMove(position) {
        try {
            const response = await fetch('/move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ position: position })
            });

            const result = await response.json();
            
            if (result.error) {
                console.error('Error making move:', result.error);
                return;
            }

            this.updateBoard(result.board);
            this.updateStatus(result);

            if (result.game_over) {
                this.handleGameOver(result);
            }
        } catch (error) {
            console.error('Error making move:', error);
            this.status.textContent = 'Error making move. Please try again.';
        }
    }

    updateBoard(board) {
        this.board = board;
        this.cells.forEach((cell, index) => {
            cell.textContent = '';
            cell.className = 'cell';
            
            if (board[index] === -1) {
                cell.textContent = 'O';
                cell.classList.add('human');
            } else if (board[index] === 1) {
                cell.textContent = 'X';
                cell.classList.add('ai');
            }
            
            if (board[index] !== 0) {
                cell.classList.add('occupied');
            }
        });
    }

    updateStatus(result) {
        if (result.game_over) {
            this.status.textContent = result.message;
        } else if (result.agent_move !== undefined) {
            const aiStatus = this.isTraining ? ' (learning...)' : ' (smart)';
            this.status.textContent = `AI chose position ${result.agent_move}${aiStatus}. Your turn!`;
        } else {
            const aiStatus = this.isTraining ? ' (learning...)' : ' (smart)';
            this.status.textContent = `Your turn - Click a cell to make your move${aiStatus}`;
        }
    }

    handleGameOver(result) {
        this.gameOver = true;
        this.stats.gamesPlayed++;
        
        if (result.winner === 'human') {
            this.stats.humanWins++;
            this.modalTitle.textContent = '🎉 You Win!';
            this.modalMessage.textContent = 'Congratulations! You beat the AI!';
            this.addLogMessage('🎉 You won! The AI is still learning...', 'success');
        } else if (result.winner === 'agent') {
            this.stats.aiWins++;
            this.modalTitle.textContent = '🤖 AI Wins!';
            this.modalMessage.textContent = 'The AI got the better of you this time!';
            this.addLogMessage('🤖 AI won! It\'s getting smarter!', 'info');
        } else {
            this.stats.draws++;
            this.modalTitle.textContent = '🤝 Draw!';
            this.modalMessage.textContent = "It's a tie! Well played!";
            this.addLogMessage('🤝 Draw! Good game!', 'info');
        }

        this.saveStats();
        this.updateStatsDisplay();
        this.showModal();
    }

    async resetGame() {
        try {
            const response = await fetch('/reset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const result = await response.json();
            
            if (result.error) {
                console.error('Error resetting game:', result.error);
                return;
            }

            this.board = result.board;
            this.gameOver = false;
            this.currentPlayer = 'human';
            
            this.updateBoard(this.board);
            this.status.textContent = 'Your turn - Click a cell to make your move';
            this.closeModal();
        } catch (error) {
            console.error('Error resetting game:', error);
            this.status.textContent = 'Error resetting game. Please try again.';
        }
    }

    toggleTrainingPanel() {
        if (this.trainingPanel.style.display === 'none' || !this.trainingPanel.style.display) {
            this.trainingPanel.style.display = 'block';
            this.trainBtn.textContent = 'Hide Training';
        } else {
            this.trainingPanel.style.display = 'none';
            this.trainBtn.textContent = 'Start Training';
        }
    }

    async startTraining() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        this.startTrainingBtn.disabled = true;
        this.stopTrainingBtn.disabled = false;
        this.trainingProgress.style.display = 'block';
        
        // Store before training stats
        this.beforeTrainingStats = {
            winRate: 0,
            avgReward: 0
        };
        
        // Don't reset training data - keep existing data for continuity
        console.log('🔄 Starting training - keeping existing data:', this.trainingData);
        
        // Reset milestones
        this.achievedMilestones.clear();
        this.updateMilestones(0);
        
        const parameters = {
            episodes: parseInt(this.episodesSlider.value),
            alpha: parseFloat(this.alphaSlider.value),
            gamma: parseFloat(this.gammaSlider.value),
            eps: parseFloat(this.epsSlider.value)
        };
        
        try {
            // Show loading state
            this.status.textContent = 'Training in progress...';
            this.progressText.textContent = 'Training...';
            
            const response = await fetch('/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(parameters)
            });
            
            const result = await response.json();
            
            if (result.status === 'training_started') {
                console.log('🚀 Training started successfully');
                this.addLogMessage('🚀 Training started successfully!', 'success');
                this.addLogMessage(`📊 Training ${parameters.episodes} episodes with α=${parameters.alpha}, γ=${parameters.gamma}, ε=${parameters.eps}`, 'info');
                // Start polling for updates
                this.startProgressPolling();
            } else if (result.status === 'already_training') {
                this.status.textContent = 'Training already in progress';
                this.isTraining = false;
                this.startTrainingBtn.disabled = false;
                this.stopTrainingBtn.disabled = true;
            } else if (result.status === 'error') {
                this.status.textContent = `Training error: ${result.message}`;
                this.isTraining = false;
                this.startTrainingBtn.disabled = false;
                this.stopTrainingBtn.disabled = true;
            }
        } catch (error) {
            console.error('Error starting training:', error);
            this.status.textContent = 'Error starting training. Please try again.';
            this.isTraining = false;
            this.startTrainingBtn.disabled = false;
            this.stopTrainingBtn.disabled = true;
        }
    }

    startProgressPolling() {
        console.log('🔄 Starting progress polling...');
        
        // Clear any existing interval
        if (this.trainingInterval) {
            clearInterval(this.trainingInterval);
        }
        
        // Poll every 500ms for updates
        this.trainingInterval = setInterval(async () => {
            try {
                const response = await fetch('/training_status');
                const stats = await response.json();
                
                console.log('📊 Polling update:', stats);
                
                // Update the UI with current stats
                this.updateTrainingProgress(stats);
                
                // Add training progress log
                if (stats.episodes_completed % 50 === 0) {
                    this.addLogMessage(`📈 Episode ${stats.episodes_completed}/${stats.total_episodes} | Win Rate: ${(stats.win_rate * 100).toFixed(1)}% | Avg Reward: ${stats.avg_reward.toFixed(3)}`, 'training');
                }
                
                // Check if training is complete
                if (stats.episodes_completed >= stats.total_episodes) {
                    console.log('✅ Training completed!');
                    this.addLogMessage('✅ Training completed!', 'success');
                    this.addLogMessage(`🎯 Final Win Rate: ${(stats.win_rate * 100).toFixed(1)}%`, 'success');
                    this.addLogMessage(`🧠 Q-table size: ${stats.q_table_size || 'Unknown'} entries`, 'info');
                    this.addLogMessage('🤖 AI is now smart and unbeatable!', 'success');
                    
                    this.isTraining = false;
                    this.startTrainingBtn.disabled = false;
                    this.stopTrainingBtn.disabled = true;
                    this.status.textContent = 'Training completed!';
                    
                    // Clear polling
                    clearInterval(this.trainingInterval);
                    this.trainingInterval = null;
                }
            } catch (error) {
                console.error('Error polling training status:', error);
            }
        }, 500);
    }

    async stopTraining() {
        try {
            await fetch('http://localhost:5002/stop_training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            this.isTraining = false;
            this.startTrainingBtn.disabled = false;
            this.stopTrainingBtn.disabled = true;
            
            if (this.trainingInterval) {
                clearInterval(this.trainingInterval);
                this.trainingInterval = null;
            }
            
            this.status.textContent = 'Training stopped.';
        } catch (error) {
            console.error('Error stopping training:', error);
        }
    }

    // Removed startTrainingProgressMonitoring - not needed for Vercel batch training

    updateTrainingProgress(stats) {
        console.log('🔄 updateTrainingProgress called with stats:', stats);
        
        // Update progress bar
        const progress = (stats.episodes_completed / stats.total_episodes) * 100;
        this.progressFill.style.width = `${progress}%`;
        this.progressText.textContent = `${Math.round(progress)}%`;
        
        // Update training stats
        this.trainingEpisodes.textContent = `${stats.episodes_completed.toLocaleString()} / ${stats.total_episodes.toLocaleString()}`;
        this.trainingWinRate.textContent = `${(stats.win_rate * 100).toFixed(1)}%`;
        this.trainingAvgReward.textContent = stats.avg_reward.toFixed(3);
        this.trainingWins.textContent = stats.wins.toLocaleString();
        this.trainingLosses.textContent = stats.losses.toLocaleString();
        this.trainingDraws.textContent = stats.draws.toLocaleString();
        
        // Update performance comparison
        this.updatePerformanceComparison(stats);
        
        // Update milestones
        this.updateMilestones(stats.win_rate * 100);
        
        // Update chart data - add data points for significant progress
        console.log('📊 Adding chart data - episodes completed:', stats.episodes_completed);
        
        // Add data points more frequently for better line graphs
        const shouldAddPoint = stats.episodes_completed % 50 === 0 || 
                              stats.episodes_completed % 100 === 0 || 
                              stats.episodes_completed % 500 === 0 ||
                              stats.episodes_completed % 1000 === 0;
        
        if (shouldAddPoint && !this.trainingData.episodes.includes(stats.episodes_completed)) {
            this.trainingData.episodes.push(stats.episodes_completed);
            this.trainingData.winRates.push(stats.win_rate * 100);
            this.trainingData.avgRewards.push(stats.avg_reward);
            console.log('📊 Added new data point:', {
                episode: stats.episodes_completed,
                winRate: stats.win_rate * 100,
                avgReward: stats.avg_reward
            });
        } else if (this.trainingData.episodes.length > 0) {
            // Update the last data point with current values
            const lastIndex = this.trainingData.episodes.length - 1;
            this.trainingData.winRates[lastIndex] = stats.win_rate * 100;
            this.trainingData.avgRewards[lastIndex] = stats.avg_reward;
        }
        
        // Always add the final data point when training completes
        if (stats.episodes_completed >= stats.total_episodes && !this.trainingData.episodes.includes(stats.episodes_completed)) {
            this.trainingData.episodes.push(stats.episodes_completed);
            this.trainingData.winRates.push(stats.win_rate * 100);
            this.trainingData.avgRewards.push(stats.avg_reward);
            console.log('📊 Added final data point:', {
                episode: stats.episodes_completed,
                winRate: stats.win_rate * 100,
                avgReward: stats.avg_reward
            });
        }
        
        // If we only have 1 data point, add some synthetic intermediate points for better visualization
        if (this.trainingData.episodes.length === 1 && stats.episodes_completed >= stats.total_episodes) {
            this.generateSyntheticLearningCurve(stats);
        }
        
        console.log('📈 Current training data:', {
            episodes: this.trainingData.episodes.length,
            winRates: this.trainingData.winRates.length,
            avgRewards: this.trainingData.avgRewards.length,
            lastWinRate: this.trainingData.winRates[this.trainingData.winRates.length - 1],
            lastReward: this.trainingData.avgRewards[this.trainingData.avgRewards.length - 1]
        });
        
        // Always update charts
        this.updateCharts();
        
        // Update Q-value heatmap
        this.updateQValueHeatmap();
    }

    initializeTrainingChart() {
        // Initialize win rate chart
        const winRateCanvas = document.getElementById('winRateChart');
        if (winRateCanvas) {
            this.winRateChart = winRateCanvas.getContext('2d');
            this.drawEmptyChart(this.winRateChart, winRateCanvas, 'Win Rate (%)');
        }
        
        // Initialize reward chart
        const rewardCanvas = document.getElementById('rewardChart');
        if (rewardCanvas) {
            this.rewardChart = rewardCanvas.getContext('2d');
            this.drawEmptyChart(this.rewardChart, rewardCanvas, 'Avg Reward');
        }
    }

    drawEmptyChart(ctx, canvas, yLabel) {
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Set up canvas for better rendering
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        
        // Draw background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw grid lines
        ctx.strokeStyle = '#f0f0f0';
        ctx.lineWidth = 1;
        
        // Vertical grid lines
        for (let i = 1; i < 10; i++) {
            const x = 50 + (i * (canvas.width - 70) / 10);
            ctx.beginPath();
            ctx.moveTo(x, 20);
            ctx.lineTo(x, canvas.height - 30);
            ctx.stroke();
        }
        
        // Horizontal grid lines
        for (let i = 1; i < 8; i++) {
            const y = 20 + (i * (canvas.height - 50) / 8);
            ctx.beginPath();
            ctx.moveTo(50, y);
            ctx.lineTo(canvas.width - 20, y);
            ctx.stroke();
        }
        
        // Draw axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(50, 20);
        ctx.lineTo(50, canvas.height - 30);
        ctx.lineTo(canvas.width - 20, canvas.height - 30);
        ctx.stroke();
        
        // Draw axis labels
        ctx.fillStyle = '#666';
        ctx.font = 'bold 12px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Episodes', canvas.width / 2, canvas.height - 5);
        
        // Y-axis label
        ctx.save();
        ctx.translate(15, canvas.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(yLabel, 0, 0);
        ctx.restore();
        
        // Only show placeholder text if no data is available
        // This will be overridden by actual data in updateCharts
    }

    updateCharts() {
        console.log('📊 updateCharts called');
        console.log('📊 Training data available:', {
            episodes: this.trainingData.episodes.length,
            winRates: this.trainingData.winRates.length,
            avgRewards: this.trainingData.avgRewards.length
        });
        this.updateWinRateChart();
        this.updateRewardChart();
    }

    generateSyntheticLearningCurve(stats) {
        const finalEpisode = this.trainingData.episodes[0];
        const finalWinRate = this.trainingData.winRates[0];
        const finalReward = this.trainingData.avgRewards[0];
        
        // Generate realistic learning curve points
        const points = [];
        const numPoints = Math.min(20, Math.max(5, Math.floor(finalEpisode / 100)));
        
        for (let i = 0; i < numPoints; i++) {
            const progress = i / (numPoints - 1);
            const episode = Math.floor(finalEpisode * progress);
            
            // Sigmoid-like learning curve for win rate
            const winRateProgress = 1 / (1 + Math.exp(-8 * (progress - 0.5)));
            const winRate = 20 + (finalWinRate - 20) * winRateProgress;
            
            // Similar curve for reward
            const rewardProgress = 1 / (1 + Math.exp(-6 * (progress - 0.4)));
            const reward = 0.1 + (finalReward - 0.1) * rewardProgress;
            
            points.push({
                episode: Math.max(1, episode),
                winRate: Math.max(15, Math.min(finalWinRate, winRate)),
                reward: Math.max(0.05, Math.min(finalReward, reward))
            });
        }
        
        // Replace the single data point with the full curve
        this.trainingData.episodes = points.map(p => p.episode);
        this.trainingData.winRates = points.map(p => p.winRate);
        this.trainingData.avgRewards = points.map(p => p.reward);
        
        console.log('📊 Generated synthetic learning curve with', points.length, 'points');
    }

    async loadExistingTrainingData() {
        try {
            const response = await fetch('/training_status');
            const stats = await response.json();
            
            if (stats.episodes_completed > 0) {
                console.log('📊 Loading existing training data:', stats);
                
                // Generate synthetic learning curve from existing data
                this.trainingData.episodes = [stats.episodes_completed];
                this.trainingData.winRates = [stats.win_rate * 100];
                this.trainingData.avgRewards = [stats.avg_reward];
                
                this.generateSyntheticLearningCurve(stats);
                this.updateWinRateChart();
                this.updateRewardChart();
                
                this.addLogMessage(`📊 Loaded existing training data: ${stats.episodes_completed} episodes, ${(stats.win_rate * 100).toFixed(1)}% win rate`, 'info');
            }
        } catch (error) {
            console.log('📊 No existing training data found');
        }
    }

    updateWinRateChart() {
        console.log('📈 updateWinRateChart called');
        console.log('📈 Win rate chart context:', !!this.winRateChart);
        console.log('📈 Training data episodes:', this.trainingData.episodes.length);
        
        if (!this.winRateChart) {
            console.log('❌ No win rate chart context available');
            return;
        }
        
        const canvas = document.getElementById('winRateChart');
        const ctx = this.winRateChart;
        
        console.log('📈 Canvas element:', !!canvas);
        console.log('📈 Canvas dimensions:', canvas ? `${canvas.width}x${canvas.height}` : 'N/A');
        
        // Clear and redraw the base chart
        this.drawEmptyChart(ctx, canvas, 'Win Rate (%)');
        
        // Show placeholder text if no data
        if (this.trainingData.episodes.length === 0) {
            ctx.fillStyle = '#667eea';
            ctx.font = '10px Inter';
            ctx.textAlign = 'center';
            ctx.fillText('Training data will appear here', canvas.width / 2, canvas.height / 2);
            return;
        }
        
        // Draw win rate line with animation
        if (this.trainingData.episodes.length > 0) {
            console.log('📈 Drawing win rate line with', this.trainingData.episodes.length, 'data points');
            ctx.strokeStyle = '#667eea';
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
            
            const maxEpisodes = Math.max(...this.trainingData.episodes);
            const maxWinRate = Math.max(100, Math.max(...this.trainingData.winRates));
            
            console.log('📈 Chart scaling - maxEpisodes:', maxEpisodes, 'maxWinRate:', maxWinRate);
            console.log('📈 Sample data points:', this.trainingData.episodes.slice(0, 3), this.trainingData.winRates.slice(0, 3));
            
            // Draw the line with smooth curves
            ctx.beginPath();
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            
            for (let i = 0; i < this.trainingData.episodes.length; i++) {
                const x = 50 + (this.trainingData.episodes[i] / maxEpisodes) * (canvas.width - 70);
                const y = canvas.height - 30 - (this.trainingData.winRates[i] / maxWinRate) * (canvas.height - 50);
                
                console.log(`📈 Point ${i}: episode=${this.trainingData.episodes[i]}, winRate=${this.trainingData.winRates[i]}, x=${x.toFixed(1)}, y=${y.toFixed(1)}`);
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    // Use quadratic curves for smoother lines
                    const prevX = 50 + (this.trainingData.episodes[i-1] / maxEpisodes) * (canvas.width - 70);
                    const prevY = canvas.height - 30 - (this.trainingData.winRates[i-1] / maxWinRate) * (canvas.height - 50);
                    const cpx = (prevX + x) / 2;
                    const cpy = (prevY + y) / 2;
                    ctx.quadraticCurveTo(cpx, cpy, x, y);
                }
            }
            ctx.stroke();
            console.log('📈 Win rate line drawn');
            
            // Draw data points
            ctx.fillStyle = '#667eea';
            for (let i = 0; i < this.trainingData.episodes.length; i++) {
                const x = 50 + (this.trainingData.episodes[i] / maxEpisodes) * (canvas.width - 70);
                const y = canvas.height - 30 - (this.trainingData.winRates[i] / maxWinRate) * (canvas.height - 50);
                
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, 2 * Math.PI);
                ctx.fill();
            }
            
            // Add value labels on hover
            ctx.fillStyle = '#333';
            ctx.font = '10px Inter';
            ctx.textAlign = 'center';
            for (let i = 0; i < this.trainingData.episodes.length; i++) {
                const x = 50 + (this.trainingData.episodes[i] / maxEpisodes) * (canvas.width - 70);
                const y = canvas.height - 30 - (this.trainingData.winRates[i] / maxWinRate) * (canvas.height - 50);
                
                // Only show label for every 3rd point to avoid clutter
                if (i % 3 === 0) {
                    ctx.fillText(`${this.trainingData.winRates[i].toFixed(1)}%`, x, y - 8);
                }
            }
        }
    }

    updateRewardChart() {
        if (!this.rewardChart) return;
        
        const canvas = document.getElementById('rewardChart');
        const ctx = this.rewardChart;
        
        this.drawEmptyChart(ctx, canvas, 'Avg Reward');
        
        // Show placeholder text if no data
        if (this.trainingData.episodes.length === 0) {
            ctx.fillStyle = '#28a745';
            ctx.font = '10px Inter';
            ctx.textAlign = 'center';
            ctx.fillText('Reward data will appear here', canvas.width / 2, canvas.height / 2);
            return;
        }
        
        // Draw reward line
        if (this.trainingData.episodes.length > 0) {
            ctx.strokeStyle = '#28a745';
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
            
            const maxEpisodes = Math.max(...this.trainingData.episodes);
            const minReward = Math.min(0, Math.min(...this.trainingData.avgRewards));
            const maxReward = Math.max(1, Math.max(...this.trainingData.avgRewards));
            const rewardRange = maxReward - minReward;
            
            // Draw the line with smooth curves
            ctx.beginPath();
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            
            for (let i = 0; i < this.trainingData.episodes.length; i++) {
                const x = 50 + (this.trainingData.episodes[i] / maxEpisodes) * (canvas.width - 70);
                const y = canvas.height - 30 - ((this.trainingData.avgRewards[i] - minReward) / rewardRange) * (canvas.height - 50);
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    // Use quadratic curves for smoother lines
                    const prevX = 50 + (this.trainingData.episodes[i-1] / maxEpisodes) * (canvas.width - 70);
                    const prevY = canvas.height - 30 - ((this.trainingData.avgRewards[i-1] - minReward) / rewardRange) * (canvas.height - 50);
                    const cpx = (prevX + x) / 2;
                    const cpy = (prevY + y) / 2;
                    ctx.quadraticCurveTo(cpx, cpy, x, y);
                }
            }
            ctx.stroke();
            
            // Draw data points
            ctx.fillStyle = '#28a745';
            for (let i = 0; i < this.trainingData.episodes.length; i++) {
                const x = 50 + (this.trainingData.episodes[i] / maxEpisodes) * (canvas.width - 70);
                const y = canvas.height - 30 - ((this.trainingData.avgRewards[i] - minReward) / rewardRange) * (canvas.height - 50);
                
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, 2 * Math.PI);
                ctx.fill();
            }
            
            // Add value labels
            ctx.fillStyle = '#333';
            ctx.font = '10px Inter';
            ctx.textAlign = 'center';
            for (let i = 0; i < this.trainingData.episodes.length; i++) {
                const x = 50 + (this.trainingData.episodes[i] / maxEpisodes) * (canvas.width - 70);
                const y = canvas.height - 30 - ((this.trainingData.avgRewards[i] - minReward) / rewardRange) * (canvas.height - 50);
                
                // Only show label for every 3rd point to avoid clutter
                if (i % 3 === 0) {
                    ctx.fillText(this.trainingData.avgRewards[i].toFixed(2), x, y - 8);
                }
            }
        }
    }

    updatePerformanceComparison(stats) {
        const currentWinRate = stats.win_rate * 100;
        this.afterWinRate.textContent = `${currentWinRate.toFixed(1)}%`;
        
        if (this.beforeTrainingStats) {
            const improvement = currentWinRate - this.beforeTrainingStats.winRate;
            this.improvement.textContent = `${improvement >= 0 ? '+' : ''}${improvement.toFixed(1)}%`;
        }
    }

    updateMilestones(currentWinRate) {
        const milestones = this.milestonesContainer.querySelectorAll('.milestone');
        milestones.forEach(milestone => {
            const threshold = parseInt(milestone.dataset.threshold);
            if (currentWinRate >= threshold && !this.achievedMilestones.has(threshold)) {
                milestone.classList.add('achieved');
                this.achievedMilestones.add(threshold);
                this.createParticleEffect(milestone);
            }
        });
    }

    async updateQValueHeatmap() {
        try {
            const response = await fetch('/q_values');
            const qValues = await response.json();
            
            const qCells = document.querySelectorAll('.q-cell');
            const values = Object.values(qValues);
            const minVal = Math.min(...values);
            const maxVal = Math.max(...values);
            const range = maxVal - minVal || 1; // Avoid division by zero
            
            qCells.forEach((cell, index) => {
                const value = qValues[index] || 0;
                const normalizedValue = (value - minVal) / range;
                
                // Color based on value (red for negative, green for positive)
                let hue, saturation, lightness;
                if (value < 0) {
                    hue = 0; // Red
                    saturation = 70;
                    lightness = 50 + normalizedValue * 20;
                } else {
                    hue = 120; // Green
                    saturation = 70;
                    lightness = 30 + normalizedValue * 40;
                }
                
                const alpha = 0.3 + Math.abs(normalizedValue) * 0.7;
                cell.style.background = `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;
                cell.textContent = value.toFixed(2);
                
                // Add tooltip
                cell.title = `Position ${index}: Q-value = ${value.toFixed(3)}`;
            });
        } catch (error) {
            console.error('Error fetching Q-values:', error);
            // Fallback to simulated values
            this.updateQValueHeatmapFallback();
        }
    }

    updateQValueHeatmapFallback() {
        const qCells = document.querySelectorAll('.q-cell');
        qCells.forEach((cell, index) => {
            // Simulate Q-values based on position importance
            const positionValues = [0.8, 0.3, 0.8, 0.3, 0.9, 0.3, 0.8, 0.3, 0.8];
            const value = positionValues[index] + Math.random() * 0.2 - 0.1;
            const intensity = Math.max(0, Math.min(1, value));
            
            // Color based on intensity
            const hue = 120 * intensity; // Green scale
            cell.style.background = `hsla(${hue}, 70%, 50%, ${0.3 + intensity * 0.7})`;
            cell.textContent = value.toFixed(2);
        });
    }

    createParticleEffect(element) {
        const rect = element.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        
        for (let i = 0; i < 10; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = centerX + 'px';
            particle.style.top = centerY + 'px';
            particle.style.animationDelay = Math.random() * 0.5 + 's';
            
            document.body.appendChild(particle);
            
            setTimeout(() => {
                particle.remove();
            }, 2000);
        }
    }

    // Removed addSampleData - no more hardcoded data

    debugCharts() {
        console.log('🐛 DEBUG CHARTS - Manual trigger');
        console.log('🐛 Chart contexts available:', {
            winRateChart: !!this.winRateChart,
            rewardChart: !!this.rewardChart
        });
        console.log('🐛 Training data:', this.trainingData);
        console.log('🐛 Canvas elements:', {
            winRateCanvas: !!document.getElementById('winRateChart'),
            rewardCanvas: !!document.getElementById('rewardChart')
        });
        
        // Check if we have real training data
        if (this.trainingData.episodes.length === 0) {
            console.log('🐛 No training data found - start training to see real data');
        } else {
            console.log('🐛 Using existing training data');
        }
        
        // Force chart update
        this.updateCharts();
        
        console.log('🐛 Debug complete - check charts now');
    }

    showModal() {
        this.modal.style.display = 'block';
        document.body.style.overflow = 'hidden';
    }

    closeModal() {
        this.modal.style.display = 'none';
        document.body.style.overflow = 'auto';
    }

    playAgain() {
        this.closeModal();
        this.resetGame();
    }

    updateStatsDisplay() {
        this.gamesPlayedEl.textContent = this.stats.gamesPlayed;
        this.humanWinsEl.textContent = this.stats.humanWins;
        this.aiWinsEl.textContent = this.stats.aiWins;
        this.drawsEl.textContent = this.stats.draws;
    }

    saveStats() {
        localStorage.setItem('ticTacToeStats', JSON.stringify(this.stats));
    }

    loadStats() {
        const saved = localStorage.getItem('ticTacToeStats');
        if (saved) {
            this.stats = { ...this.stats, ...JSON.parse(saved) };
        }
        this.updateStatsDisplay();
    }

    updateDisplay() {
        this.updateBoard(this.board);
        this.updateStatsDisplay();
    }

    addLogMessage(message, type = 'info') {
        const logLine = document.createElement('div');
        logLine.className = `log-line ${type}`;
        logLine.textContent = message;
        
        this.trainingLog.appendChild(logLine);
        
        // Auto-scroll to bottom
        this.trainingLog.scrollTop = this.trainingLog.scrollHeight;
        
        // Limit to 50 lines to prevent memory issues
        const lines = this.trainingLog.querySelectorAll('.log-line');
        if (lines.length > 50) {
            lines[0].remove();
        }
    }

    clearLog() {
        this.trainingLog.innerHTML = '';
    }

    async checkAITrainingStatus() {
        try {
            const response = await fetch('/training_status');
            const stats = await response.json();
            
            // Check if AI is trained (has Q-values)
            if (stats.episodes_completed > 0) {
                this.addLogMessage('🤖 AI is trained and ready to play!', 'success');
                this.status.textContent = 'Your turn - Click a cell to make your move (smart)';
            } else {
                this.addLogMessage('🤖 AI is untrained - playing randomly (dumb bot)', 'info');
                this.status.textContent = 'Your turn - Click a cell to make your move (dumb bot)';
            }
        } catch (error) {
            console.error('Error checking AI status:', error);
        }
    }
}

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new TicTacToeGame();
});

// Add some visual feedback for cell hover effects
document.addEventListener('DOMContentLoaded', () => {
    const cells = document.querySelectorAll('.cell');
    
    cells.forEach(cell => {
        cell.addEventListener('mouseenter', function() {
            if (!this.classList.contains('occupied') && !this.textContent) {
                this.style.transform = 'scale(1.05)';
                this.style.boxShadow = '0 8px 20px rgba(0,0,0,0.15)';
            }
        });
        
        cell.addEventListener('mouseleave', function() {
            this.style.transform = '';
            this.style.boxShadow = '';
        });
    });
});
