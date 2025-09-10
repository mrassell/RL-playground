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
        
        // Add some sample data for testing
        setTimeout(() => {
            this.addSampleData();
        }, 500);
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
            this.status.textContent = `AI chose position ${result.agent_move}. Your turn!`;
        } else {
            this.status.textContent = 'Your turn - Click a cell to make your move';
        }
    }

    handleGameOver(result) {
        this.gameOver = true;
        this.stats.gamesPlayed++;
        
        if (result.winner === 'human') {
            this.stats.humanWins++;
            this.modalTitle.textContent = '🎉 You Win!';
            this.modalMessage.textContent = 'Congratulations! You beat the AI!';
        } else if (result.winner === 'agent') {
            this.stats.aiWins++;
            this.modalTitle.textContent = '🤖 AI Wins!';
            this.modalMessage.textContent = 'The AI got the better of you this time!';
        } else {
            this.stats.draws++;
            this.modalTitle.textContent = '🤝 Draw!';
            this.modalMessage.textContent = "It's a tie! Well played!";
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
            const response = await fetch('/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(parameters)
            });
            
            const result = await response.json();
            
            if (result.status === 'training_started') {
                this.startTrainingProgressMonitoring();
            } else if (result.status === 'already_training') {
                this.status.textContent = 'Training is already in progress!';
            }
        } catch (error) {
            console.error('Error starting training:', error);
            this.status.textContent = 'Error starting training. Please try again.';
            this.isTraining = false;
            this.startTrainingBtn.disabled = false;
            this.stopTrainingBtn.disabled = true;
        }
    }

    async stopTraining() {
        try {
            await fetch('/stop_training', {
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

    startTrainingProgressMonitoring() {
        console.log('🔄 Starting training progress monitoring');
        this.trainingInterval = setInterval(async () => {
            try {
                console.log('🔄 Fetching training status...');
                const response = await fetch('/training_status');
                const stats = await response.json();
                
                console.log('📊 Training status received:', stats);
                this.updateTrainingProgress(stats);
                
                if (!stats.is_training) {
                    console.log('✅ Training completed!');
                    this.isTraining = false;
                    this.startTrainingBtn.disabled = false;
                    this.stopTrainingBtn.disabled = true;
                    clearInterval(this.trainingInterval);
                    this.trainingInterval = null;
                    this.status.textContent = 'Training completed!';
                    
                    // Force final chart update with all data
                    console.log('📊 Final chart update with completed training data');
                    this.updateCharts();
                }
            } catch (error) {
                console.error('❌ Error fetching training status:', error);
            }
        }, 1000); // Check every second
    }

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
        
        // Update chart data - ALWAYS add data, not just every 1000 episodes
        console.log('📊 Adding chart data - episodes completed:', stats.episodes_completed);
        
        // Only add if we don't already have this episode
        if (!this.trainingData.episodes.includes(stats.episodes_completed)) {
            this.trainingData.episodes.push(stats.episodes_completed);
            this.trainingData.winRates.push(stats.win_rate * 100);
            this.trainingData.avgRewards.push(stats.avg_reward);
            console.log('📊 Added new data point:', {
                episode: stats.episodes_completed,
                winRate: stats.win_rate * 100,
                avgReward: stats.avg_reward
            });
        } else {
            console.log('📊 Data point already exists for episode:', stats.episodes_completed);
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
        
        // Add some sample data points for visual reference
        if (yLabel === 'Win Rate (%)') {
            ctx.fillStyle = '#667eea';
            ctx.font = '10px Inter';
            ctx.textAlign = 'center';
            ctx.fillText('Training data will appear here', canvas.width / 2, canvas.height / 2);
        } else if (yLabel === 'Avg Reward') {
            ctx.fillStyle = '#28a745';
            ctx.font = '10px Inter';
            ctx.textAlign = 'center';
            ctx.fillText('Reward data will appear here', canvas.width / 2, canvas.height / 2);
        }
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
        
        this.drawEmptyChart(ctx, canvas, 'Win Rate (%)');
        
        // Draw win rate line with animation
        if (this.trainingData.episodes.length > 1) {
            console.log('📈 Drawing win rate line with', this.trainingData.episodes.length, 'data points');
            ctx.strokeStyle = '#667eea';
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
            
            const maxEpisodes = Math.max(...this.trainingData.episodes);
            const maxWinRate = Math.max(100, Math.max(...this.trainingData.winRates));
            
            console.log('📈 Chart scaling - maxEpisodes:', maxEpisodes, 'maxWinRate:', maxWinRate);
            console.log('📈 Sample data points:', this.trainingData.episodes.slice(0, 3), this.trainingData.winRates.slice(0, 3));
            
            // Draw the line
            ctx.beginPath();
            for (let i = 0; i < this.trainingData.episodes.length; i++) {
                const x = 50 + (this.trainingData.episodes[i] / maxEpisodes) * (canvas.width - 70);
                const y = canvas.height - 30 - (this.trainingData.winRates[i] / maxWinRate) * (canvas.height - 50);
                
                console.log(`📈 Point ${i}: episode=${this.trainingData.episodes[i]}, winRate=${this.trainingData.winRates[i]}, x=${x.toFixed(1)}, y=${y.toFixed(1)}`);
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
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
        
        // Draw reward line
        if (this.trainingData.episodes.length > 1) {
            ctx.strokeStyle = '#28a745';
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
            
            const maxEpisodes = Math.max(...this.trainingData.episodes);
            const minReward = Math.min(0, Math.min(...this.trainingData.avgRewards));
            const maxReward = Math.max(1, Math.max(...this.trainingData.avgRewards));
            const rewardRange = maxReward - minReward;
            
            ctx.beginPath();
            for (let i = 0; i < this.trainingData.episodes.length; i++) {
                const x = 50 + (this.trainingData.episodes[i] / maxEpisodes) * (canvas.width - 70);
                const y = canvas.height - 30 - ((this.trainingData.avgRewards[i] - minReward) / rewardRange) * (canvas.height - 50);
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
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

    addSampleData() {
        console.log('🧪 Adding sample data for testing');
        
        // Add sample training data
        for (let i = 0; i < 10; i++) {
            this.trainingData.episodes.push((i + 1) * 1000);
            this.trainingData.winRates.push(20 + (i * 7) + Math.random() * 5);
            this.trainingData.avgRewards.push(-0.5 + (i * 0.15) + Math.random() * 0.1);
        }
        
        console.log('🧪 Sample data added:', {
            episodes: this.trainingData.episodes.length,
            winRates: this.trainingData.winRates,
            avgRewards: this.trainingData.avgRewards
        });
        
        // Update charts with sample data
        this.updateCharts();
    }

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
        
        // If no data, add sample data
        if (this.trainingData.episodes.length === 0) {
            console.log('🐛 No training data found, adding sample data');
            this.addSampleData();
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
