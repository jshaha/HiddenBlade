/**
 * EMG Gesture Recognition Dashboard
 * Real-time WebSocket client for EMG visualization and gesture display
 */

class EMGDashboard {
    constructor() {
        this.ws = null;
        this.isRunning = false;
        this.emgData = [];
        this.channelColors = [
            '#00f0ff', '#ff00aa', '#00ff88', '#ffaa00',
            '#ff6b6b', '#4ecdc4', '#a55eea', '#26de81'
        ];
        this.maxDataPoints = 200;

        // Canvas for EMG chart
        this.canvas = document.getElementById('emgChart');
        this.ctx = this.canvas.getContext('2d');

        // Initialize data arrays for 8 channels
        for (let i = 0; i < 8; i++) {
            this.emgData.push([]);
        }

        // Bind methods
        this.resizeCanvas = this.resizeCanvas.bind(this);
        this.drawEMG = this.drawEMG.bind(this);

        // Setup
        this.setupCanvas();
        this.startAnimation();

        // Start automatically in mock mode
        setTimeout(() => this.toggleSystem(), 500);
    }

    setupCanvas() {
        this.resizeCanvas();
        window.addEventListener('resize', this.resizeCanvas);
    }

    resizeCanvas() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = Math.max(300, rect.height - 80);
    }

    connect() {
        const wsUrl = `ws://${window.location.host}/ws`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
            // Attempt reconnect after delay
            if (this.isRunning) {
                setTimeout(() => this.connect(), 2000);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    handleMessage(data) {
        if (data.type === 'update') {
            // Update EMG data
            if (data.emg) {
                for (let i = 0; i < data.emg.length; i++) {
                    this.emgData[i].push(data.emg[i]);
                    if (this.emgData[i].length > this.maxDataPoints) {
                        this.emgData[i].shift();
                    }
                }
            }

            // Update gesture display
            this.updateGestureDisplay(data.gesture, data.confidence, data.is_new);

            // Update command display
            this.updateCommandDisplay(data.command);

            // Update stats
            if (data.stats) {
                this.updateStats(data.stats);
            }
        }
    }

    updateGestureDisplay(gesture, confidence, isNew) {
        const handSvg = document.getElementById('handSvg');
        const gestureLabel = document.getElementById('gestureLabel');
        const confidenceBar = document.getElementById('confidenceBar');
        const confidenceValue = document.getElementById('confidenceValue');
        const gestureGlow = document.getElementById('gestureGlow');

        // Update hand pose
        handSvg.className = 'hand-svg';
        if (gesture === 'closed_hand') {
            handSvg.classList.add('closed-hand');
            gestureLabel.style.color = '#ff00aa';
        } else if (gesture === 'open_hand') {
            handSvg.classList.add('open-hand');
            gestureLabel.style.color = '#00ff88';
        } else if (gesture === 'pointing') {
            handSvg.classList.add('pointing');
            gestureLabel.style.color = '#ffaa00';
        } else {
            gestureLabel.style.color = '#00f0ff';
        }

        // Update label
        gestureLabel.textContent = gesture.toUpperCase().replace('_', ' ');

        // Update confidence
        const confPercent = Math.round(confidence * 100);
        confidenceBar.style.width = `${confPercent}%`;
        confidenceValue.textContent = `${confPercent}%`;

        // Pulse effect on new gesture
        if (isNew) {
            gestureGlow.classList.add('active');
            gestureLabel.style.transform = 'scale(1.1)';
            setTimeout(() => {
                gestureGlow.classList.remove('active');
                gestureLabel.style.transform = 'scale(1)';
            }, 500);
        }
    }

    updateCommandDisplay(command) {
        const hexDisplay = document.getElementById('commandHex');
        const binaryDisplay = document.getElementById('commandBinary');

        hexDisplay.textContent = `0x${command}`;

        // Convert hex to binary
        const decimal = parseInt(command, 16);
        const binary = decimal.toString(2).padStart(8, '0');
        binaryDisplay.textContent = binary;
    }

    updateStats(stats) {
        // Update status
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        if (stats.running) {
            statusDot.classList.add('online');
            statusText.textContent = stats.mock_mode ? 'MOCK MODE' : 'LIVE';
        } else {
            statusDot.classList.remove('online');
            statusText.textContent = 'OFFLINE';
        }

        // Update gesture counts
        const counts = stats.gesture_counts || {};
        const maxCount = Math.max(1, ...Object.values(counts));

        document.getElementById('countClosedHand').textContent = counts.closed_hand || 0;
        document.getElementById('countOpenHand').textContent = counts.open_hand || 0;
        document.getElementById('countPointing').textContent = counts.pointing || 0;

        document.getElementById('barClosedHand').style.width =
            `${((counts.closed_hand || 0) / maxCount) * 100}%`;
        document.getElementById('barOpenHand').style.width =
            `${((counts.open_hand || 0) / maxCount) * 100}%`;
        document.getElementById('barPointing').style.width =
            `${((counts.pointing || 0) / maxCount) * 100}%`;

        // Update performance stats
        document.getElementById('inferenceRate').textContent =
            stats.inference_rate?.toFixed(1) || '0.0';
        document.getElementById('totalInferences').textContent =
            stats.inference_count || 0;

        // Update uptime
        const uptime = stats.uptime || 0;
        const minutes = Math.floor(uptime / 60);
        const seconds = Math.floor(uptime % 60);
        document.getElementById('uptime').textContent =
            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }

    updateConnectionStatus(connected) {
        const wsStatus = document.getElementById('wsStatus');
        wsStatus.textContent = connected ? 'Connected' : 'Disconnected';
        wsStatus.className = 'conn-status' + (connected ? ' connected' : '');
    }

    drawEMG() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear canvas
        ctx.fillStyle = '#22222f';
        ctx.fillRect(0, 0, width, height);

        // Draw grid
        ctx.strokeStyle = '#2a2a3a';
        ctx.lineWidth = 1;

        // Vertical grid lines
        for (let x = 0; x < width; x += 50) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }

        // Horizontal grid lines
        for (let y = 0; y < height; y += 30) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Draw each channel
        const channelHeight = height / 8;
        const padding = 5;

        for (let ch = 0; ch < 8; ch++) {
            const data = this.emgData[ch];
            if (data.length < 2) continue;

            const yOffset = ch * channelHeight + channelHeight / 2;

            // Draw channel label
            ctx.fillStyle = this.channelColors[ch];
            ctx.font = '10px Orbitron';
            ctx.fillText(`CH${ch + 1}`, 5, yOffset - channelHeight / 2 + 12);

            // Draw signal
            ctx.beginPath();
            ctx.strokeStyle = this.channelColors[ch];
            ctx.lineWidth = 1.5;

            const xStep = (width - 40) / this.maxDataPoints;

            for (let i = 0; i < data.length; i++) {
                const x = 40 + i * xStep;
                // Normalize signal to channel height
                const normalized = data[i] / 500; // Assume signal range
                const y = yOffset - normalized * (channelHeight / 2 - padding);

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }

            ctx.stroke();

            // Draw glow effect
            ctx.strokeStyle = this.channelColors[ch] + '40';
            ctx.lineWidth = 4;
            ctx.stroke();
        }
    }

    startAnimation() {
        const animate = () => {
            this.drawEMG();
            requestAnimationFrame(animate);
        };
        animate();
    }

    toggleSystem() {
        const btn = document.getElementById('startBtn');
        const btnText = btn.querySelector('.btn-text');
        const btnIcon = btn.querySelector('.btn-icon');

        if (!this.isRunning) {
            // Start system
            this.isRunning = true;
            this.connect();
            btn.classList.add('running');
            btnText.textContent = 'STOP SYSTEM';
            btnIcon.textContent = '⏹';
        } else {
            // Stop system
            this.isRunning = false;
            this.disconnect();
            btn.classList.remove('running');
            btnText.textContent = 'START SYSTEM';
            btnIcon.textContent = '▶';

            // Reset displays
            document.getElementById('gestureLabel').textContent = 'WAITING...';
            document.getElementById('confidenceBar').style.width = '0%';
            document.getElementById('confidenceValue').textContent = '0%';
            document.getElementById('statusDot').classList.remove('online');
            document.getElementById('statusText').textContent = 'OFFLINE';
        }
    }
}

// Initialize dashboard
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new EMGDashboard();
});

// Global function for button
function toggleSystem() {
    if (dashboard) {
        dashboard.toggleSystem();
    }
}
