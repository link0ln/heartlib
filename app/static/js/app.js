// HeartMuLa Web UI Application

class HeartMuLaApp {
    constructor() {
        this.currentJobId = null;
        this.pollInterval = null;
        this.gpuPollInterval = null;
        this.audioPlayer = null;
        this.isPlaying = false;
        this.library = [];

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadLibrary();
        this.startGpuMonitoring();
        this.checkHealth();
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                this.switchTab(item.dataset.tab);
            });
        });

        // Form submission
        document.getElementById('generate-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.startGeneration();
        });

        // Tag chips
        document.querySelectorAll('.tag-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                this.toggleTag(chip);
            });
        });

        // Slider value displays
        document.getElementById('temperature').addEventListener('input', (e) => {
            document.getElementById('temp-value').textContent = e.target.value;
        });

        document.getElementById('cfg_scale').addEventListener('input', (e) => {
            document.getElementById('cfg-value').textContent = e.target.value;
        });

        // Audio player
        this.audioPlayer = document.getElementById('audio-player');
        this.audioPlayer.addEventListener('timeupdate', () => this.updatePlayerTime());
        this.audioPlayer.addEventListener('ended', () => this.onAudioEnded());
        this.audioPlayer.addEventListener('loadedmetadata', () => this.onAudioLoaded());

        document.getElementById('btn-play').addEventListener('click', () => this.togglePlay());
        document.getElementById('player-seek').addEventListener('input', (e) => this.seekAudio(e));
        document.getElementById('btn-download').addEventListener('click', () => this.downloadTrack());
    }

    switchTab(tab) {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.toggle('active', item.dataset.tab === tab);
        });

        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === `tab-${tab}`);
        });

        if (tab === 'library') {
            this.loadLibrary();
        }
    }

    toggleTag(chip) {
        const tag = chip.dataset.tag;
        const tagsInput = document.getElementById('tags');
        let tags = tagsInput.value.split(',').map(t => t.trim()).filter(t => t);

        if (chip.classList.contains('selected')) {
            chip.classList.remove('selected');
            tags = tags.filter(t => t !== tag);
        } else {
            chip.classList.add('selected');
            if (!tags.includes(tag)) {
                tags.push(tag);
            }
        }

        tagsInput.value = tags.join(', ');
    }

    async checkHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();

            if (data.model_loaded) {
                document.getElementById('model-dtype').textContent =
                    data.gpu_available ? 'FP16 GPU' : 'CPU';
            }
        } catch (error) {
            console.error('Health check failed:', error);
        }
    }

    async startGpuMonitoring() {
        this.updateGpuStats();
        this.gpuPollInterval = setInterval(() => this.updateGpuStats(), 2000);
    }

    async updateGpuStats() {
        try {
            const response = await fetch('/api/gpu');
            const data = await response.json();

            if (data.available) {
                document.getElementById('gpu-name').textContent = data.name || 'GPU';
                document.getElementById('gpu-util').textContent = `${data.utilization}%`;
                document.getElementById('gpu-util-bar').style.width = `${data.utilization}%`;

                const memUsed = (data.memory_used / 1024).toFixed(1);
                const memTotal = (data.memory_total / 1024).toFixed(1);
                const memPercent = (data.memory_used / data.memory_total * 100).toFixed(0);
                document.getElementById('gpu-mem').textContent = `${memUsed} / ${memTotal} GB`;
                document.getElementById('gpu-mem-bar').style.width = `${memPercent}%`;

                document.getElementById('gpu-temp').textContent = `${data.temperature}°C`;
                document.getElementById('gpu-power').textContent = `${data.power_draw}W / ${data.power_limit}W`;
            } else {
                document.getElementById('gpu-name').textContent = 'Not available';
            }
        } catch (error) {
            console.error('GPU stats update failed:', error);
        }
    }

    async startGeneration() {
        const form = document.getElementById('generate-form');
        const btn = document.getElementById('btn-generate');

        const lyrics = document.getElementById('lyrics').value;
        const tags = document.getElementById('tags').value;
        const duration = parseInt(document.getElementById('duration').value);
        const temperature = parseFloat(document.getElementById('temperature').value);
        const cfg_scale = parseFloat(document.getElementById('cfg_scale').value);

        if (!lyrics.trim()) {
            alert('Please enter lyrics');
            return;
        }

        if (!tags.trim()) {
            alert('Please enter at least one tag');
            return;
        }

        btn.disabled = true;
        btn.innerHTML = '<span class="btn-icon">⏳</span> Starting...';

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    lyrics,
                    tags,
                    max_duration_ms: duration,
                    temperature,
                    cfg_scale,
                }),
            });

            const data = await response.json();

            if (response.ok) {
                this.currentJobId = data.job_id;
                this.showGenerationStatus();
                this.startPolling();

                // Save to library
                this.saveToLibrary({
                    job_id: data.job_id,
                    lyrics: lyrics.substring(0, 100),
                    tags,
                    duration,
                    created_at: new Date().toISOString(),
                    status: 'processing',
                });
            } else {
                alert(`Error: ${data.detail || 'Failed to start generation'}`);
                btn.disabled = false;
                btn.innerHTML = '<span class="btn-icon">♪</span> Generate Music';
            }
        } catch (error) {
            console.error('Generation start failed:', error);
            alert('Failed to start generation. Please try again.');
            btn.disabled = false;
            btn.innerHTML = '<span class="btn-icon">♪</span> Generate Music';
        }
    }

    showGenerationStatus() {
        document.getElementById('generation-status').style.display = 'block';
        document.getElementById('result-player').style.display = 'none';
        document.getElementById('status-badge').className = 'status-badge processing';
        document.getElementById('status-badge').textContent = 'processing';
        document.getElementById('generation-progress').style.width = '0%';
        document.getElementById('progress-text').textContent = '0%';
        document.getElementById('status-message').textContent = 'Starting generation...';
    }

    startPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }

        this.pollInterval = setInterval(() => this.checkJobStatus(), 1000);
    }

    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    async checkJobStatus() {
        if (!this.currentJobId) return;

        try {
            const response = await fetch(`/generate/${this.currentJobId}/status`);
            const data = await response.json();

            const progress = data.progress || 0;
            document.getElementById('generation-progress').style.width = `${progress}%`;
            document.getElementById('progress-text').textContent = `${Math.round(progress)}%`;

            const badge = document.getElementById('status-badge');
            badge.className = `status-badge ${data.status}`;
            badge.textContent = data.status;

            // Update status message
            if (data.status === 'queued') {
                document.getElementById('status-message').textContent = 'Waiting in queue...';
            } else if (data.status === 'processing') {
                if (progress < 10) {
                    document.getElementById('status-message').textContent = 'Preparing model...';
                } else if (progress < 90) {
                    document.getElementById('status-message').textContent = 'Generating audio frames...';
                } else {
                    document.getElementById('status-message').textContent = 'Encoding audio...';
                }
            } else if (data.status === 'completed') {
                document.getElementById('status-message').textContent = 'Generation complete!';
                this.onGenerationComplete(data);
            } else if (data.status === 'failed') {
                document.getElementById('status-message').textContent = `Error: ${data.error || 'Unknown error'}`;
                this.onGenerationFailed(data);
            }
        } catch (error) {
            console.error('Status check failed:', error);
        }
    }

    onGenerationComplete(data) {
        this.stopPolling();

        const btn = document.getElementById('btn-generate');
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">♪</span> Generate Music';

        // Show player
        document.getElementById('result-player').style.display = 'block';
        this.loadAudio(data.result_url);

        // Update library
        this.updateLibraryItem(this.currentJobId, 'completed');
    }

    onGenerationFailed(data) {
        this.stopPolling();

        const btn = document.getElementById('btn-generate');
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">♪</span> Generate Music';

        // Update library
        this.updateLibraryItem(this.currentJobId, 'failed');
    }

    loadAudio(url) {
        this.audioPlayer.src = url;
        this.audioPlayer.load();
    }

    onAudioLoaded() {
        const duration = this.audioPlayer.duration;
        document.getElementById('player-duration').textContent = this.formatTime(duration);
        document.getElementById('player-time').textContent = `0:00 / ${this.formatTime(duration)}`;

        // Generate simple waveform visualization
        this.generateWaveform();
    }

    generateWaveform() {
        const container = document.getElementById('waveform');
        container.innerHTML = '';

        const canvas = document.createElement('canvas');
        canvas.width = container.offsetWidth;
        canvas.height = container.offsetHeight;
        container.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        const bars = 100;
        const barWidth = canvas.width / bars;
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
        gradient.addColorStop(0, '#8b5cf6');
        gradient.addColorStop(1, '#ec4899');

        ctx.fillStyle = gradient;

        for (let i = 0; i < bars; i++) {
            const height = Math.random() * 0.6 + 0.2;
            const barHeight = canvas.height * height;
            const y = (canvas.height - barHeight) / 2;
            ctx.fillRect(i * barWidth + 1, y, barWidth - 2, barHeight);
        }
    }

    togglePlay() {
        const btn = document.getElementById('btn-play');

        if (this.isPlaying) {
            this.audioPlayer.pause();
            btn.textContent = '▶';
            this.isPlaying = false;
        } else {
            this.audioPlayer.play();
            btn.textContent = '⏸';
            this.isPlaying = true;
        }
    }

    onAudioEnded() {
        const btn = document.getElementById('btn-play');
        btn.textContent = '▶';
        this.isPlaying = false;
        document.getElementById('player-seek').value = 0;
    }

    updatePlayerTime() {
        const current = this.audioPlayer.currentTime;
        const duration = this.audioPlayer.duration;
        const percent = (current / duration) * 100;

        document.getElementById('player-seek').value = percent;
        document.getElementById('player-time').textContent =
            `${this.formatTime(current)} / ${this.formatTime(duration)}`;
    }

    seekAudio(e) {
        const percent = e.target.value;
        const duration = this.audioPlayer.duration;
        this.audioPlayer.currentTime = (percent / 100) * duration;
    }

    formatTime(seconds) {
        if (isNaN(seconds)) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    downloadTrack() {
        if (this.currentJobId) {
            const link = document.createElement('a');
            link.href = `/generate/${this.currentJobId}/result`;
            link.download = `heartmula_${this.currentJobId}.wav`;
            link.click();
        }
    }

    // Library Management
    saveToLibrary(track) {
        let library = JSON.parse(localStorage.getItem('heartmula_library') || '[]');
        library.unshift(track);
        library = library.slice(0, 50); // Keep last 50
        localStorage.setItem('heartmula_library', JSON.stringify(library));
    }

    updateLibraryItem(jobId, status) {
        let library = JSON.parse(localStorage.getItem('heartmula_library') || '[]');
        const index = library.findIndex(t => t.job_id === jobId);
        if (index !== -1) {
            library[index].status = status;
            localStorage.setItem('heartmula_library', JSON.stringify(library));
        }
    }

    loadLibrary() {
        const library = JSON.parse(localStorage.getItem('heartmula_library') || '[]');
        const grid = document.getElementById('library-grid');
        const emptyState = document.getElementById('empty-library');

        if (library.length === 0) {
            grid.style.display = 'none';
            emptyState.style.display = 'block';
            return;
        }

        grid.style.display = 'grid';
        emptyState.style.display = 'none';

        grid.innerHTML = library.map(track => `
            <div class="track-card" data-job-id="${track.job_id}" data-status="${track.status}">
                <div class="track-card-header">
                    <div class="track-card-title">${this.getTrackTitle(track)}</div>
                    <div class="track-card-date">${this.formatDate(track.created_at)}</div>
                </div>
                <div class="track-card-tags">
                    ${track.tags.split(',').slice(0, 4).map(tag =>
                        `<span class="tag-chip">${tag.trim()}</span>`
                    ).join('')}
                </div>
                <div class="track-card-controls">
                    ${track.status === 'completed' ? `
                        <button class="btn-play" onclick="app.playLibraryTrack('${track.job_id}')">▶</button>
                        <span class="track-card-duration">${this.formatDuration(track.duration)}</span>
                    ` : `
                        <span class="status-badge ${track.status}">${track.status}</span>
                    `}
                </div>
            </div>
        `).join('');
    }

    getTrackTitle(track) {
        // Extract first line from lyrics
        const firstLine = track.lyrics.split('\n').find(l => l.trim() && !l.startsWith('['));
        return firstLine ? firstLine.substring(0, 30) + (firstLine.length > 30 ? '...' : '') : 'Untitled Track';
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diff = now - date;

        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
        return date.toLocaleDateString();
    }

    formatDuration(ms) {
        const seconds = Math.floor(ms / 1000);
        return this.formatTime(seconds);
    }

    playLibraryTrack(jobId) {
        this.currentJobId = jobId;
        this.loadAudio(`/generate/${jobId}/result`);
        document.getElementById('result-player').style.display = 'block';
        this.switchTab('create');

        // Auto-play
        setTimeout(() => {
            this.audioPlayer.play();
            document.getElementById('btn-play').textContent = '⏸';
            this.isPlaying = true;
        }, 500);
    }
}

// Initialize app
const app = new HeartMuLaApp();
