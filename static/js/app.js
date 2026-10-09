// app.js - AI Music Generation Studio Client Logic

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const numNotesSlider = document.getElementById('num-notes');
    const notesValSpan = document.getElementById('notes-val');

    const tempSlider = document.getElementById('temperature');
    const tempValSpan = document.getElementById('temp-val');

    const bpmSlider = document.getElementById('bpm-tempo');
    const bpmValSpan = document.getElementById('bpm-val');

    const btnGenerate = document.getElementById('btn-generate');
    const loadingState = document.getElementById('gen-loading');

    const playerEmpty = document.getElementById('player-empty');
    const playerActive = document.getElementById('player-active');
    const playerStatusBadge = document.getElementById('player-status-badge');

    const audioElement = document.getElementById('audio-element');
    const trackTitle = document.getElementById('current-track-title');
    const trackMeta = document.getElementById('current-track-meta');
    const btnDlMidi = document.getElementById('btn-dl-midi');
    const btnDlWav = document.getElementById('btn-dl-wav');
    const pianoRollImg = document.getElementById('piano-roll-img');

    const historyContainer = document.getElementById('history-container');
    const btnRefreshHistory = document.getElementById('btn-refresh-history');

    // Stats
    const statVocab = document.getElementById('stat-vocab');
    const statFiles = document.getElementById('stat-files');
    const statLoss = document.getElementById('stat-loss');

    // Slider Listeners
    numNotesSlider.addEventListener('input', (e) => {
        notesValSpan.textContent = `${e.target.value} notes`;
    });

    tempSlider.addEventListener('input', (e) => {
        tempValSpan.textContent = parseFloat(e.target.value).toFixed(2);
    });

    bpmSlider.addEventListener('input', (e) => {
        bpmValSpan.textContent = `${e.target.value} BPM`;
    });

    // Fetch Stats & History
    fetchStats();
    fetchHistory();

    // Generate Button Click
    btnGenerate.addEventListener('click', async () => {
        const notes = parseInt(numNotesSlider.value, 10);
        const temp = parseFloat(tempSlider.value);
        const bpm = parseInt(bpmSlider.value, 10);

        // UI Loading
        btnGenerate.disabled = true;
        loadingState.classList.remove('hidden');
        playerStatusBadge.textContent = 'Synthesizing...';

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ notes, temperature: temp, bpm })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to generate music');
            }

            // Update Player UI
            playerEmpty.classList.add('hidden');
            playerActive.classList.remove('hidden');
            playerStatusBadge.textContent = 'Composition Ready';

            trackTitle.textContent = data.midi_filename;
            trackMeta.textContent = `${data.num_notes} Notes • ${bpm} BPM • Temperature ${temp}`;

            // Set Audio Source
            if (data.wav_url) {
                audioElement.src = data.wav_url;
                audioElement.play().catch(() => {});
                btnDlWav.href = data.wav_url;
                btnDlWav.classList.remove('hidden');
            } else {
                btnDlWav.classList.add('hidden');
            }

            btnDlMidi.href = data.midi_url;
            pianoRollImg.src = data.roll_url + '?t=' + new Date().getTime();

            // Refresh History
            fetchHistory();

        } catch (err) {
            alert('Generation error: ' + err.message);
            playerStatusBadge.textContent = 'Error';
        } finally {
            btnGenerate.disabled = false;
            loadingState.classList.add('hidden');
        }
    });

    btnRefreshHistory.addEventListener('click', fetchHistory);

    async function fetchStats() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            if (data.vocab_size) statVocab.textContent = data.vocab_size;
            if (data.midi_files) statFiles.textContent = data.midi_files;
            if (data.best_loss) statLoss.textContent = parseFloat(data.best_loss).toFixed(3);
        } catch (e) {
            console.warn('Could not fetch stats', e);
        }
    }

    async function fetchHistory() {
        try {
            const res = await fetch('/api/history');
            const data = await res.json();

            if (!data.history || data.history.length === 0) {
                historyContainer.innerHTML = '<p class="text-muted">No generated tracks yet.</p>';
                return;
            }

            historyContainer.innerHTML = '';
            data.history.slice(0, 8).forEach(item => {
                const el = document.createElement('div');
                el.className = 'history-item';
                el.innerHTML = `
                    <span class="history-title">${item.filename}</span>
                    <div class="history-links">
                        ${item.wav_url ? `<a href="${item.wav_url}" target="_blank">▶ Play WAV</a>` : ''}
                        <a href="${item.midi_url}" download>MIDI</a>
                    </div>
                `;
                historyContainer.appendChild(el);
            });
        } catch (e) {
            historyContainer.innerHTML = '<p class="text-muted">Unable to load history.</p>';
        }
    }
});

