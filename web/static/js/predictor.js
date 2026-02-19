/**
 * predictor.js — Predictor Page Logic
 * Handles: tabs, upload, cloud library, SSE progress, results display
 */

const Predictor = {
    // ── State ───────────────────────────────────────────────────────
    selectedFile: null,
    selectedRockType: 'MEC',
    useHybrid: false,
    activeSource: 'cloud',   // 'cloud' | 'upload'

    // ── Init (called by app.js on page nav) ─────────────────────────
    init() {
        this.bindSourceTabs();
        this.bindUploadZone();
        this.bindControls();
        this.loadCloudLibrary();
        // Re-enable buttons if we already have a file (e.g. from global state)
        if (this.selectedFile || TopoFlow.selectedFile) {
            this.enableButtons();
        }
    },

    // ── Source Tabs ─────────────────────────────────────────────────
    bindSourceTabs() {
        document.querySelectorAll('#source-tabs .tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('#source-tabs .tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById('tab-cloud').classList.toggle('active', tab.dataset.tab === 'cloud');
                document.getElementById('tab-upload').classList.toggle('active', tab.dataset.tab === 'upload');
                this.activeSource = tab.dataset.tab;
            });
        });
    },

    // ── Upload Zone ─────────────────────────────────────────────────
    bindUploadZone() {
        const zone = document.getElementById('upload-zone');
        const input = document.getElementById('file-input');
        if (!zone || !input) return;

        zone.addEventListener('dragover', e => {
            e.preventDefault();
            zone.classList.add('drag-over');
        });
        zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
        zone.addEventListener('drop', e => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file) this.setFile(file);
        });

        input.addEventListener('change', () => {
            if (input.files[0]) this.setFile(input.files[0]);
        });
    },

    setFile(file) {
        if (!file.name.endsWith('.npy')) {
            alert('Please select a .npy NumPy binary file.');
            return;
        }
        this.selectedFile = file;
        TopoFlow.selectedFile = file;

        const fname = document.getElementById('upload-filename');
        if (fname) {
            fname.textContent = '📦 ' + file.name;
            fname.classList.remove('hidden');
        }

        this.updateSelectedInfo(file.name, `Uploaded file · ${(file.size / 1e6).toFixed(1)} MB`);
        this.enableButtons();
    },

    // ── Cloud Library — Accordion ─────────────────────────────────────
    async loadCloudLibrary() {
        const grid = document.getElementById('sample-grid');
        if (!grid) return;

        let samples;
        try {
            const res = await fetch('/api/rocks');
            if (!res.ok) throw new Error('API error');
            samples = await res.json();
        } catch {
            grid.innerHTML = `<div class="lib-empty">Supabase not connected — upload a local .npy file.</div>`;
            return;
        }

        // Group samples by rock_type
        const groups = {};
        for (const s of samples) {
            if (!groups[s.rock_type]) groups[s.rock_type] = [];
            groups[s.rock_type].push(s);
        }

        grid.innerHTML = Object.entries(groups).map(([rock, items]) => {
            const isGnn = items[0]?.gnn_wins;
            const badgeClass = isGnn ? 'badge--gnn' : 'badge--physics';
            const badgeLabel = isGnn ? '🤖 GNN Wins' : '📐 Physics Wins';

            const sampleCards = items.map(s => `
              <div class="lib-sample" data-id="${s.id}" data-rock="${s.rock_type}">
                <div class="lib-sample__name">${s.name.split('—')[1]?.trim() || s.name}</div>
                <div class="lib-sample__meta">φ = ${(s.porosity * 100).toFixed(1)}%</div>
                <button class="btn btn--primary btn--sm lib-sample__btn" data-id="${s.id}" data-rock="${s.rock_type}">
                  Download &amp; Select
                </button>
                <div class="lib-sample__status"></div>
              </div>
            `).join('');

            return `
              <div class="lib-row">
                <button class="lib-row__header" type="button">
                  <span class="lib-row__name">${rock}</span>
                  <span class="sample-badge ${badgeClass}">${badgeLabel}</span>
                  <span class="lib-row__arrow">›</span>
                </button>
                <div class="lib-row__samples">
                  ${sampleCards}
                </div>
              </div>
            `;
        }).join('');

        // Wire accordion toggles
        grid.querySelectorAll('.lib-row__header').forEach(header => {
            header.addEventListener('click', () => {
                const row = header.closest('.lib-row');
                const panel = row.querySelector('.lib-row__samples');
                const isOpen = panel.classList.contains('is-open');
                // Close all others
                grid.querySelectorAll('.lib-row__samples').forEach(p => p.classList.remove('is-open'));
                grid.querySelectorAll('.lib-row__header').forEach(h => h.classList.remove('open'));
                if (!isOpen) {
                    panel.classList.add('is-open');
                    header.classList.add('open');
                }
            });
        });

        // Wire Download & Select buttons
        grid.querySelectorAll('.lib-sample__btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const sampleId = btn.dataset.id;
                const rock = btn.dataset.rock;
                const status = btn.closest('.lib-sample').querySelector('.lib-sample__status');
                const meta = samples.find(s => s.id === sampleId);

                btn.disabled = true;
                btn.textContent = 'Downloading…';
                status.textContent = '';

                try {
                    const slash = sampleId.indexOf('/');
                    const folder = sampleId.substring(0, slash);
                    const filename = sampleId.substring(slash + 1);
                    const res = await fetch(`/api/rocks/${encodeURIComponent(folder)}/${encodeURIComponent(filename)}`);
                    if (!res.ok) {
                        const err = await res.json().catch(() => ({}));
                        throw new Error(err.detail || `HTTP ${res.status}`);
                    }
                    const blob = await res.blob();
                    const file = new File([blob], filename, { type: 'application/octet-stream' });

                    this.selectedFile = file;
                    TopoFlow.selectedFile = file;
                    this.selectedRockType = rock;
                    TopoFlow.selectedRockType = rock;
                    this.updateSelectedInfo(
                        meta?.name || sampleId,
                        `Cloud · φ = ${(meta.porosity * 100).toFixed(1)}% · ${meta.shape.join('×')}`
                    );
                    this.enableButtons();

                    // Clear other selection highlights
                    grid.querySelectorAll('.lib-sample__btn').forEach(b => {
                        b.textContent = 'Download & Select';
                        b.disabled = false;
                        b.classList.remove('btn--ghost');
                    });
                    btn.textContent = '✓ Selected';
                    btn.classList.add('btn--ghost');

                } catch (err) {
                    btn.textContent = 'Download & Select';
                    btn.disabled = false;
                    status.textContent = err.message;
                    status.style.color = 'var(--danger)';
                }
            });
        });
    },



    async selectCloudSample(card, samples) {
        document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');

        const sampleId = card.dataset.id;   // e.g. "MEC_Carbonate/mec_sample_a.npy"
        const meta = samples.find(s => s.id === sampleId);
        this.selectedRockType = meta?.rock_type || 'MEC';

        const statusDiv = document.createElement('div');
        statusDiv.style.cssText = 'font-size:0.75rem;margin-top:6px;color:var(--muted)';
        statusDiv.textContent = 'Downloading…';
        card.appendChild(statusDiv);

        try {
            // sampleId is "Folder/filename.npy" — split and encode each part
            const slash = sampleId.indexOf('/');
            const folder = sampleId.substring(0, slash);
            const filename = sampleId.substring(slash + 1);
            const url = `/api/rocks/${encodeURIComponent(folder)}/${encodeURIComponent(filename)}`;

            const res = await fetch(url);
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}: Download failed`);
            }
            const blob = await res.blob();
            const file = new File([blob], filename, { type: 'application/octet-stream' });

            this.selectedFile = file;
            TopoFlow.selectedFile = file;
            this.updateSelectedInfo(
                meta?.name || sampleId,
                `Cloud sample · φ = ${(meta.porosity * 100).toFixed(1)}% · ${meta.shape.join('×')}`
            );
            this.enableButtons();
            statusDiv.style.color = 'var(--primary)';
            statusDiv.textContent = '✓ Ready';
        } catch (err) {
            statusDiv.style.color = 'var(--danger)';
            statusDiv.textContent = err.message;
        }
    },


    // ── Controls ─────────────────────────────────────────────────────
    bindControls() {
        const btnPredict = document.getElementById('btn-predict');
        const btnVisualize = document.getElementById('btn-visualize');
        const btnClear = document.getElementById('btn-clear');
        const rockSelect = document.getElementById('rock-type-select');
        const qualitySelect = document.getElementById('viz-quality');

        btnPredict?.addEventListener('click', () => this.runPrediction());
        btnVisualize?.addEventListener('click', () => this.runVisualize());
        btnClear?.addEventListener('click', () => this.clearAll());

        rockSelect?.addEventListener('change', e => {
            this.selectedRockType = e.target.value;
            TopoFlow.selectedRockType = e.target.value;
        });

        qualitySelect?.addEventListener('change', e => {
            if (this.selectedFile) Viz3D.rerender(e.target.value);
        });

        document.querySelectorAll('input[name="model-mode"]').forEach(r => {
            r.addEventListener('change', e => { this.useHybrid = e.target.value === 'hybrid'; });
        });
    },

    updateSelectedInfo(name, meta, icon = '🪨') {
        const info = document.getElementById('selected-info');
        if (!info) return;
        info.classList.remove('hidden');
        document.getElementById('selected-name').textContent = name;
        document.getElementById('selected-meta').textContent = meta;
        document.getElementById('selected-icon').textContent = icon;
    },

    enableButtons() {
        document.getElementById('btn-predict')?.removeAttribute('disabled');
        document.getElementById('btn-visualize')?.removeAttribute('disabled');
    },

    clearAll() {
        this.selectedFile = null;
        TopoFlow.selectedFile = null;
        document.getElementById('selected-info')?.classList.add('hidden');
        document.getElementById('progress-section')?.classList.add('hidden');
        document.getElementById('results-section')?.classList.add('hidden');
        document.getElementById('btn-predict')?.setAttribute('disabled', '');
        document.getElementById('btn-visualize')?.setAttribute('disabled', '');
        document.getElementById('upload-filename')?.classList.add('hidden');
        const input = document.getElementById('file-input');
        if (input) input.value = '';
        document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('selected'));
    },

    // ── Visualize Only ───────────────────────────────────────────────
    async runVisualize() {
        const file = this.selectedFile || TopoFlow.selectedFile;
        if (!file) {
            alert('Please select a sample from the cloud library or upload a .npy file first.');
            return;
        }
        document.getElementById('results-section')?.classList.remove('hidden');
        const quality = document.getElementById('viz-quality')?.value || 'balanced';
        await Viz3D.render(file, quality);
    },

    // ── Run Prediction (SSE) ─────────────────────────────────────────
    async runPrediction() {
        const file = this.selectedFile || TopoFlow.selectedFile;
        if (!file) {
            alert('Please select a sample from the cloud library or upload a .npy file first.');
            return;
        }

        // Show progress, hide old results
        const progressSection = document.getElementById('progress-section');
        const resultsSection = document.getElementById('results-section');
        progressSection?.classList.remove('hidden');
        resultsSection?.classList.add('hidden');

        this.setProgress('Uploading file…', 5, '');

        // POST the file
        const formData = new FormData();
        formData.append('file', file);
        formData.append('rock_type', this.selectedRockType || TopoFlow.selectedRockType || 'MEC');
        formData.append('use_hybrid', this.useHybrid ? 'true' : 'false');

        let jobId;
        try {
            const res = await fetch('/api/predict', { method: 'POST', body: formData });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const data = await res.json();
            jobId = data.job_id;
        } catch (err) {
            this.setProgress('Error: ' + err.message, 0, '');
            return;
        }

        // Open SSE stream for progress
        const evtSource = new EventSource(`/api/predict/progress/${jobId}`);
        
        // Set a timeout to check job status if SSE doesn't connect
        let connectionTimeout = setTimeout(async () => {
            if (evtSource.readyState === EventSource.CONNECTING) {
                console.warn('SSE connection timeout, checking job status...');
                evtSource.close();
                try {
                    const statusRes = await fetch(`/api/predict/status/${jobId}`);
                    if (statusRes.ok) {
                        const status = await statusRes.json();
                        if (status.error) {
                            this.setProgress('Error', 0, status.error);
                            this._showError(status.error, jobId);
                        } else if (status.result) {
                            this.setProgress('Complete!', 100, '');
                            setTimeout(() => {
                                progressSection?.classList.add('hidden');
                                this.showResults(status.result);
                            }, 600);
                        }
                    }
                } catch (err) {
                    console.error('Failed to check job status:', err);
                    this._showError('Connection timeout - check server logs', jobId);
                }
            }
        }, 10000); // 10 second timeout

        evtSource.addEventListener('progress', (e) => {
            clearTimeout(connectionTimeout);
            const msg = JSON.parse(e.data);
            this.setProgress(msg.step, msg.pct, msg.detail || '');
        });

        evtSource.addEventListener('complete', (e) => {
            clearTimeout(connectionTimeout);
            evtSource.close();
            const msg = JSON.parse(e.data);
            this.setProgress('Complete!', 100, '');
            setTimeout(() => {
                progressSection?.classList.add('hidden');
                this.showResults(msg.result);
            }, 600);
        });

        evtSource.addEventListener('error', async (e) => {
            clearTimeout(connectionTimeout);
            evtSource.close();
            let errorMsg = 'unknown error';
            
            // Try to parse error from event data
            if (e.data) {
                try {
                    const msg = JSON.parse(e.data);
                    errorMsg = msg.error || 'unknown error';
                } catch (_) {
                    errorMsg = e.data || 'unknown error';
                }
            } else {
                // If no data, try to fetch job status from API
                try {
                    const statusRes = await fetch(`/api/predict/status/${jobId}`);
                    if (statusRes.ok) {
                        const status = await statusRes.json();
                        if (status.error) {
                            errorMsg = status.error;
                        } else {
                            errorMsg = 'SSE connection error - check server logs';
                        }
                    }
                } catch (fetchErr) {
                    console.error('Failed to fetch job status:', fetchErr);
                    errorMsg = 'Connection error - check server logs for details';
                }
            }
            
            console.error('SSE error event:', e);
            console.error('Error message:', errorMsg);
            
            this._showError(errorMsg, jobId);
        });

        evtSource.onerror = (e) => {
            clearTimeout(connectionTimeout);
            console.error('EventSource onerror:', e);
            evtSource.close();
            this._showError('SSE connection failed - check server logs', jobId);
        };
    },

    setProgress(step, pct, detail) {
        const fill = document.getElementById('progress-fill');
        const stepText = document.getElementById('progress-step-text');
        const pctText = document.getElementById('progress-pct-text');
        const detailText = document.getElementById('progress-detail');
        if (fill) fill.style.width = pct + '%';
        if (stepText) stepText.textContent = step;
        if (pctText) pctText.textContent = pct + '%';
        if (detailText) detailText.textContent = detail;
    },

    _showError(errorMsg, jobId) {
        this.setProgress('Inference error', 0, errorMsg);
        // Show error in results section too
        const resultsSection = document.getElementById('results-section');
        const progressSection = document.getElementById('progress-section');
        if (progressSection) progressSection.classList.add('hidden');
        if (resultsSection) {
            resultsSection.classList.remove('hidden');
            resultsSection.innerHTML = `
                <div class="card" style="border-color:var(--danger);padding:24px;text-align:center;">
                    <div style="font-size:2rem;margin-bottom:12px;">⚠️</div>
                    <div style="font-weight:600;color:var(--danger);margin-bottom:8px;">Prediction Failed</div>
                    <div style="color:var(--text);font-size:0.9rem;line-height:1.6;word-break:break-word;">${errorMsg}</div>
                    ${errorMsg.includes('Model file not found') ? `
                        <div style="margin-top:16px;padding:12px;background:rgba(255,255,255,0.05);border-radius:6px;font-size:0.85rem;color:var(--muted);">
                            <strong>Solution:</strong> Train a model first using the training scripts in <code>src/</code> directory.
                        </div>
                    ` : ''}
                    <div style="margin-top:12px;font-size:0.75rem;color:var(--muted);">
                        Job ID: ${jobId || 'N/A'}<br>
                        Check browser console (F12) and <code>topoflow.log</code> for more details.
                    </div>
                </div>
            `;
        }
    },

    // ── Show Results ─────────────────────────────────────────────────
    showResults(r) {
        const section = document.getElementById('results-section');
        section?.classList.remove('hidden');

        // Metric cards
        document.getElementById('res-si').textContent = r.permeability?.toExponential(3) || '—';
        document.getElementById('res-md').textContent = r.k_mdarcy?.toFixed(4) || '—';
        document.getElementById('res-log').textContent = r.log_k?.toFixed(2) || '—';

        if (r.baseline_k) {
            const bc = document.getElementById('baseline-card');
            bc?.classList.remove('hidden');
            document.getElementById('res-baseline').textContent = r.baseline_k.toExponential(3);
        }

        // Tier
        document.getElementById('tier-icon').textContent = r.icon || '🟡';
        document.getElementById('tier-name').textContent = r.tier || '—';
        document.getElementById('tier-name').style.color = r.color || 'var(--text)';
        document.getElementById('tier-quality').textContent = r.quality || '';

        // Scale marker
        const marker = document.getElementById('scale-marker');
        if (marker) marker.style.left = (r.scale_pct || 0) + '%';

        // Winner badge
        const wbc = document.getElementById('winner-badge-container');
        if (wbc) {
            const isWinner = r.log_k > -14;
            wbc.innerHTML = `
        <div class="winner-badge ${isWinner ? 'winner-badge--gnn' : 'winner-badge--physics'}">
          <div class="winner-badge__icon">${isWinner ? '🤖' : '📐'}</div>
          <div>
            <div class="winner-badge__title">${isWinner ? 'GNN REGIME' : 'PHYSICS REGIME'}</div>
            <div class="winner-badge__sub">
              ${isWinner ? 'High pore heterogeneity detected — GNN likely outperforms baseline.'
                    : 'Uniform pore structure — Kozeny-Carman is competitive.'}
            </div>
          </div>
        </div>`;
        }

        // Interpretation
        document.getElementById('interp-flow').textContent = r.flow || '—';
        document.getElementById('interp-analogy').textContent = r.analogy || '—';
        document.getElementById('interp-reservoir').textContent = r.reservoir || '—';
        document.getElementById('interp-porosity').textContent =
            `φ = ${((r.porosity || 0) * 100).toFixed(1)}% (${r.porosity_label || '—'}). ${r.porosity_note || ''}`;

        // Hybrid panel
        if (r.use_hybrid && r.correction_pct !== undefined) {
            const hp = document.getElementById('hybrid-panel');
            hp?.classList.remove('hidden');
            const dir = r.correction_pct > 0 ? 'lower' : 'higher';
            document.getElementById('hybrid-text').textContent =
                `GNN correction: ${Math.abs(r.correction_pct).toFixed(1)}% ${dir} than Kozeny-Carman baseline. ` +
                `Baseline log₁₀(K) = ${r.baseline_log_k?.toFixed(2)}.`;
        }

        // Trigger 3D visualization
        if (TopoFlow.selectedFile) {
            const quality = document.getElementById('viz-quality')?.value || 'balanced';
            Viz3D.render(TopoFlow.selectedFile, quality);
        }
    },
};
