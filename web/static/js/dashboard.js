/**
 * dashboard.js — Scientific Validation Dashboard
 * Fetches /api/dashboard-data and renders 4 Plotly.js charts.
 * All charts use the Bio-Digital palette.
 */

const Dashboard = {
    data: null,

    _patchCanvas2D() {
        if (window._topoflowCanvas2DPatched) return;
        window._topoflowCanvas2DPatched = true;
        try {
            var proto = HTMLCanvasElement.prototype;
            var orig = proto.getContext;
            proto.getContext = function(type, options) {
                if (type === '2d') {
                    try {
                        options = (options && typeof options === 'object') ? Object.assign({}, options, { willReadFrequently: true }) : { willReadFrequently: true };
                    } catch (_) {}
                }
                return orig.apply(this, [type, options]);
            };
        } catch (_) {}
    },

    async init() {
        if (this.data) {
            this._renderAll();
            return;
        }
        try {
            const res = await fetch('/api/dashboard-data');
            this.data = await res.json();
            this._renderAll();
            this._populateTable();
        } catch (err) {
            console.error('Dashboard fetch failed:', err);
        }
    },

    _renderAll() {
        const d = this.data;
        if (!d) return;
        this._patchCanvas2D();
        this._renderMSE(d);
        this._renderImprovement(d);
        this._renderThreshold(d);
        this._renderDonut(d);
    },

    // ── Shared layout defaults ───────────────────────────────────────
    _baseLayout(overrides = {}) {
        return {
            paper_bgcolor: '#161B22',
            plot_bgcolor: '#161B22',
            font: { family: 'Inter, sans-serif', color: '#E0E0E0', size: 12 },
            margin: { l: 48, r: 16, t: 20, b: 60 },
            showlegend: true,
            legend: { font: { color: '#888' } },
            xaxis: {
                gridcolor: '#21262D',
                linecolor: '#30363D',
                tickfont: { color: '#888' },
                zerolinecolor: '#30363D',
            },
            yaxis: {
                gridcolor: '#21262D',
                linecolor: '#30363D',
                tickfont: { color: '#888' },
                zerolinecolor: '#30363D',
            },
            ...overrides,
        };
    },

    _plotConfig() {
        return {
            responsive: true,
            displayModeBar: false,
        };
    },

    // ── Chart 1: Grouped MSE Bar ─────────────────────────────────────
    _renderMSE(d) {
        const el = document.getElementById('chart-mse');
        if (!el) return;

        const traces = [
            {
                type: 'bar',
                name: 'Baseline (KC)',
                x: d.rocks,
                y: d.baseline_mse,
                marker: { color: '#2D5BFF', opacity: 0.85 },
            },
            {
                type: 'bar',
                name: 'GNN',
                x: d.rocks,
                y: d.gnn_mse,
                marker: { color: '#00FF9D', opacity: 0.85 },
            },
        ];

        const layout = this._baseLayout({
            barmode: 'group',
            yaxis: {
                title: { text: 'Test MSE (log₁₀ K)', font: { color: '#888', size: 11 } },
                gridcolor: '#21262D',
                tickfont: { color: '#888' },
            },
        });

        Plotly.newPlot('chart-mse', traces, layout, this._plotConfig());
    },

    // ── Chart 2: Improvement % Bar ───────────────────────────────────
    _renderImprovement(d) {
        const el = document.getElementById('chart-improvement');
        if (!el) return;

        const colors = d.gnn_wins.map((win, i) =>
            win ? '#00FF9D' : '#484f58'
        );

        const text = d.improvement.map((v, i) =>
            d.gnn_wins[i] ? `+${v.toFixed(1)}%` : 'Physics Wins'
        );

        const traces = [{
            type: 'bar',
            x: d.rocks,
            y: d.improvement,
            marker: { color: colors },
            text: text,
            textposition: 'outside',
            textfont: { color: '#E0E0E0', size: 11 },
        }];

        const layout = this._baseLayout({
            yaxis: {
                title: { text: 'Improvement % (GNN vs Baseline)', font: { color: '#888', size: 11 } },
                gridcolor: '#21262D',
                tickfont: { color: '#888' },
                range: [-10, 55],
            },
        });

        Plotly.newPlot('chart-improvement', traces, layout, this._plotConfig());
    },

    // ── Chart 3: Threshold Scatter ───────────────────────────────────
    _renderThreshold(d) {
        const el = document.getElementById('chart-threshold');
        if (!el) return;

        const markerColors = d.gnn_wins.map(win => win ? '#00FF9D' : '#2D5BFF');
        const markerSize = d.samples.map(n => Math.sqrt(n) * 1.8);

        const scatter = {
            type: 'scatter',
            mode: 'markers+text',
            x: d.cv_values,
            y: d.improvement,
            text: d.rocks,
            textposition: 'top center',
            textfont: { size: 10, color: '#E0E0E0' },
            marker: {
                color: markerColors,
                size: markerSize,
                opacity: 0.85,
                line: { width: 1, color: '#30363D' },
            },
            name: 'Rocks',
        };

        // Threshold vertical line at Cv = 1.5
        const thresholdLine = {
            type: 'scatter',
            mode: 'lines',
            x: [1.5, 1.5],
            y: [-15, 55],
            line: { color: 'rgba(224,224,224,0.4)', dash: 'dash', width: 2 },
            name: 'Threshold (Cv = 1.5)',
        };

        // Zero improvement line
        const zeroLine = {
            type: 'scatter',
            mode: 'lines',
            x: [0, 4],
            y: [0, 0],
            line: { color: 'rgba(136,136,136,0.4)', dash: 'dot', width: 1 },
            name: 'Break-even',
        };

        const layout = this._baseLayout({
            xaxis: {
                title: { text: 'Pore Heterogeneity (Cv = σ / μ)', font: { color: '#888', size: 11 } },
                gridcolor: '#21262D',
                tickfont: { color: '#888' },
                range: [0, 4],
            },
            yaxis: {
                title: { text: 'GNN Improvement % over Baseline', font: { color: '#888', size: 11 } },
                gridcolor: '#21262D',
                tickfont: { color: '#888' },
                range: [-20, 60],
            },
            annotations: [
                {
                    x: 0.9, y: 48,
                    text: '📐 Physics Regime',
                    showarrow: false,
                    font: { color: '#2D5BFF', size: 11 },
                },
                {
                    x: 2.8, y: 48,
                    text: '🤖 GNN Regime',
                    showarrow: false,
                    font: { color: '#00FF9D', size: 11 },
                },
            ],
        });

        Plotly.newPlot('chart-threshold', [scatter, thresholdLine, zeroLine], layout, this._plotConfig());
    },

    // ── Chart 4: Sample Distribution Donut ──────────────────────────
    _renderDonut(d) {
        const el = document.getElementById('chart-donut');
        if (!el) return;

        const palette = ['#00FF9D', '#2D5BFF', '#484f58', '#30363D', '#888888'];

        const traces = [{
            type: 'pie',
            labels: d.rocks,
            values: d.samples,
            hole: 0.55,
            marker: {
                colors: palette,
                line: { color: '#0E1117', width: 2 },
            },
            textfont: { color: '#E0E0E0', size: 11 },
            hovertemplate: '%{label}: %{value} samples<extra></extra>',
        }];

        const layout = {
            paper_bgcolor: '#161B22',
            plot_bgcolor: '#161B22',
            font: { family: 'Inter, sans-serif', color: '#E0E0E0', size: 11 },
            margin: { l: 0, r: 0, t: 20, b: 0 },
            showlegend: true,
            legend: {
                orientation: 'v',
                font: { color: '#888', size: 10 },
                x: 1, y: 0.5,
            },
            annotations: [{
                text: `<b>1,231</b><br>samples`,
                x: 0.5, y: 0.5,
                font: { size: 13, color: '#E0E0E0' },
                showarrow: false,
            }],
        };

        Plotly.newPlot('chart-donut', traces, layout, this._plotConfig());
    },

    // ── Results Table ────────────────────────────────────────────────
    _populateTable() {
        const tbody = document.getElementById('results-tbody');
        if (!tbody || !this.data) return;

        const d = this.data;
        tbody.innerHTML = d.rocks.map((rock, i) => {
            const win = d.gnn_wins[i];
            return `<tr>
        <td>${rock}</td>
        <td>${d.samples[i]}</td>
        <td>${d.cv_values[i].toFixed(2)}</td>
        <td>${d.baseline_mse[i].toFixed(3)}</td>
        <td>${d.gnn_mse[i].toFixed(3)}</td>
        <td class="${win ? 'td-gnn' : 'td-physics'}">${win ? '🤖 GNN' : '📐 Physics'}</td>
        <td class="${win ? 'td-gnn' : 'td-physics'}">${win ? '+' + d.improvement[i].toFixed(1) + '%' : '—'}</td>
      </tr>`;
        }).join('');
    },
};
