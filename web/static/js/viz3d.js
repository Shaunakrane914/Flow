/**
 * viz3d.js — Browser-Side 3D Volume Renderer
 * Uses server-downsampled JSON + Plotly.js Volume trace
 */

const Viz3D = {
    lastVoxelData: null,
    _canvas2DPatched: false,

    _patchCanvas2D() {
        if (Viz3D._canvas2DPatched) return;
        Viz3D._canvas2DPatched = true;
        try {
            var proto = HTMLCanvasElement.prototype;
            var orig = proto.getContext;
            proto.getContext = function(type, options) {
                if (type === '2d') {
                    try {
                        options = options && typeof options === 'object'
                            ? Object.assign({}, options, { willReadFrequently: true })
                            : { willReadFrequently: true };
                    } catch (_) {}
                }
                return orig.call(this, type, options);
            };
        } catch (_) {}
    },

    /**
     * Send .npy file to server for downsampling, then render in Plotly.
     * @param {File|Blob} file  — the .npy file
     * @param {string} quality  — "low" | "balanced" | "hd"
     */
    async render(file, quality = 'balanced') {
        const container = document.getElementById('viz-3d');
        if (!container) return;

        container.innerHTML = `
      <div class="flex-center" style="height:100%;flex-direction:column;gap:12px;">
        <div class="spinner"></div>
        <div class="text-muted" style="font-size:0.85rem;">Processing 3D structure…</div>
      </div>`;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('quality', quality);

        let voxelData;
        try {
            const res = await fetch('/api/visualize', { method: 'POST', body: formData });
            if (!res.ok) throw new Error(`Server error ${res.status}`);
            voxelData = await res.json();
        } catch (err) {
            container.innerHTML = `<div class="flex-center" style="height:100%;color:var(--danger);">
        Visualization failed: ${err.message}</div>`;
            return;
        }

        Viz3D.lastVoxelData = voxelData;
        
        // Check if we have data to render
        if (!voxelData.x || voxelData.x.length === 0) {
            console.warn('No voxel data received:', voxelData);
            container.innerHTML = `<div class="flex-center" style="height:100%;color:var(--danger);">
                <div style="text-align:center;">
                    <div style="font-size:2rem;margin-bottom:12px;">⚠️</div>
                    <div style="font-weight:600;margin-bottom:8px;">No pore data to visualize</div>
                    <div style="font-size:0.85rem;color:var(--muted);">The chunk appears to be mostly solid rock (porosity too low or threshold too high).</div>
                    <div style="font-size:0.85rem;color:var(--muted);margin-top:8px;">Porosity: ${(voxelData.porosity * 100).toFixed(2)}%</div>
                    <div style="font-size:0.75rem;color:var(--muted);margin-top:8px;">Rendered voxels: ${voxelData.rendered_voxels || 0}</div>
                </div>
            </div>`;
            return;
        }
        
        // Check if Plotly is available
        if (typeof Plotly === 'undefined') {
            container.innerHTML = `<div class="flex-center" style="height:100%;color:var(--danger);">
                <div style="text-align:center;">
                    <div style="font-size:2rem;margin-bottom:12px;">⚠️</div>
                    <div style="font-weight:600;margin-bottom:8px;">Plotly.js not loaded</div>
                    <div style="font-size:0.85rem;color:var(--muted);">Please refresh the page.</div>
                </div>
            </div>`;
            return;
        }
        
        Viz3D._plotVolume(voxelData);
        Viz3D._updateStats(voxelData);
    },

    /** Re-render at a different quality without re-uploading */
    async rerender(quality) {
        if (!TopoFlow.selectedFile) return;
        await Viz3D.render(TopoFlow.selectedFile, quality);
    },

    _plotVolume(d) {
        // "Deep Water" colorscale - makes pores look like actual water flowing through rock
        // Black (transparent) → Deep Blue → Bright Cyan → White (for largest vugs)
        // This is instantly readable: blue = water/fluid, black = solid rock
        const deepWaterColorscale = [
            [0.0, 'rgba(0,0,0,0)'],        // Transparent (solid rock)
            [0.1, 'rgba(0,0,0,0)'],        // Still transparent
            [0.2, 'rgba(0,20,60,0.3)'],    // Deep blue (narrow throats)
            [0.4, 'rgba(0,50,150,0.6)'],   // Medium blue (medium pores)
            [0.7, 'rgba(0,150,255,0.8)'],  // Bright cyan (large pores)
            [1.0, 'rgba(255,255,255,1.0)'] // White (largest vugs)
        ];

        const { x, y, z, values, shape, sparse_mode } = d;
        const [sx, sy, sz] = shape;

        // Debug logging
        console.log('Plotting volume:', {
            numPoints: x.length,
            shape: shape,
            valueRange: values.length > 0 ? [Math.min(...values), Math.max(...values)] : 'empty',
            porosity: d.porosity,
            sparseMode: sparse_mode
        });

        // Use scatter3d for sparse data, volume for full grid
        let volumeTrace;
        if (sparse_mode || x.length < 1000) {
            // Sparse mode: use scatter3d markers
            console.log('Using scatter3d mode (sparse data)');
            volumeTrace = {
                type: 'scatter3d',
                mode: 'markers',
                x, y, z,
                marker: {
                    size: 5,  // Slightly larger for better visibility
                    color: values,
                    colorscale: deepWaterColorscale,
                    cmin: 0.1,
                    cmax: 1.0,
                    showscale: true,
                    colorbar: {
                        title: { text: 'Pore Size', font: { color: '#00BFFF', size: 12 } },
                        tickfont: { color: '#888', size: 10 },
                        tickvals: [0.2, 0.5, 0.8],
                        ticktext: ['Narrow', 'Medium', 'Large Vug'],
                        len: 0.6,
                        thickness: 12,
                        x: 1.02,
                    },
                    opacity: 0.8,  // More visible
                },
                name: 'Pores',
                hoverinfo: 'skip',
            };
        } else {
            // Full grid mode: use volume trace
            console.log('Using volume trace mode (full grid)');
            volumeTrace = {
                type: 'volume',
                x, y, z,
                value: values,
                isomin: 0.1,
                isomax: 1.0,
                opacity: 0.4,  // Increased for better visibility
                surface: { count: 35 },  // Increased from 20 to 35 for smoother surfaces
                colorscale: deepWaterColorscale,
                showscale: true,
                colorbar: {
                    title: { text: 'Pore Size', font: { color: '#00BFFF', size: 12 } },
                    tickfont: { color: '#888', size: 10 },
                    tickvals: [0.2, 0.5, 0.8],
                    ticktext: ['Narrow', 'Medium', 'Large Vug'],
                    len: 0.6,
                    thickness: 12,
                    x: 1.02,
                },
                caps: { x: { show: false }, y: { show: false }, z: { show: false } },
            };
        }

        // INLET / OUTLET labels (more visible with water theme)
        const labelZ = sz / 2;
        const labelTrace = {
            type: 'scatter3d',
            x: [0, sx],
            y: [sy / 2, sy / 2],
            z: [labelZ, labelZ],
            mode: 'text',
            text: ['INLET', 'OUTLET'],
            textposition: 'top center',
            textfont: { size: 14, color: '#00BFFF', family: 'Arial Black' },  // Bright cyan, bold
            hoverinfo: 'none',
            showlegend: false,
        };

        // Flow direction arrow (cone) - blue to match water theme
        const coneTrace = {
            type: 'cone',
            x: [sx * 0.2], y: [sy / 2], z: [sz / 2],
            u: [sx * 0.3], v: [0], w: [0],
            sizemode: 'absolute',
            sizeref: 3,
            anchor: 'tail',
            colorscale: [[0, '#0066CC'], [1, '#00BFFF']],  // Deep blue to cyan
            showscale: false,
            hoverinfo: 'none',
            showlegend: false,
        };

        const layout = {
            scene: {
                xaxis: { visible: false },
                yaxis: { visible: false },
                zaxis: { visible: false },
                bgcolor: '#0A0A0A',  // Very dark background (almost black) for contrast
                camera: { 
                    eye: { x: 1.5, y: 1.5, z: 1.0 },
                    center: { x: sx/2, y: sy/2, z: sz/2 }
                },
            },
            paper_bgcolor: '#0E1117',
            plot_bgcolor: '#0A0A0A',  // Dark background for water contrast
            margin: { l: 0, r: 0, b: 0, t: 0 },
            height: 500,
        };

        // Apply willReadFrequently before Plotly creates any canvas (silences Chrome Canvas2D warning)
        Viz3D._patchCanvas2D();

        try {
            Plotly.newPlot('viz-3d', [volumeTrace, labelTrace, coneTrace], layout, {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['toImage'],
                // Reduce Canvas2D getImageData readbacks (browser performance hint)
                plotGlPixelRatio: 1,
                toImageButtonOptions: { format: 'png', scale: 1 },
            }).catch(err => {
                console.error('Plotly rendering error:', err);
                // Fallback to scatter3d if volume fails
                const scatterTrace = {
                    type: 'scatter3d',
                    mode: 'markers',
                    x, y, z,
                    marker: {
                        size: 3,
                        color: values,
                        colorscale: deepWaterColorscale,
                        showscale: true,
                        colorbar: volumeTrace.colorbar,
                        opacity: 0.6
                    },
                    name: 'Pores'
                };
                Plotly.newPlot('viz-3d', [scatterTrace, labelTrace, coneTrace], layout, {
                    responsive: true,
                    displayModeBar: true,
                });
            });
        } catch (err) {
            console.error('Failed to create Plotly plot:', err);
            const container = document.getElementById('viz-3d');
            if (container) {
                container.innerHTML = `<div class="flex-center" style="height:100%;color:var(--danger);">
                    <div style="text-align:center;">
                        <div style="font-size:2rem;margin-bottom:12px;">⚠️</div>
                        <div style="font-weight:600;margin-bottom:8px;">Visualization Error</div>
                        <div style="font-size:0.85rem;color:var(--muted);">${err.message}</div>
                    </div>
                </div>`;
            }
        }
    },

    _updateStats(d) {
        const statsBar = document.getElementById('viz-stats');
        if (statsBar) {
            statsBar.style.display = 'flex';
            document.getElementById('viz-rendered').textContent = d.rendered_voxels.toLocaleString();
            document.getElementById('viz-porosity').textContent = (d.porosity * 100).toFixed(1) + '%';
            document.getElementById('viz-step').textContent = `1/${d.step}`;
        }
    },
};
