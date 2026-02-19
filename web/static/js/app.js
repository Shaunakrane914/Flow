/**
 * app.js — SPA Router & Global State
 * Bio-Digital TopoFlow GNN
 * Hash routing: URL becomes #home, #predictor, #dashboard, #methodology
 */

const TopoFlow = {
    currentPage: 'home',
    selectedFile: null,
    selectedRockType: 'MEC',
    selectedJobId: null,
};

const VALID_PAGES = ['home', 'predictor', 'dashboard', 'methodology'];

// ── Page map: which module to initialize per page ──────────────────
const PAGE_INIT = {
    dashboard: () => typeof Dashboard !== 'undefined' && Dashboard.init(),
    predictor: () => typeof Predictor !== 'undefined' && Predictor.init(),
    home: () => { },
    methodology: () => initMethodologyTabs(),
};

// ── Router ─────────────────────────────────────────────────────────
async function navigate(page, updateHash = true) {
    if (!VALID_PAGES.includes(page)) page = 'home';

    // Update URL hash so browser address bar shows e.g. #dashboard
    if (updateHash && window.location.hash !== `#${page}`) {
        window.location.hash = page;
        return; // hashchange event will re-call navigate()
    }

    const outlet = document.getElementById('page-content');
    outlet.style.opacity = '0.5';

    try {
        const res = await fetch(`/fragment/${page}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        outlet.innerHTML = await res.text();
        outlet.style.opacity = '1';

        // Update active nav button
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.page === page);
        });

        TopoFlow.currentPage = page;
        (PAGE_INIT[page] || (() => { }))();

    } catch (err) {
        outlet.innerHTML = `<div class="card" style="text-align:center;padding:40px;">
      <div style="font-size:2rem;margin-bottom:12px;">⚠️</div>
      <div style="color:var(--danger);">Failed to load page: ${err.message}</div>
    </div>`;
        outlet.style.opacity = '1';
    }
}

// ── Methodology tab handler ────────────────────────────────────────
function initMethodologyTabs() {
    const tabs = document.querySelectorAll('#method-tabs .tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            document.querySelectorAll('[id^="tab-"]').forEach(p => {
                if (!p.closest('#method-tabs')) p.classList.remove('active');
            });
            const panel = document.getElementById(`tab-${tab.dataset.tab}`);
            if (panel) panel.classList.add('active');
        });
    });
}

// ── Bootstrap ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Wire nav buttons — just set the hash, hashchange handles the rest
    document.querySelectorAll('.nav-btn[data-page]').forEach(btn => {
        btn.addEventListener('click', () => {
            window.location.hash = btn.dataset.page;
        });
    });

    // Listen for hash changes (covers: nav clicks, back/forward, direct links)
    window.addEventListener('hashchange', () => {
        const page = window.location.hash.replace('#', '') || 'home';
        navigate(page, false);   // false = don't update hash again (already set)
    });

    // ── Event delegation for in-fragment [data-page] buttons ─────────
    // Inline <script> tags inside innerHTML fragments do NOT execute
    // (browser security). Use delegation on the outlet instead.
    document.getElementById('page-content').addEventListener('click', e => {
        const btn = e.target.closest('[data-page]:not(.nav-btn)');
        if (btn) {
            e.preventDefault();
            window.location.hash = btn.dataset.page;
        }
    });

    // On initial load — read hash from URL or default to home
    const initialPage = window.location.hash.replace('#', '') || 'home';
    navigate(initialPage, false);
});
