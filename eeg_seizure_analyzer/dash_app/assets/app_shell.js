/* Desktop conventions for the NED-Net window.
 *
 * - Starts in dark mode when the system is set to dark (and follows changes).
 * - Ctrl+1..9 (Cmd on macOS) switch sections; Ctrl/Cmd+O opens a recording.
 * - The settings panel of two-pane pages can be resized (drag its left edge,
 *   double-click to reset) and hidden (toolbar button, Alt+Cmd/Ctrl+0).
 * - window.nednet.* is what the native menu (desktop.py) calls.
 */
(function () {
    "use strict";

    // Shortcut number -> sidebar section
    var SHORTCUTS = {
        "1": "upload",
        "2": "viewer",
        "3": "seizures",
        "4": "spikes",
        "5": "training",
        "6": "training_spikes",
        "7": "ml_datasets",
        "8": "analysis",
        "9": "results"
    };

    function byId(id) { return document.getElementById(id); }

    // Run fn once the element exists (Dash renders after page load).
    function whenPresent(id, fn, timeoutMs) {
        var waited = 0, step = 100, limit = timeoutMs || 15000;
        (function poll() {
            var el = byId(id);
            if (el) { fn(el); return; }
            waited += step;
            if (waited < limit) { setTimeout(poll, step); }
        })();
    }

    function go(tabId) {
        var item = byId("section-" + tabId);
        if (item) { item.click(); }
    }

    // Open a recording: go to Load, then press its "Load File" (or
    // "Load Multiple...") button if the start screen is showing.
    function open(kind) {
        go("upload");
        var btn = kind === "multiple" ? "landing-load-project-btn" : "landing-load-file-btn";
        whenPresent(btn, function (el) { el.click(); }, 3000);
    }

    function isDark() {
        var t = byId("theme-toggle");
        return !!(t && t.checked);
    }

    function setDark(wantDark) {
        var t = byId("theme-toggle");
        if (t && t.checked !== wantDark) { t.click(); }
    }

    function toggleTheme() { setDark(!isDark()); }

    // ── Settings panel: show/hide and width (kept across pages) ──────
    var root = document.documentElement;
    var PANEL_MIN = 260, PANEL_MAX = 640;

    function store(key, value) {
        try { localStorage.setItem(key, value); } catch (e) { /* private mode */ }
    }
    function recall(key) {
        try { return localStorage.getItem(key); } catch (e) { return null; }
    }

    // Plotly graphs re-fit their width on a window resize event
    function refitGraphs() {
        setTimeout(function () { window.dispatchEvent(new Event("resize")); }, 30);
    }

    function setPanelHidden(hidden) {
        root.classList.toggle("panel-hidden", hidden);
        var btn = byId("panel-toggle");
        if (btn) {
            btn.classList.toggle("active", !hidden);
            btn.title = hidden ? "Show the settings panel" : "Hide the settings panel";
        }
        store("ned-panel-hidden", hidden ? "1" : "0");
        refitGraphs();
    }

    function togglePanel() { setPanelHidden(!root.classList.contains("panel-hidden")); }

    function setPanelWidth(px) {
        px = Math.max(PANEL_MIN, Math.min(PANEL_MAX, Math.round(px)));
        root.style.setProperty("--ned-panel-w", px + "px");
        return px;
    }

    // Restore the last width and visibility
    var savedW = parseInt(recall("ned-panel-w"), 10);
    if (savedW) { setPanelWidth(savedW); }
    if (recall("ned-panel-hidden") === "1") {
        root.classList.add("panel-hidden");
        whenPresent("panel-toggle", function () { setPanelHidden(true); });
    }

    document.addEventListener("click", function (e) {
        if (!e.target.closest) { return; }
        if (e.target.closest("#panel-toggle")) { togglePanel(); }
        // The NED-Net name in the sidebar: the Load page (start page, or the
        // open recording's overview with "Back to start")
        if (e.target.closest("#brand-home")) { go("upload"); }
    });

    // Drag the handle on the panel's left edge
    document.addEventListener("mousedown", function (e) {
        var handle = e.target.closest && e.target.closest(".page-panel-resizer");
        if (!handle || e.button !== 0) { return; }
        e.preventDefault();
        var split = handle.parentElement;
        var right = split.getBoundingClientRect().right;
        handle.classList.add("dragging");
        root.classList.add("ned-resizing");
        var width = null;
        function onMove(ev) { width = setPanelWidth(right - ev.clientX); }
        function onUp() {
            document.removeEventListener("mousemove", onMove);
            document.removeEventListener("mouseup", onUp);
            handle.classList.remove("dragging");
            root.classList.remove("ned-resizing");
            if (width) { store("ned-panel-w", String(width)); }
            refitGraphs();
        }
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
    });

    // Double-click the handle: back to the default width
    document.addEventListener("dblclick", function (e) {
        if (e.target.closest && e.target.closest(".page-panel-resizer")) {
            root.style.removeProperty("--ned-panel-w");
            store("ned-panel-w", "");
            refitGraphs();
        }
    });

    window.nednet = { go: go, open: open, toggleTheme: toggleTheme,
                      togglePanel: togglePanel };

    // Follow the system appearance
    var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
    if (media) {
        whenPresent("theme-toggle", function () {
            if (media.matches) { setDark(true); }
        });
        var onChange = function (e) { setDark(e.matches); };
        if (media.addEventListener) { media.addEventListener("change", onChange); }
        else if (media.addListener) { media.addListener(onChange); }
    }

    // Keyboard shortcuts
    var isMac = /Mac/.test(navigator.platform || navigator.userAgent);
    document.addEventListener("keydown", function (e) {
        var mod = isMac ? e.metaKey : e.ctrlKey;
        // Alt+Cmd+0 (Ctrl+Alt+0): show/hide the settings panel, as in macOS apps
        if (mod && e.altKey && !e.shiftKey && e.code === "Digit0") {
            e.preventDefault();
            togglePanel();
            return;
        }
        if (!mod || e.altKey || e.shiftKey) { return; }
        var key = (e.key || "").toLowerCase();
        if (SHORTCUTS[key]) {
            e.preventDefault();
            go(SHORTCUTS[key]);
        } else if (key === "o") {
            e.preventDefault();
            open("file");
        }
    });
})();
