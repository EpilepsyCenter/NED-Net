/**
 * NED-Net Video <-> EEG sync engine.
 *
 * Draws a CSS-based vertical playhead over a Plotly graph that tracks
 * the HTML5 video's currentTime.  Uses an absolute-positioned div
 * overlay so it never conflicts with Plotly's own shape management.
 * Also supports clicking on the graph to seek the video.
 */
(function () {
    "use strict";

    var PLAYHEAD_COLOR = "#f85149";
    var PLAYHEAD_WIDTH = 2;

    // Known video-graph pairings
    var PAIRS = {
        "viewer-video-player": "viewer-graph",
        "sz-insp-video":       "sz-insp-eeg-graph",
        "tr-review-video":     "tr-review-graph",
    };

    var _activePair = null;
    var _raf = null;
    var _playheadEl = null;

    function _getPlotlyDiv(wrapperId) {
        // Dash dcc.Graph wraps plotly in a child div with class "js-plotly-plot"
        var wrapper = document.getElementById(wrapperId);
        if (!wrapper) return null;
        // Check if wrapper itself is the plotly div
        if (wrapper._fullLayout) return wrapper;
        // Otherwise search children
        var plotly = wrapper.querySelector(".js-plotly-plot");
        if (plotly && plotly._fullLayout) return plotly;
        // Try any child with _fullLayout
        var children = wrapper.querySelectorAll("div");
        for (var i = 0; i < children.length; i++) {
            if (children[i]._fullLayout) return children[i];
        }
        return null;
    }

    function _getPlotArea(plotlyDiv) {
        // Get the plot area rect from the Plotly SVG
        var plotBg = plotlyDiv.querySelector(".draglayer .nsewdrag");
        if (!plotBg) plotBg = plotlyDiv.querySelector(".nsewdrag");
        if (!plotBg) return null;
        return plotBg.getBoundingClientRect();
    }

    function _timeToPixel(plotlyDiv, time) {
        var xaxis = plotlyDiv._fullLayout.xaxis;
        if (!xaxis) return null;

        var range = xaxis.range;
        if (!range || range.length < 2) return null;

        var plotArea = _getPlotArea(plotlyDiv);
        if (!plotArea) return null;

        var fraction = (time - range[0]) / (range[1] - range[0]);
        if (fraction < 0 || fraction > 1) return null;

        return plotArea.left + fraction * plotArea.width;
    }

    function _ensurePlayhead() {
        if (_playheadEl) return _playheadEl;
        _playheadEl = document.createElement("div");
        _playheadEl.id = "ned-playhead";
        _playheadEl.style.cssText =
            "position:fixed;" +
            "width:" + PLAYHEAD_WIDTH + "px;" +
            "background:" + PLAYHEAD_COLOR + ";" +
            "pointer-events:none;" +
            "z-index:9999;" +
            "display:none;" +
            "opacity:0.85;";
        document.body.appendChild(_playheadEl);
        return _playheadEl;
    }

    function _tick() {
        if (!_activePair) return;

        var video = document.getElementById(_activePair.videoId);
        var plotlyDiv = _getPlotlyDiv(_activePair.graphId);

        if (!video || !plotlyDiv) {
            _raf = requestAnimationFrame(_tick);
            return;
        }

        var t = video.currentTime;
        var ph = _ensurePlayhead();
        var px = _timeToPixel(plotlyDiv, t);

        if (px !== null) {
            var plotArea = _getPlotArea(plotlyDiv);
            if (plotArea) {
                ph.style.left = px + "px";
                ph.style.top = plotArea.top + "px";
                ph.style.height = plotArea.height + "px";
                ph.style.display = "block";
            }
        } else {
            ph.style.display = "none";
        }

        _raf = requestAnimationFrame(_tick);
    }

    function _onGraphClick(data) {
        if (!_activePair) return;
        if (data.points && data.points.length > 0) {
            var clickTime = data.points[0].x;
            var video = document.getElementById(_activePair.videoId);
            if (video && typeof clickTime === "number") {
                video.currentTime = clickTime;
            }
        }
    }

    function _startPair(videoId, graphId) {
        _stopCurrent();

        _activePair = { videoId: videoId, graphId: graphId };
        _raf = requestAnimationFrame(_tick);

        // Bind click-to-seek
        setTimeout(function () {
            var plotlyDiv = _getPlotlyDiv(graphId);
            if (plotlyDiv && plotlyDiv.on) {
                plotlyDiv.on("plotly_click", _onGraphClick);
            }
        }, 1500);
    }

    function _stopCurrent() {
        if (_raf) {
            cancelAnimationFrame(_raf);
            _raf = null;
        }
        if (_playheadEl) {
            _playheadEl.style.display = "none";
        }
        if (_activePair) {
            var plotlyDiv = _getPlotlyDiv(_activePair.graphId);
            if (plotlyDiv && plotlyDiv.removeListener) {
                plotlyDiv.removeListener("plotly_click", _onGraphClick);
            }
        }
        _activePair = null;
    }

    // Poll for video elements
    function _scan() {
        for (var videoId in PAIRS) {
            var video = document.getElementById(videoId);
            if (video) {
                if (_activePair && _activePair.videoId === videoId) return;
                _startPair(videoId, PAIRS[videoId]);
                return;
            }
        }
        // Clean up if active video no longer exists
        if (_activePair && !document.getElementById(_activePair.videoId)) {
            _stopCurrent();
        }
    }

    setInterval(_scan, 1000);

    // Expose for manual seeking
    window.nedVideoSync = {
        seek: function (videoId, time) {
            var video = document.getElementById(videoId);
            if (video && time != null) {
                video.currentTime = time;
            }
        },
    };
})();
