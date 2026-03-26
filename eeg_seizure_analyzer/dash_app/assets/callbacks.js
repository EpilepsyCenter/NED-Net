// Clientside callbacks are registered via dash.clientside_callback in Python.
// This file is loaded automatically by Dash from the assets/ folder.

if (!window.dash_clientside) {
    window.dash_clientside = {};
}

window.dash_clientside.ned = {
    // Sync slider -> input
    syncSliderToInput: function(sliderVal) {
        return sliderVal;
    },
    // Sync input -> slider
    syncInputToSlider: function(inputVal) {
        return inputVal;
    }
};
