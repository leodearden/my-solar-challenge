/**
 * Shared Plotly chart renderer used by home and fleet results pages.
 *
 * Expects Plotly.js to be loaded before this script.
 */
"use strict";

var SOLAR_CHART_CONFIG = { responsive: true, displayModeBar: true, scrollZoom: false };

/**
 * Render a Plotly chart from a JSON string into a DOM element.
 *
 * @param {string} elementId - ID of the target DOM element.
 * @param {string} jsonStr   - JSON-encoded Plotly figure (data + layout).
 */
function renderChart(elementId, jsonStr) {
    if (!jsonStr || jsonStr === '{}') return;
    var el = document.getElementById(elementId);
    if (!el) return;
    try {
        var fig = JSON.parse(jsonStr);
        Plotly.newPlot(elementId, fig.data, fig.layout, SOLAR_CHART_CONFIG);
    } catch (e) {
        console.error('Failed to render chart ' + elementId, e);
    }
}
