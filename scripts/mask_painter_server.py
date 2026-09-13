import argparse
import base64
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from mask_painter_layers import LAYERS, LayerStore


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lattice Mask Painter</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #151515;
      --panel: #202020;
      --line: #383838;
      --text: #f3f3f3;
      --muted: #aaa;
      --accent: #5ee36b;
      --danger: #ff6868;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      height: 100dvh;
      min-height: 480px;
      background: var(--bg);
      color: var(--text);
      font-family: Segoe UI, system-ui, sans-serif;
      display: grid;
      grid-template-columns: 1fr 280px;
      overflow: hidden;
    }
    main {
      min-width: 0;
      min-height: 0;
      padding: 18px;
    }
    .viewport {
      position: relative;
      width: 100%;
      height: 100%;
      overflow: hidden;
      border: 1px solid #444;
      background: #050505;
      touch-action: none;
      cursor: crosshair;
      user-select: none;
      overscroll-behavior: contain;
    }
    aside {
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      overflow-y: auto;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 18px;
      font-weight: 650;
    }
    .hint {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .stage {
      position: absolute;
      top: 0;
      left: 0;
      transform-origin: 0 0;
      will-change: transform;
    }
    canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      image-rendering: auto;
      cursor: inherit;
    }
    #image, #otherMasks, #cursor {
      pointer-events: none;
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 13px;
    }
    .label-line { display: flex; justify-content: space-between; gap: 8px; }
    .label-line > span { white-space: nowrap; font-variant-numeric: tabular-nums; }
    input[type="range"] { width: 100%; }
    .row.tools { grid-template-columns: 1fr 1fr 36px; }
    .zoom-controls {
      display: grid;
      grid-template-columns: 36px 1fr 36px 36px;
      align-items: center;
      gap: 6px;
    }
    .zoom-controls output {
      text-align: center;
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }
    .icon-button {
      width: 36px;
      height: 36px;
      padding: 8px;
      display: grid;
      place-items: center;
    }
    .icon-button img { width: 18px; height: 18px; filter: invert(1); }
    button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    button:disabled { opacity: 0.4; cursor: default; }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    button {
      min-height: 36px;
      border: 1px solid #474747;
      background: #292929;
      color: var(--text);
      border-radius: 6px;
      font: inherit;
      cursor: pointer;
    }
    button:hover { border-color: #666; background: #303030; }
    button.active { border-color: var(--accent); color: var(--accent); }
    button.danger { color: var(--danger); }
    button.primary {
      border-color: #58ba63;
      background: #24482a;
      color: #d9ffdc;
    }
    .status {
      min-height: 22px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    [hidden] { display: none !important; }
    .layer-heading, .layer-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 30px 30px;
      align-items: center;
      gap: 6px;
    }
    .layer-heading { font-size: 12px; color: var(--muted); margin-bottom: 6px; }
    .layer-heading span:not(:first-child) { text-align: center; }
    .layer-row { min-height: 42px; border-bottom: 1px solid var(--line); }
    .layer-choice { display: flex; align-items: center; gap: 7px; min-width: 0; cursor: pointer; }
    .layer-choice input { margin: 0; }
    .layer-choice span:last-child { overflow-wrap: anywhere; }
    .swatch { width: 10px; height: 18px; flex: 0 0 10px; border-radius: 2px; }
    .layer-row > input { justify-self: center; width: 16px; height: 16px; cursor: pointer; }
    .layer-row:has(input[type="radio"]:checked) { background: #303030; }
    .layer-heading img { width: 16px; height: 16px; filter: invert(0.7); }
    #layerPanel { border-top: 1px solid var(--line); padding-top: 12px; }
    kbd {
      padding: 1px 5px;
      border: 1px solid #555;
      border-bottom-width: 2px;
      border-radius: 4px;
      background: #171717;
      font-size: 11px;
    }
    @media (max-width: 700px) {
      body {
        height: auto;
        min-height: 100dvh;
        grid-template-columns: minmax(0, 1fr);
        grid-template-rows: min(60dvh, 540px) auto;
        overflow: auto;
      }
      main { padding: 10px; }
      aside { border-left: 0; border-top: 1px solid var(--line); overflow: visible; }
    }
  </style>
</head>
<body>
  <main>
    <div id="viewport" class="viewport" tabindex="0" aria-label="Lattice mask canvas">
      <div id="stage" class="stage">
        <canvas id="image"></canvas>
        <canvas id="otherMasks"></canvas>
        <canvas id="mask"></canvas>
        <canvas id="cursor"></canvas>
      </div>
    </div>
  </main>
  <aside>
    <section>
      <h1>Lattice Mask Painter</h1>
    </section>

    <section id="layerPanel" aria-label="Mask layers" hidden>
      <div class="layer-heading"><span>Layers</span><span title="Visible" aria-label="Visible"><img src="/icons/eye.svg" alt=""></span><span title="Include in combined mask" aria-label="Include in combined mask"><img src="/icons/layers.svg" alt=""></span></div>
      <div id="layerList"></div>
    </section>

    <div class="row tools">
      <button id="paint" class="active" title="Brush (B; hold Shift for horizontal/vertical strokes)" aria-pressed="true">Brush</button>
      <button id="erase" title="Eraser (E; hold Shift for horizontal/vertical strokes)" aria-pressed="false">Eraser</button>
      <button id="pan" class="icon-button" title="Pan (H, or hold Space and drag)" aria-label="Pan" aria-pressed="false"><img src="/icons/hand.svg" alt=""></button>
    </div>

    <div class="zoom-controls" role="group" aria-label="Zoom">
      <button id="zoomOut" class="icon-button" title="Zoom out" aria-label="Zoom out"><img src="/icons/zoom-out.svg" alt=""></button>
      <output id="zoomValue" for="zoom">100%</output>
      <button id="zoomIn" class="icon-button" title="Zoom in" aria-label="Zoom in"><img src="/icons/zoom-in.svg" alt=""></button>
      <button id="fit" class="icon-button" title="Fit to view (0)" aria-label="Fit to view"><img src="/icons/maximize.svg" alt=""></button>
    </div>
    <input id="zoom" aria-label="Zoom percentage" type="range" min="100" max="1200" step="1" value="100">

    <label>
      <span class="label-line">Brush size <span><span id="brushValue">18</span> px</span></span>
      <input id="brush" type="range" min="1" max="80" value="18">
    </label>

    <label>
      <span class="label-line">Overlay opacity <span><span id="opacityValue">55</span>%</span></span>
      <input id="opacity" type="range" min="10" max="90" value="55">
    </label>

    <div class="row">
      <button id="undo" title="Undo (Ctrl+Z)">Undo</button>
      <button id="redo" title="Redo (Ctrl+Y)">Redo</button>
    </div>
    <div class="row">
      <button id="loadExisting">Load mask</button>
      <button id="clear" class="danger">Clear</button>
    </div>
    <button id="save" class="primary" disabled>Save mask</button>

    <div id="status" class="status">Loading...</div>
  </aside>

  <script>
    const config = __PAINTER_CONFIG__;
    const viewport = document.getElementById("viewport");
    const stage = document.getElementById("stage");
    const imageCanvas = document.getElementById("image");
    const maskCanvas = document.getElementById("mask");
    const cursorCanvas = document.getElementById("cursor");
    const imageCtx = imageCanvas.getContext("2d");
    const otherCanvas = document.getElementById("otherMasks");
    const otherCtx = otherCanvas.getContext("2d");
    const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
    const cursorCtx = cursorCanvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const brushEl = document.getElementById("brush");
    const brushValue = document.getElementById("brushValue");
    const opacityEl = document.getElementById("opacity");
    const opacityValue = document.getElementById("opacityValue");
    const paintBtn = document.getElementById("paint");
    const eraseBtn = document.getElementById("erase");
    const panBtn = document.getElementById("pan");
    const zoomEl = document.getElementById("zoom");
    const zoomValue = document.getElementById("zoomValue");
    const saveBtn = document.getElementById("save");

    let mode = "paint";
    let ready = false;
    let drawing = false;
    let panning = false;
    let spaceHeld = false;
    let pointerInside = false;
    let activePointer = null;
    let panLast = null;
    let last = null;
    let axisLock = null;
    let brush = Number(brushEl.value);
    let opacity = Number(opacityEl.value) / 100;
    let undoStack = [];
    let redoStack = [];
    let revision = 0;
    let savedRevision = 0;
    let baseRevision = null;
    let loading = false;
    let saving = false;
    let activeLayer = 0;
    const layers = config.layers.map(layer => ({ ...layer, visible: true, included: true, canvas: document.createElement("canvas") }));
    const view = { scale: 1, fit: 1, x: 0, y: 0, width: 0, height: 0 };

    if (config.layered) {
      document.getElementById("layerPanel").hidden = false;
      document.getElementById("save").textContent = "Save layers";
      document.getElementById("loadExisting").textContent = "Reload layers";
      document.getElementById("clear").textContent = "Clear layer";
      for (const [index, layer] of layers.entries()) {
        const row = document.createElement("div");
        row.className = "layer-row";
        const choice = document.createElement("label");
        choice.className = "layer-choice";
        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = "activeLayer";
        radio.value = layer.id;
        radio.checked = index === 0;
        radio.addEventListener("change", () => selectLayer(index));
        layer.radio = radio;
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = `rgb(${layer.color.join(",")})`;
        const label = document.createElement("span");
        label.textContent = layer.label;
        choice.append(radio, swatch, label);
        const visible = document.createElement("input");
        visible.type = "checkbox";
        visible.checked = true;
        visible.title = `Show ${layer.label}`;
        visible.setAttribute("aria-label", visible.title);
        visible.addEventListener("change", () => { layer.visible = visible.checked; renderOtherLayers(); });
        layer.visibilityInput = visible;
        const included = document.createElement("input");
        included.type = "checkbox";
        included.checked = true;
        included.title = `Include ${layer.label} in combined mask`;
        included.setAttribute("aria-label", included.title);
        included.addEventListener("change", () => { finishPointer(); layer.included = included.checked; snapshot(); });
        layer.includeInput = included;
        row.append(choice, visible, included);
        document.getElementById("layerList").append(row);
      }
    }

    const img = new Image();
    img.onload = () => {
      for (const c of [imageCanvas, maskCanvas, otherCanvas, cursorCanvas, ...layers.map(layer => layer.canvas)]) {
        c.width = img.naturalWidth;
        c.height = img.naturalHeight;
      }
      stage.style.width = img.naturalWidth + "px";
      stage.style.height = img.naturalHeight + "px";
      imageCtx.drawImage(img, 0, 0);
      setMaskStyle();
      ready = true;
      resizeView(true);
      snapshot(false);
      saveBtn.disabled = false;
      status("Ready.");
      if (config.layered || new URLSearchParams(window.location.search).get("autoload") === "1") {
        document.getElementById("loadExisting").click();
      }
    };
    img.onerror = () => status("Frame could not be loaded.");
    img.src = "/frame";

    function status(text) {
      statusEl.textContent = text;
    }

    function setMaskStyle() {
      maskCanvas.style.opacity = String(opacity);
      otherCanvas.style.opacity = String(opacity);
      maskCtx.lineCap = "round";
      maskCtx.lineJoin = "round";
    }

    function syncActiveLayer() {
      const context = layers[activeLayer].canvas.getContext("2d");
      context.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      context.drawImage(maskCanvas, 0, 0);
    }

    function renderOtherLayers() {
      otherCtx.clearRect(0, 0, otherCanvas.width, otherCanvas.height);
      for (const [index, layer] of layers.entries()) {
        if (index !== activeLayer && layer.visible) otherCtx.drawImage(layer.canvas, 0, 0);
      }
      maskCanvas.style.visibility = layers[activeLayer].visible ? "visible" : "hidden";
    }

    function selectLayer(index) {
      if (!ready || loading) return;
      finishPointer();
      syncActiveLayer();
      activeLayer = index;
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      maskCtx.drawImage(layers[index].canvas, 0, 0);
      layers[index].radio.checked = true;
      renderOtherLayers();
      drawCursor(null);
    }

    function setLoading(value) {
      loading = value;
      saveBtn.disabled = value;
      document.getElementById("loadExisting").disabled = value;
      for (const input of document.querySelectorAll("#layerPanel input")) input.disabled = value;
    }

    function setMode(next) {
      finishPointer();
      mode = next;
      for (const [button, value] of [[paintBtn, "paint"], [eraseBtn, "erase"], [panBtn, "pan"]]) {
        button.classList.toggle("active", mode === value);
        button.setAttribute("aria-pressed", String(mode === value));
      }
      updateCursor();
      drawCursor(null);
    }

    function updateCursor() {
      viewport.style.cursor = panning ? "grabbing" : mode === "pan" || spaceHeld ? "grab" : "crosshair";
    }

    function applyView() {
      const width = maskCanvas.width * view.scale;
      const height = maskCanvas.height * view.scale;
      view.x = width <= view.width ? (view.width - width) / 2 : Math.min(0, Math.max(view.width - width, view.x));
      view.y = height <= view.height ? (view.height - height) / 2 : Math.min(0, Math.max(view.height - height, view.y));
      stage.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
      const percent = Math.round(view.scale / view.fit * 100);
      zoomEl.value = percent;
      zoomValue.value = `${percent}%`;
      document.getElementById("zoomOut").disabled = percent <= 100;
      document.getElementById("zoomIn").disabled = percent >= 1200;
    }

    function resizeView(reset = false) {
      if (!ready) return;
      finishPointer();
      const zoom = reset ? 1 : view.scale / view.fit;
      const centerX = reset ? maskCanvas.width / 2 : (view.width / 2 - view.x) / view.scale;
      const centerY = reset ? maskCanvas.height / 2 : (view.height / 2 - view.y) / view.scale;
      view.width = viewport.clientWidth;
      view.height = viewport.clientHeight;
      view.fit = Math.min(view.width / maskCanvas.width, view.height / maskCanvas.height) * 0.96;
      view.scale = view.fit * zoom;
      view.x = view.width / 2 - centerX * view.scale;
      view.y = view.height / 2 - centerY * view.scale;
      applyView();
      drawCursor(null);
    }

    function zoomTo(scale, anchorX = view.width / 2, anchorY = view.height / 2) {
      if (!ready || activePointer !== null) return;
      const x = (anchorX - view.x) / view.scale;
      const y = (anchorY - view.y) / view.scale;
      view.scale = Math.max(view.fit, Math.min(view.fit * 12, scale));
      view.x = anchorX - x * view.scale;
      view.y = anchorY - y * view.scale;
      applyView();
      drawCursor(null);
    }

    function pos(event) {
      const rect = maskCanvas.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left) * maskCanvas.width / rect.width,
        y: (event.clientY - rect.top) * maskCanvas.height / rect.height,
      };
    }

    function drawAt(a, b) {
      maskCtx.save();
      maskCtx.globalCompositeOperation = mode === "paint" ? "source-over" : "destination-out";
      maskCtx.strokeStyle = `rgb(${layers[activeLayer].color.join(",")})`;
      maskCtx.fillStyle = maskCtx.strokeStyle;
      maskCtx.lineWidth = brush;
      maskCtx.beginPath();
      if (!b) {
        maskCtx.arc(a.x, a.y, brush / 2, 0, Math.PI * 2);
        maskCtx.fill();
      } else {
        maskCtx.moveTo(a.x, a.y);
        maskCtx.lineTo(b.x, b.y);
        maskCtx.stroke();
      }
      maskCtx.restore();
    }

    function constrainStroke(p, shiftKey) {
      if (!shiftKey) {
        if (axisLock) {
          axisLock = null;
          last = null;
        }
        return p;
      }
      if (!axisLock) axisLock = { origin: { ...(last || p) }, axis: null };
      const dx = p.x - axisLock.origin.x;
      const dy = p.y - axisLock.origin.y;
      if (!axisLock.axis) {
        // Ignore a little screen-space jitter before choosing the stroke axis.
        if (Math.hypot(dx, dy) * view.scale < 3) return axisLock.origin;
        axisLock.axis = Math.abs(dx) >= Math.abs(dy) ? "x" : "y";
      }
      return axisLock.axis === "x"
        ? { x: p.x, y: axisLock.origin.y }
        : { x: axisLock.origin.x, y: p.y };
    }

    function drawCursor(p) {
      cursorCtx.clearRect(0, 0, cursorCanvas.width, cursorCanvas.height);
      if (!p || mode === "pan" || spaceHeld || panning) return;
      cursorCtx.save();
      cursorCtx.strokeStyle = mode === "paint" ? `rgb(${layers[activeLayer].color.join(",")})` : "#ff6868";
      cursorCtx.lineWidth = 1.5 / view.scale;
      cursorCtx.beginPath();
      cursorCtx.arc(p.x, p.y, brush / 2, 0, Math.PI * 2);
      cursorCtx.stroke();
      cursorCtx.restore();
    }

    function snapshot(changed = true) {
      if (!ready) return;
      syncActiveLayer();
      // Store alpha only; RGB is fixed per layer, including at antialiased edges.
      const masks = layers.map(layer => {
        const rgba = layer.canvas.getContext("2d").getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;
        const alpha = new Uint8Array(rgba.length / 4);
        for (let i = 0; i < alpha.length; i++) alpha[i] = rgba[i * 4 + 3];
        return alpha;
      });
      undoStack.push({ masks, activeLayer, included: layers.map(layer => layer.included),
        legacy: config.layered ? null : maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height) });
      if (undoStack.length > 60) undoStack.shift();
      redoStack = [];
      if (changed) revision++;
    }

    function restore(data) {
      for (const [index, layer] of layers.entries()) {
        const context = layer.canvas.getContext("2d");
        const rgba = context.createImageData(maskCanvas.width, maskCanvas.height);
        for (let i = 0; i < data.masks[index].length; i++) {
          rgba.data.set(layer.color, i * 4);
          rgba.data[i * 4 + 3] = data.masks[index][i];
        }
        context.putImageData(rgba, 0, 0);
        layer.included = data.included[index];
        if (layer.includeInput) layer.includeInput.checked = layer.included;
      }
      activeLayer = data.activeLayer;
      if (layers[activeLayer].radio) layers[activeLayer].radio.checked = true;
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      if (data.legacy) maskCtx.putImageData(data.legacy, 0, 0);
      else maskCtx.drawImage(layers[activeLayer].canvas, 0, 0);
      renderOtherLayers();
      revision++;
    }

    function undo() {
      if (!ready || loading) return;
      finishPointer();
      if (undoStack.length <= 1) return;
      redoStack.push(undoStack.pop());
      restore(undoStack[undoStack.length - 1]);
    }

    function redo() {
      if (!ready || loading) return;
      finishPointer();
      if (!redoStack.length) return;
      const data = redoStack.pop();
      undoStack.push(data);
      restore(data);
    }

    function finishPointer() {
      if (drawing) snapshot();
      drawing = false;
      panning = false;
      last = null;
      axisLock = null;
      panLast = null;
      const pointer = activePointer;
      activePointer = null;
      if (pointer !== null && viewport.hasPointerCapture(pointer)) viewport.releasePointerCapture(pointer);
      updateCursor();
    }

    viewport.addEventListener("pointerdown", (event) => {
      if (!ready || loading || activePointer !== null || (event.button !== 0 && event.button !== 1)) return;
      const p = pos(event);
      const pan = event.button === 1 || spaceHeld || mode === "pan";
      if (!pan && (p.x < 0 || p.y < 0 || p.x >= maskCanvas.width || p.y >= maskCanvas.height)) return;
      event.preventDefault();
      viewport.focus({ preventScroll: true });
      activePointer = event.pointerId;
      viewport.setPointerCapture(event.pointerId);
      if (pan) {
        panning = true;
        panLast = { x: event.clientX, y: event.clientY };
        drawCursor(null);
      } else {
        layers[activeLayer].visible = true;
        if (layers[activeLayer].visibilityInput) layers[activeLayer].visibilityInput.checked = true;
        renderOtherLayers();
        drawing = true;
        last = p;
        axisLock = event.shiftKey ? { origin: { ...p }, axis: null } : null;
        drawAt(last, null);
        drawCursor(last);
      }
      updateCursor();
    });
    viewport.addEventListener("pointermove", (event) => {
      if (!ready || (activePointer !== null && activePointer !== event.pointerId)) return;
      if (panning) {
        view.x += event.clientX - panLast.x;
        view.y += event.clientY - panLast.y;
        panLast = { x: event.clientX, y: event.clientY };
        applyView();
        return;
      }
      const raw = pos(event);
      const p = drawing ? constrainStroke(raw, event.shiftKey) : raw;
      drawCursor(p);
      if (!drawing) return;
      drawAt(last || p, last ? p : null);
      last = p;
    });
    for (const eventName of ["pointerup", "pointercancel", "lostpointercapture"]) {
      viewport.addEventListener(eventName, (event) => {
        if (event.pointerId === activePointer) finishPointer();
      });
    }
    viewport.addEventListener("pointerenter", () => { pointerInside = true; });
    viewport.addEventListener("pointerleave", () => { pointerInside = false; drawCursor(null); });
    viewport.addEventListener("auxclick", (event) => { if (event.button === 1) event.preventDefault(); });
    viewport.addEventListener("wheel", (event) => {
      event.preventDefault();
      const rect = viewport.getBoundingClientRect();
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? view.height : 1);
      zoomTo(view.scale * Math.exp(-delta * 0.002), event.clientX - rect.left - viewport.clientLeft, event.clientY - rect.top - viewport.clientTop);
      if (ready) drawCursor(drawing ? last : pos(event));
    }, { passive: false });
    new ResizeObserver(() => resizeView()).observe(viewport);
    zoomEl.addEventListener("input", () => zoomTo(view.fit * Number(zoomEl.value) / 100));
    document.getElementById("zoomIn").addEventListener("click", () => zoomTo(view.scale * 1.25));
    document.getElementById("zoomOut").addEventListener("click", () => zoomTo(view.scale / 1.25));
    document.getElementById("fit").addEventListener("click", () => resizeView(true));

    brushEl.addEventListener("input", () => {
      brush = Number(brushEl.value);
      brushValue.textContent = brush;
    });
    opacityEl.addEventListener("input", () => {
      opacity = Number(opacityEl.value) / 100;
      opacityValue.textContent = opacityEl.value;
      setMaskStyle();
    });

    paintBtn.addEventListener("click", () => setMode("paint"));
    eraseBtn.addEventListener("click", () => setMode("erase"));
    panBtn.addEventListener("click", () => setMode("pan"));
    document.getElementById("undo").addEventListener("click", undo);
    document.getElementById("redo").addEventListener("click", redo);
    document.getElementById("clear").addEventListener("click", () => {
      if (!ready || loading) return;
      finishPointer();
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      snapshot();
    });
    document.getElementById("loadExisting").addEventListener("click", async () => {
      if (!ready || loading || saving) return;
      finishPointer();
      if (config.layered) {
        if (revision !== savedRevision && !confirm("Discard unsaved changes and reload saved layers?")) return;
        setLoading(true);
        status("Loading layers...");
        try {
          const response = await fetch("/layers", { cache: "no-store" });
          const result = await response.json();
          if (!response.ok || !result.ok) throw new Error(result.error || "Could not load layers");
          const images = await Promise.all(layers.map(layer => new Promise((resolve, reject) => {
            const image = new Image();
            image.onload = () => resolve(image);
            image.onerror = () => reject(new Error(`Could not load ${layer.label}`));
            image.src = result.layers[layer.id];
          })));
          for (const [index, layer] of layers.entries()) {
            const context = layer.canvas.getContext("2d");
            context.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
            context.drawImage(images[index], 0, 0);
            const data = context.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
            for (let i = 0; i < data.data.length; i += 4) {
              const selected = data.data[i] || data.data[i + 1] || data.data[i + 2];
              data.data.set(layer.color, i);
              data.data[i + 3] = selected ? 255 : 0;
            }
            context.putImageData(data, 0, 0);
            layer.included = result.included.includes(layer.id);
            layer.includeInput.checked = layer.included;
          }
          maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
          maskCtx.drawImage(layers[activeLayer].canvas, 0, 0);
          renderOtherLayers();
          baseRevision = result.revision;
          snapshot();
          savedRevision = revision;
          status("Loaded saved layers.");
        } catch (error) {
          status(`Load failed:\n${error.message}`);
        } finally {
          setLoading(false);
          if (baseRevision === null) saveBtn.disabled = true;
        }
        return;
      }
      const loaded = new Image();
      loaded.onload = () => {
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        maskCtx.globalAlpha = 1;
        maskCtx.drawImage(loaded, 0, 0, maskCanvas.width, maskCanvas.height);
        const data = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
        for (let i = 0; i < data.data.length; i += 4) {
          const v = data.data[i] || data.data[i + 1] || data.data[i + 2];
          data.data[i] = 70;
          data.data[i + 1] = 255;
          data.data[i + 2] = 85;
          data.data[i + 3] = v > 0 ? 255 : 0;
        }
        maskCtx.putImageData(data, 0, 0);
        snapshot();
        savedRevision = revision;
        status("Loaded existing mask.");
      };
      loaded.onerror = () => status("No existing mask found yet.");
      loaded.src = "/mask?t=" + Date.now();
    });

    saveBtn.addEventListener("click", async () => {
      if (!ready || loading || saving || (config.layered && baseRevision === null)) return;
      finishPointer();
      syncActiveLayer();
      const payload = config.layered ? {
        base_revision: baseRevision,
        layers: Object.fromEntries(layers.map(layer => [layer.id, layer.canvas.toDataURL("image/png")])),
        included: layers.filter(layer => layer.included).map(layer => layer.id),
      } : { image: maskCanvas.toDataURL("image/png") };
      const savingRevision = revision;
      saving = true;
      saveBtn.disabled = true;
      document.getElementById("loadExisting").disabled = true;
      status("Saving...");
      try {
        const response = await fetch("/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const result = await response.json();
        if (result.ok) { savedRevision = savingRevision; baseRevision = result.revision || null; }
        status(result.ok ? config.layered
          ? `Saved 3 layers + combined mask.\n${result.mask}`
          : `Saved:\n${result.mask}\n${result.overlay}` : `Save failed:\n${result.error}`);
      } catch (error) {
        status(`Save failed:\n${error.message}`);
      } finally {
        saving = false;
        saveBtn.disabled = false;
        document.getElementById("loadExisting").disabled = false;
      }
    });

    window.addEventListener("keydown", (event) => {
      if (event.key === "Shift" && drawing && !axisLock && last) {
        axisLock = { origin: { ...last }, axis: null };
      } else if (event.code === "Space" && (pointerInside || !event.target.closest("button, input, textarea, select"))) {
        event.preventDefault();
        if (drawing) finishPointer();
        spaceHeld = true;
        updateCursor();
        drawCursor(null);
      } else if (event.ctrlKey && event.key.toLowerCase() === "z") {
        event.preventDefault();
        undo();
      } else if (event.ctrlKey && event.key.toLowerCase() === "y") {
        event.preventDefault();
        redo();
      } else if (event.key.toLowerCase() === "b") {
        setMode("paint");
      } else if (event.key.toLowerCase() === "e") {
        setMode("erase");
      } else if (event.key.toLowerCase() === "h") {
        setMode("pan");
      } else if (event.key === "0") {
        resizeView(true);
      } else if (event.key === "[") {
        brushEl.value = Math.max(Number(brushEl.min), Number(brushEl.value) - 2);
        brushEl.dispatchEvent(new Event("input"));
      } else if (event.key === "]") {
        brushEl.value = Math.min(Number(brushEl.max), Number(brushEl.value) + 2);
        brushEl.dispatchEvent(new Event("input"));
      }
    });
    window.addEventListener("keyup", (event) => {
      if (event.key === "Shift" && !event.shiftKey && axisLock) {
        axisLock = null;
        // Resume at the physical pointer without drawing a connector to it.
        last = null;
        drawCursor(null);
      } else if (event.code === "Space") {
        spaceHeld = false;
        updateCursor();
      }
    });
    window.addEventListener("blur", () => {
      spaceHeld = false;
      finishPointer();
      drawCursor(null);
    });
    window.addEventListener("beforeunload", (event) => {
      if (revision !== savedRevision || drawing || saving) {
        event.preventDefault();
        event.returnValue = "";
      }
    });
  </script>
</body>
</html>
"""


class PainterHandler(BaseHTTPRequestHandler):
    server_version = "MaskPainter/2.0"

    def do_GET(self):
        route = urlparse(self.path).path
        store = getattr(self.server, "layer_store", None)
        if route == "/":
            config = {"layered": store is not None, "layers": LAYERS if store else [LAYERS[0]]}
            html = HTML.replace("__PAINTER_CONFIG__", json.dumps(config))
            self.send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
        elif route == "/layers" and store:
            try:
                self.send_json(store.load())
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=500)
        elif route == "/frame":
            self.send_file(self.server.frame_path)
        elif route in {"/icons/hand.svg", "/icons/zoom-in.svg", "/icons/zoom-out.svg", "/icons/maximize.svg", "/icons/eye.svg", "/icons/layers.svg"}:
            self.send_file(Path(__file__).with_name("mask_painter_assets") / Path(route).name)
        elif route == "/mask":
            if store:
                self.send_file(store.directory / "combined_mask.png")
            elif self.server.mask_path.exists():
                self.send_file(self.server.mask_path)
            else:
                self.send_error(404, "Mask not found")
        else:
            self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/save":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 32 * 1024 * 1024:
                raise ValueError("Invalid request size")
            payload = self.rfile.read(length).decode("utf-8")
            data = json.loads(payload)
            store = getattr(self.server, "layer_store", None)
            if store:
                self.send_json(store.save(data))
                return
            prefix, encoded = data["image"].split(",", 1)
            if "base64" not in prefix:
                raise ValueError("Expected a base64 data URL")
            raw = base64.b64decode(encoded)
            arr = np.frombuffer(raw, dtype=np.uint8)
            rgba = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if rgba is None:
                raise ValueError("Could not decode PNG")
            alpha = rgba[:, :, 3] if rgba.ndim == 3 and rgba.shape[2] == 4 else rgba
            mask = (alpha > 0).astype(np.uint8) * 255
            self.server.mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.server.mask_path), mask)
            self.save_overlay(mask)
            self.send_json(
                {
                    "ok": True,
                    "mask": str(self.server.mask_path),
                    "overlay": str(self.server.overlay_path),
                }
            )
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)

    def save_overlay(self, mask):
        frame = cv2.imread(str(self.server.frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            return
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = np.zeros_like(frame)
        color[:, :, 1] = 255
        color[:, :, 2] = 70
        blended = cv2.addWeighted(frame, 0.62, color, 0.38, 0)
        overlay = frame.copy()
        overlay[mask > 0] = blended[mask > 0]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1, lineType=cv2.LINE_AA)
        self.server.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.server.overlay_path), overlay)

    def send_file(self, path):
        if not path.exists():
            self.send_error(404, f"File not found: {path}")
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_bytes(path.read_bytes(), content_type)

    def send_json(self, data, status=200):
        self.send_bytes(
            json.dumps(data, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def send_bytes(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))


def parse_args():
    parser = argparse.ArgumentParser(description="Serve a local brush mask painter.")
    parser.add_argument("--frame", default="work/frames/00000.jpg")
    parser.add_argument("--mask", default="work/init_masks/lattice_manual_mask.png")
    parser.add_argument("--overlay", default="work/init_masks/lattice_manual_overlay.png")
    parser.add_argument("--layers-dir", help="Enable independent near/far/reflection layers; --mask seeds near only on first launch")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main():
    args = parse_args()
    frame_path = Path(args.frame).resolve()
    if not frame_path.exists():
        raise FileNotFoundError(frame_path)

    server = ThreadingHTTPServer((args.host, args.port), PainterHandler)
    server.frame_path = frame_path
    server.mask_path = Path(args.mask).resolve()
    server.overlay_path = Path(args.overlay).resolve()
    server.layer_store = LayerStore(args.layers_dir, frame_path, server.mask_path) if args.layers_dir else None
    print(f"Mask painter: http://{args.host}:{args.port}/")
    print(f"Frame: {server.frame_path}")
    print(f"Mask output: {server.mask_path}")
    if server.layer_store:
        print(f"Layer outputs: {server.layer_store.directory}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
