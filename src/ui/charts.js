function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  return {
    ctx,
    width: rect.width,
    height: rect.height,
  };
}

function extent(values, paddingRatio = 0.1) {
  const min = Math.min(...values);
  const max = Math.max(...values);

  if (Number.isNaN(min) || Number.isNaN(max)) {
    return [0, 1];
  }

  if (Math.abs(max - min) < 1e-12) {
    return [min - 0.5, max + 0.5];
  }

  const span = max - min;
  return [min - span * paddingRatio, max + span * paddingRatio];
}

function drawAxes(ctx, dims, xLabel, yLabel) {
  const { left, top, right, bottom } = dims;

  ctx.strokeStyle = "#8d9aad";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.stroke();

  ctx.fillStyle = "#516178";
  ctx.font = "12px 'IBM Plex Sans', 'Avenir Next', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(xLabel, (left + right) / 2, bottom + 32);

  ctx.save();
  ctx.translate(left - 40, (top + bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

function drawGrid(ctx, dims) {
  const { left, top, right, bottom } = dims;
  ctx.strokeStyle = "rgba(90, 112, 141, 0.2)";
  ctx.lineWidth = 1;

  const n = 4;
  for (let i = 1; i <= n; i += 1) {
    const x = left + ((right - left) * i) / (n + 1);
    const y = top + ((bottom - top) * i) / (n + 1);

    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
  }
}

export function drawScatter(canvas, points) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const dims = {
    left: 62,
    right: width - 20,
    top: 20,
    bottom: height - 48,
  };

  drawGrid(ctx, dims);
  drawAxes(
    ctx,
    dims,
    "Drift MSE (lower is better)",
    "Stress regression MSE (lower is better)",
  );

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const [xMin, xMax] = extent(xs);
  const [yMin, yMax] = extent(ys);

  function xToPx(x) {
    return dims.left + ((x - xMin) / (xMax - xMin)) * (dims.right - dims.left);
  }

  function yToPx(y) {
    return dims.bottom - ((y - yMin) / (yMax - yMin)) * (dims.bottom - dims.top);
  }

  ctx.font = "12px 'IBM Plex Sans', 'Avenir Next', sans-serif";
  for (const p of points) {
    const x = xToPx(p.x);
    const y = yToPx(p.y);

    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = "#233247";
    ctx.textAlign = "left";
    ctx.fillText(p.label, x + 8, y - 8);
  }
}

export function drawEquity(canvas, series, stressMarkers) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const dims = {
    left: 62,
    right: width - 20,
    top: 20,
    bottom: height - 48,
  };

  drawGrid(ctx, dims);
  drawAxes(ctx, dims, "Time", "Equity");

  const allY = series.flatMap((s) => s.values);
  const [yMin, yMax] = extent(allY, 0.08);
  const maxT = Math.max(...series.map((s) => s.values.length - 1));

  function xToPx(t) {
    return dims.left + (t / Math.max(1, maxT)) * (dims.right - dims.left);
  }

  function yToPx(y) {
    return dims.bottom - ((y - yMin) / (yMax - yMin)) * (dims.bottom - dims.top);
  }

  ctx.strokeStyle = "rgba(207, 79, 55, 0.12)";
  ctx.lineWidth = 1;
  for (let i = 0; i < stressMarkers.length; i += 1) {
    const x = xToPx(stressMarkers[i]);
    ctx.beginPath();
    ctx.moveTo(x, dims.top);
    ctx.lineTo(x, dims.bottom);
    ctx.stroke();
  }

  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let t = 0; t < s.values.length; t += 1) {
      const x = xToPx(t);
      const y = yToPx(s.values[t]);
      if (t === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.stroke();
  }

  drawLegend(ctx, series, width, 10);
}

function drawLegend(ctx, series, xStart, yStart) {
  let x = xStart - 280;
  const y = yStart;

  ctx.font = "12px 'IBM Plex Sans', 'Avenir Next', sans-serif";
  for (const item of series) {
    ctx.fillStyle = item.color;
    ctx.fillRect(x, y, 14, 3);

    ctx.fillStyle = "#22344d";
    ctx.textAlign = "left";
    ctx.fillText(item.label, x + 18, y + 4);

    x += ctx.measureText(item.label).width + 44;
  }
}
