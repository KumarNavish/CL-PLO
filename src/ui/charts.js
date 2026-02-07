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

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [0, 1];
  }

  if (Math.abs(max - min) < 1e-12) {
    return [min - 1, max + 1];
  }

  const span = max - min;
  return [min - span * paddingRatio, max + span * paddingRatio];
}

function drawAxes(ctx, dims, xLabel, yLabel) {
  const { left, top, right, bottom } = dims;

  ctx.strokeStyle = "#8b99af";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.stroke();

  ctx.fillStyle = "#4d5f78";
  ctx.font = "12px 'IBM Plex Sans', 'Avenir Next', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(xLabel, (left + right) / 2, bottom + 30);

  ctx.save();
  ctx.translate(left - 38, (top + bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

function drawGrid(ctx, dims, ticks = 4) {
  const { left, top, right, bottom } = dims;
  ctx.strokeStyle = "rgba(83, 106, 137, 0.18)";
  ctx.lineWidth = 1;

  for (let i = 1; i <= ticks; i += 1) {
    const x = left + ((right - left) * i) / (ticks + 1);
    const y = top + ((bottom - top) * i) / (ticks + 1);

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

function drawLegend(ctx, legendItems, width, y) {
  let x = width - 250;
  ctx.font = "12px 'IBM Plex Sans', 'Avenir Next', sans-serif";

  for (const item of legendItems) {
    ctx.fillStyle = item.color;
    ctx.fillRect(x, y, 14, 3);

    ctx.fillStyle = "#24354d";
    ctx.textAlign = "left";
    ctx.fillText(item.label, x + 18, y + 4);

    x += ctx.measureText(item.label).width + 46;
  }
}

function pct(x) {
  return `${x.toFixed(1)}%`;
}

export function drawImpactBars(canvas, impact) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const dims = {
    left: 64,
    right: width - 18,
    top: 24,
    bottom: height - 46,
  };

  const values = impact.flatMap((g) => [g.anchor, g.proj, 0]);
  const [rawMin, rawMax] = extent(values, 0.15);
  const yMin = Math.min(rawMin, 0);
  const yMax = Math.max(rawMax, 0);

  function yToPx(y) {
    return dims.bottom - ((y - yMin) / (yMax - yMin)) * (dims.bottom - dims.top);
  }

  function xCenter(i, n) {
    return dims.left + ((i + 0.5) / n) * (dims.right - dims.left);
  }

  drawGrid(ctx, dims, 4);
  drawAxes(ctx, dims, "Metrics", "Delta vs naive (percentage points)");

  const zeroY = yToPx(0);
  ctx.strokeStyle = "#3d4f6b";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(dims.left, zeroY);
  ctx.lineTo(dims.right, zeroY);
  ctx.stroke();

  const groupWidth = (dims.right - dims.left) / impact.length;
  const barWidth = Math.min(22, groupWidth * 0.22);

  for (let i = 0; i < impact.length; i += 1) {
    const group = impact[i];
    const cx = xCenter(i, impact.length);

    drawBar(ctx, cx - barWidth * 0.7, barWidth, group.anchor, yToPx, zeroY, "#1f7b63");
    drawBar(ctx, cx + barWidth * 0.7, barWidth, group.proj, yToPx, zeroY, "#2758ad");

    ctx.fillStyle = "#22354d";
    ctx.font = "12px 'IBM Plex Sans', 'Avenir Next', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(group.label, cx, dims.bottom + 16);

    labelValue(ctx, cx - barWidth * 0.7, yToPx(group.anchor), group.anchor, "#1f7b63", zeroY);
    labelValue(ctx, cx + barWidth * 0.7, yToPx(group.proj), group.proj, "#2758ad", zeroY);
  }

  drawLegend(
    ctx,
    [
      { label: "Anchor", color: "#1f7b63" },
      { label: "Anchor + Projection", color: "#2758ad" },
    ],
    width,
    10,
  );
}

function drawBar(ctx, centerX, width, value, yToPx, zeroY, color) {
  const y = yToPx(value);
  const top = Math.min(y, zeroY);
  const h = Math.max(1, Math.abs(y - zeroY));

  ctx.fillStyle = color;
  ctx.fillRect(centerX - width / 2, top, width, h);
}

function labelValue(ctx, x, y, value, color, zeroY) {
  ctx.fillStyle = color;
  ctx.font = "11px 'IBM Plex Sans', 'Avenir Next', sans-serif";
  ctx.textAlign = "center";

  if (y < zeroY) {
    ctx.fillText(pct(value), x, y - 6);
  } else {
    ctx.fillText(pct(value), x, y + 14);
  }
}

export function drawEquity(canvas, series, stressMarkers) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const dims = {
    left: 62,
    right: width - 20,
    top: 22,
    bottom: height - 46,
  };

  drawGrid(ctx, dims, 4);
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

  ctx.strokeStyle = "rgba(191, 71, 55, 0.14)";
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
