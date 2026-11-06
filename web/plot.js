/**
 * Stacked live panels: multi-trace lines, and a bar chart for RMSF.
 *
 * 🔴 NO PLOTTING LIBRARY, BECAUSE THE PAGE HAS NO BUILD STEP AND A CDN SCRIPT
 * IS A DEPENDENCY A READER CANNOT AUDIT. Everything a running simulation needs
 * to show is a few hundred points against time plus one bar per residue, which
 * is this file; a charting library would be the largest thing on the page by an
 * order of magnitude and the only part a user could not read.
 *
 * 🔴 EACH PANEL SCALES TO ITS OWN RANGE. Potential energy is tens of thousands,
 * RMSD is hundredths of a nanometre. One shared axis flattens every trace but
 * the largest into the frame edge - which is what makes a combined plot of
 * these five quantities useless rather than merely ugly.
 */

function tokens() {
  const style = getComputedStyle(document.body);
  return {
    ink: style.getPropertyValue("--ink").trim() || "#111",
    line: style.getPropertyValue("--line").trim() || "#ccc",
    muted: style.getPropertyValue("--muted").trim() || "#777",
  };
}

function formatShort(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "–";
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude >= 100000 || magnitude < 0.001) return value.toExponential(2);
  if (magnitude >= 100) return value.toFixed(1);
  return value.toFixed(4);
}

/** One panel: `series` are {key, label, colour}. */
export class Panel {
  constructor(canvas, options) {
    this.canvas = canvas;
    this.title = options.title;
    this.unit = options.unit ?? "";
    this.series = options.series;
    this.kind = options.kind ?? "line";
    this.labels = options.labels ?? null;
    this.rows = [];
    this.bars = null;
  }

  push(row) { this.rows.push(row); }
  setBars(values) { this.bars = values; }
  clear() { this.rows = []; this.bars = null; }

  draw() {
    const canvas = this.canvas;
    // The backing store is sized from devicePixelRatio every draw: a canvas
    // sized once in CSS pixels is blurry on any retina display, and one sized
    // only at load is blurry after the window moves between monitors.
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (width === 0 || height === 0) return;
    if (canvas.width !== Math.round(width * ratio)) canvas.width = Math.round(width * ratio);
    if (canvas.height !== Math.round(height * ratio)) canvas.height = Math.round(height * ratio);
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const colour = tokens();

    // 🔴 THE FONT IS SET BEFORE THE PADDING IS CHOSEN, BECAUSE THE PADDING
    // DEPENDS ON MEASURED TEXT. The heading and the legend used to be drawn at
    // the same x on the same line, so the four-way RMSD panel - the one panel
    // this interface exists for - rendered "RMSD / nm" over "complex" and
    // neither was readable.
    context.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    const heading = this.unit === "" ? this.title : `${this.title} / ${this.unit}`;
    const headingWidth = context.measureText(heading).width;
    const entryWidth = (definition) => 22 + context.measureText(definition.label).width;
    const legendWidth = this.series.length > 1
      ? this.series.reduce((total, d) => total + entryWidth(d), 0) : 0;

    const pad = { left: 58, right: 10, top: 20, bottom: 20 };
    // 🔴 AND IT WRAPS RATHER THAN CLIPS. `.panels` is a grid down to 19rem, at
    // which width the heading plus a four-series legend cannot fit on one line
    // however it is packed - and a dropped or truncated legend entry is worse
    // than a taller header, because it reads as a different series.
    const legendBeside = legendWidth === 0
      || pad.left + headingWidth + 14 + legendWidth <= width - pad.right;
    if (!legendBeside) pad.top = 32;
    this.legendStart = legendBeside ? pad.left + headingWidth + 14 : pad.left;
    this.legendY = legendBeside ? 6 : 18;
    this.legendLimit = width - pad.right;

    const plotWidth = width - pad.left - pad.right;
    const plotHeight = height - pad.top - pad.bottom;
    context.fillStyle = colour.muted;
    context.textAlign = "left";
    context.fillText(heading, pad.left, 12);
    context.strokeStyle = colour.line;
    context.lineWidth = 1;
    context.strokeRect(pad.left, pad.top, plotWidth, plotHeight);
    if (plotWidth <= 0 || plotHeight <= 0) return;

    if (this.kind === "bar") { this.#drawBars(context, pad, plotWidth, plotHeight, colour); return; }
    this.#drawLines(context, pad, plotWidth, plotHeight, colour);
  }

  #range() {
    let low = Infinity;
    let high = -Infinity;
    for (const row of this.rows) {
      for (const definition of this.series) {
        const value = row[definition.key];
        if (value === null || value === undefined || !Number.isFinite(value)) continue;
        if (value < low) low = value;
        if (value > high) high = value;
      }
    }
    if (!Number.isFinite(low)) return null;
    // 🔴 A DEGENERATE RANGE MUST BE WIDENED, NOT DIVIDED BY. The first few
    // frames of a constant quantity give high === low, and scaling by zero puts
    // every point on one pixel row or produces NaN coordinates that silently
    // abort the path.
    if (high - low < 1e-12) { const pad = Math.max(Math.abs(high) * 1e-6, 1e-9); return [low - pad, high + pad]; }
    const margin = (high - low) * 0.08;
    // 🔴 NEVER AN AXIS BELOW ZERO FOR A QUANTITY THAT CANNOT GO THERE. RMSD,
    // RMSF, Rg and temperature all have a hard floor at zero, and an RMSD
    // series starts at EXACTLY zero by construction (frame one against itself),
    // so the 8% margin printed "-0.0160" as the bottom label - a number that
    // cannot exist, on the panel a reader looks at first.
    return [low >= 0 ? Math.max(0, low - margin) : low - margin, high + margin];
  }

  #drawLines(context, pad, plotWidth, plotHeight, colour) {
    if (this.rows.length < 2) return;
    const range = this.#range();
    if (range === null) return;
    const [low, high] = range;
    const firstTime = this.rows[0].time;
    const lastTime = this.rows[this.rows.length - 1].time;
    const span = Math.max(lastTime - firstTime, 1e-9);
    const xOf = (time) => pad.left + (plotWidth * (time - firstTime)) / span;
    const yOf = (value) => pad.top + plotHeight
      - (plotHeight * (value - low)) / (high - low);

    context.fillStyle = colour.muted;
    context.textAlign = "right";
    context.fillText(formatShort(high), pad.left - 5, pad.top + 8);
    context.fillText(formatShort(low), pad.left - 5, pad.top + plotHeight);
    context.textAlign = "center";
    context.fillText(`${firstTime.toFixed(2)} ps`, pad.left + 26, pad.top + plotHeight + 14);
    context.fillText(`${lastTime.toFixed(2)} ps`,
      pad.left + plotWidth - 26, pad.top + plotHeight + 14);

    let legendX = this.legendStart ?? pad.left + 6;
    for (const definition of this.series) {
      let started = false;
      context.strokeStyle = definition.colour;
      context.lineWidth = 1.6;
      context.beginPath();
      for (const row of this.rows) {
        const value = row[definition.key];
        if (value === null || value === undefined || !Number.isFinite(value)) continue;
        const x = xOf(row.time);
        const y = yOf(value);
        if (!started) { context.moveTo(x, y); started = true; } else context.lineTo(x, y);
      }
      if (started) context.stroke();
      if (this.series.length > 1 && started) {
        const y = this.legendY ?? 6;
        const entry = 22 + context.measureText(definition.label).width;
        if (legendX + entry - 10 <= (this.legendLimit ?? Infinity)) {
          context.fillStyle = definition.colour;
          context.fillRect(legendX, y, 8, 2);
          context.textAlign = "left";
          context.fillText(definition.label, legendX + 12, y + 5);
          legendX += entry;
        }
      }
    }
  }

  #drawBars(context, pad, plotWidth, plotHeight, colour) {
    const values = this.bars;
    if (values === null || values.length === 0) return;
    let high = 0;
    for (const value of values) if (Number.isFinite(value) && value > high) high = value;
    if (high <= 0) high = 1;
    const slot = plotWidth / values.length;
    context.fillStyle = this.series[0].colour;
    for (let index = 0; index < values.length; index += 1) {
      const value = Number.isFinite(values[index]) ? values[index] : 0;
      const barHeight = (plotHeight * value) / high;
      context.fillRect(pad.left + index * slot + slot * 0.15,
        pad.top + plotHeight - barHeight, Math.max(slot * 0.7, 1), barHeight);
    }
    context.fillStyle = colour.muted;
    context.textAlign = "right";
    context.fillText(formatShort(high), pad.left - 5, pad.top + 8);
    context.fillText("0", pad.left - 5, pad.top + plotHeight);
    // 🔴 LABEL ONLY WHAT FITS. One label per residue is unreadable past about
    // thirty and turns into a grey smear; a stride keeps the axis meaningful at
    // any length.
    if (this.labels !== null) {
      context.textAlign = "center";
      const stride = Math.max(1, Math.ceil(values.length / Math.floor(plotWidth / 46)));
      for (let index = 0; index < values.length; index += stride) {
        context.fillText(this.labels[index] ?? String(index + 1),
          pad.left + index * slot + slot / 2, pad.top + plotHeight + 14);
      }
    }
  }
}

/** A stack of panels that share a redraw. */
export class PanelStack {
  constructor(panels) { this.panels = panels; }
  push(row) { for (const panel of this.panels) panel.push(row); }
  setBars(key, values) {
    for (const panel of this.panels) if (panel.kind === "bar") panel.setBars(values);
    void key;
  }
  clear() { for (const panel of this.panels) panel.clear(); }
  draw() { for (const panel of this.panels) panel.draw(); }
}
