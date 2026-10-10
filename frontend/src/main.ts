import './style.css';

type Row = {origin?: string; target: string; actual: number | null; lower: number; point: number; upper: number; horizon?: number; node?: string};
type Meta = {version: string; nodes: string[]; horizons: number[]; folds: string[]; models: Record<string, number[]>; scenarios: string[]; summary: Array<Record<string, number | string>>};
type Backtest = {rows: Row[]; metrics: {mase: number; mae_mw: number; coverage_80: number}; joint_coverage: number; peak_threshold: number; mase_scale: number};
type Forecast = {origin: string; scenario: string; forecasts: Row[]; warnings: string[]; repaired_counts: Record<string, number>; history: Array<{timestamp: string; values: Record<string, number>}>};
const element = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const select = (id: string) => element<HTMLSelectElement>(id);
const format = (value: number, digits = 0) => value.toLocaleString('en', {maximumFractionDigits: digits, minimumFractionDigits: digits});
const percent = (value: number) => `${format(value * 100, 1)}%`;
const label = (value: string) => ({tree: 'Quantile tree', transformer: 'Temporal transformer', ALL_FIVE: 'All five regions', clean: 'Complete readings', missing_6h: 'Missing last 6 hours', missing_24h: 'Missing last 24 hours', missing_10pct: '10% missing in last week', bias_up_10pct_24h: 'Readings biased upward 10%'}[value] ?? value);
let meta: Meta;
let view = 'backtest';
let serial = 0;
let currentRows: Row[] = [];
let replay: Backtest | null = null;
let forecast: Forecast | null = null;
let offset = 0;
const params = () => ({fold: select('fold').value, family: select('family').value, seed: select('seed').value, node: select('node').value, horizon: select('horizon').value});

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Please check the supplied values.');
  return data;
}
function options(id: string, values: (string | number)[], names = label) {
  select(id).replaceChildren(...values.map(value => new Option(names(String(value)), String(value))));
}
function busy(value: boolean) { element('loading').hidden = !value; element('loading').textContent = value ? 'Loading verified forecasts…' : ''; }
function error(message = '') { element('error').hidden = !message; element('error').textContent = message; }
function card(title: string, value: string, detail: string) { return `<div class="card"><span>${title}</span><strong>${value}</strong><small>${detail}</small></div>`; }
function chart(rows: Row[], peak: number) {
  const width = 1000, height = 310, left = 70, right = 22, top = 24, bottom = 45;
  const low = Math.max(0, Math.min(...rows.flatMap(r => [r.lower, r.actual ?? r.lower])) * .92);
  const high = Math.max(...rows.flatMap(r => [r.upper, r.actual ?? r.upper])) * 1.05;
  const x = (i: number) => left + i / Math.max(1, rows.length - 1) * (width - left - right);
  const y = (v: number) => height - bottom - (v - low) / Math.max(1, high - low) * (height - top - bottom);
  const line = (field: 'point' | 'actual') => rows.map((r, i) => `${x(i)},${y(r[field] ?? r.point)}`).join(' ');
  const band = [...rows.map((r, i) => `${x(i)},${y(r.upper)}`), ...rows.map((r, i) => `${x(i)},${y(r.lower)}`).reverse()].join(' ');
  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Actual demand, forecast and simultaneous interval"><title>Actual demand and forecast interval in megawatts</title>`;
  for (let i = 0; i <= 4; i++) { const v = low + (high - low) * i / 4; svg += `<line class="grid" x1="${left}" x2="${width-right}" y1="${y(v)}" y2="${y(v)}"/><text x="${left-12}" y="${y(v)+4}" text-anchor="end">${format(v)}</text>`; }
  for (let i = 0; i < 5; i++) { const at = Math.round(i * (rows.length - 1) / 4); svg += `<text x="${x(at)}" y="${height-14}" text-anchor="${i === 0 ? 'start' : i === 4 ? 'end' : 'middle'}">${rows[at].target.slice(5,16)}</text>`; }
  svg += `<polygon fill="#d5ebe4" points="${band}"/><polyline class="actual-line" points="${line('actual')}"/><polyline class="forecast-line" points="${line('point')}"/>`;
  rows.forEach((r, i) => { if ((r.actual ?? 0) > peak) svg += `<circle fill="#b56634" r="3" cx="${x(i)}" cy="${y(r.actual!)}"><title>${r.target}: peak demand ${format(r.actual!)} MW</title></circle>`; });
  return svg + '</svg>';
}
function forecastChart(data: Forecast, node: string) {
  const past = data.history.map((r, i) => ({x: i - data.history.length + 1, y: r.values[node]}));
  const rows = data.forecasts.filter(r => r.node === node);
  const high = Math.max(...past.map(r => r.y), ...rows.map(r => Math.max(r.upper, r.actual ?? 0))) * 1.08;
  const x = (v: number) => 65 + (v + 72) / 240 * 900;
  const y = (v: number) => 265 - v / high * 240;
  let svg = '<svg viewBox="0 0 1000 310" role="img" aria-label="Observed history and four forecast horizons"><title>Forecast intervals at 1, 6, 24 and 168 hours</title>';
  for (let i = 0; i < 5; i++) { const v = high * i / 4; svg += `<line class="grid" x1="65" x2="965" y1="${y(v)}" y2="${y(v)}"/><text x="55" y="${y(v)+4}" text-anchor="end">${format(v)}</text>`; }
  svg += `<polyline class="actual-line" points="${past.map(p => `${x(p.x)},${y(p.y)}`).join(' ')}"/><line x1="${x(0)}" x2="${x(0)}" y1="20" y2="265" stroke="#899994" stroke-dasharray="4 5"/>`;
  rows.forEach(r => { const px = x(r.horizon!); svg += `<line x1="${px}" x2="${px}" y1="${y(r.lower)}" y2="${y(r.upper)}" stroke="#73b6a1" stroke-width="9"/><circle cx="${px}" cy="${y(r.point)}" r="4" fill="#15785e"><title>${r.horizon}h: ${format(r.point)} MW, band ${format(r.lower)}–${format(r.upper)} MW</title></circle>`; if (r.actual !== null) svg += `<circle cx="${px}" cy="${y(r.actual)}" r="3" fill="#203b50"/>`; });
  for (const h of [-72, -24, 0, 24, 72, 168]) svg += `<text x="${x(h)}" y="295" text-anchor="middle">${h === 0 ? 'Origin' : `${h > 0 ? '+' : ''}${h}h`}</text>`;
  return svg + '</svg>';
}
function table(rows: Row[]) {
  return `<details><summary>Inspect values · ${rows.length} forecasts</summary><div class="table-scroll"><table><thead><tr><th>Target (source clock)</th><th>Actual MW</th><th>Forecast MW</th><th>Lower MW</th><th>Upper MW</th></tr></thead><tbody>${rows.map(r => `<tr><td>${r.target}</td><td>${r.actual === null ? 'Unavailable' : format(r.actual,1)}</td><td>${format(r.point,1)}</td><td>${format(r.lower,1)}</td><td>${format(r.upper,1)}</td></tr>`).join('')}</tbody></table></div></details>`;
}
function download() {
  const text = 'target,actual_mw,point_mw,lower_mw,upper_mw\n' + currentRows.map(r => [r.target, r.actual ?? '', r.point, r.lower, r.upper].join(',')).join('\n');
  const url = URL.createObjectURL(new Blob([text], {type: 'text/csv'}));
  const link = document.createElement('a'); link.href = url; link.download = 'regional-demand-forecast.csv'; link.click(); URL.revokeObjectURL(url);
}
function heading(title: string, note: string) { return `<div class="panel-heading"><div><p class="eyebrow">${note}</p><h2>${title}</h2></div><button id="download" class="subtle">Download CSV ↓</button></div>`; }
const legend = '<div class="legend"><span class="actual-key">Actual demand</span><span class="forecast-key">Point forecast</span><span class="band-key">Simultaneous band</span><span class="peak-key">Training-defined peak</span></div>';
function renderReplay() {
  if (!replay) return;
  const p = params(); currentRows = replay.rows.slice(offset, offset + 168);
  const mae = currentRows.reduce((v,r) => v + Math.abs(r.actual! - r.point), 0) / currentRows.length;
  const coverage = currentRows.filter(r => r.actual! >= r.lower && r.actual! <= r.upper).length / currentRows.length;
  element('content').innerHTML = heading(`${label(p.node)} · ${p.horizon}-hour horizon`, `${label(p.family)} · seed ${p.seed}`) +
    `<div class="cards">${card('Window MASE', format(mae/replay.mase_scale,3), 'Lower is better; training-scaled error')}${card('Region covered', percent(coverage), 'Visible window · individual region')}${card('All regions covered', percent(replay.joint_coverage), 'Whole evaluation period · joint coverage')}</div><div class="chart-panel">${legend}${chart(currentRows, replay.peak_threshold)}<p class="caption">Hourly target times on the source clock · MW. Peaks exceed the region’s training-window 95th percentile.</p><label class="slider-label">Move the viewing window<input id="window" type="range" min="0" max="${Math.max(0,replay.rows.length-168)}" step="24" value="${offset}"></label><p class="range-label">${currentRows[0].target} → ${currentRows.at(-1)!.target}</p></div>${table(currentRows)}`;
  element<HTMLInputElement>('window').oninput = (event) => { offset = Number((event.target as HTMLInputElement).value); renderReplay(); };
  element('download').onclick = download;
}
function renderForecast() {
  if (!forecast) { element('content').innerHTML = '<div class="empty"><p class="eyebrow">REPLAY THE OBSERVATIONS</p><h2>What would the model forecast?</h2><p>Choose an origin and an observation scenario, then run the saved model. No training happens here.</p><p>Forecasts use readings available at the origin. Recorded future outcomes appear only for comparison.</p></div>'; return; }
  const p = params(); currentRows = forecast.forecasts.filter(r => r.node === p.node);
  const repaired = Object.values(forecast.repaired_counts).reduce((a,b) => a+b,0);
  element('content').innerHTML = heading(`${label(p.node)} · four forecast horizons`, `${label(p.family)} · origin ${forecast.origin}`) +
    `<div class="notice">${forecast.warnings.map(w => `<p>${w}</p>`).join('')}</div><div class="cards">${card('Readings repaired', format(repaired), 'Across five regional histories')}${card('Input scenario', label(forecast.scenario), 'Synthetic failures change readings only')}</div><div class="chart-panel"><div class="legend"><span class="actual-key">Observed / recorded outcome</span><span class="forecast-key">Forecast with band</span></div>${forecastChart(forecast,p.node)}<p class="caption">Past 72 hours and four discrete future horizons. Vertical bars are prediction bands; no intermediate forecasts are implied. Past readings reflect the selected repair or bias scenario.</p></div>${table(currentRows)}`;
  element('download').onclick = download;
}
async function renderEvidence(ticket: number) {
  const stress = await api<{metrics: Array<{family: string; scenario: string; mase: number; simultaneous_leaf_coverage: number}>}>('/api/stress');
  if (ticket !== serial) return;
  const scenarios = meta.scenarios.map(scenario => {
    const cells = ['tree','transformer'].map(family => { const group = stress.metrics.filter(r => r.family === family && r.scenario === scenario); return `<td>${format(group.reduce((s,r) => s+r.mase,0)/group.length,3)}</td><td>${percent(group.reduce((s,r) => s+r.simultaneous_leaf_coverage,0)/group.length)}</td>`; }).join('');
    return `<tr><td>${label(scenario)}</td>${cells}</tr>`;
  }).join('');
  element('content').innerHTML = `<div class="panel-heading"><div><p class="eyebrow">EVIDENCE BEFORE CONFIDENCE</p><h2>Accuracy and uncertainty, together.</h2></div></div><p class="body-copy">The tree is more accurate. The transformer’s wider calibrated bands cover more joint outcomes in this forward test. Neither result guarantees coverage on new inputs.</p><div class="table-scroll"><table><thead><tr><th>Model / output</th><th>MASE</th><th>Joint coverage</th><th>Mean width MW</th></tr></thead><tbody>${meta.summary.map(r => `<tr><td>${label(String(r.family))} · ${r.method}</td><td>${format(Number(r.mase),3)}</td><td>${percent(Number(r.simultaneous_leaf_coverage))}</td><td>${format(Number(r.width_mw))}</td></tr>`).join('')}</tbody></table></div><p class="caption">Equal-weight folds, horizons and seeds within each family. MASE and width include all six nodes; joint coverage requires all five leaves together. Raw outputs are marginal quantiles; calibrated outputs are simultaneous bands.</p><h3>When observations fail</h3><div class="table-scroll"><table><thead><tr><th>Scenario</th><th>Tree MASE</th><th>Tree joint</th><th>Transformer MASE</th><th>Transformer joint</th></tr></thead><tbody>${scenarios}</tbody></table></div><p class="caption">Matched daily midnight origins only, not the full hourly test. Missing values use past readings, with gaps longer than 24 hours rejected. Real future labels are unchanged. Repaired inputs do not receive a new calibration guarantee.</p>`;
}
async function refresh() {
  const ticket = ++serial; error(); busy(true);
  element('controls').hidden = view === 'evidence'; element('workspace').classList.toggle('evidence',view === 'evidence');
  element('forecast-controls').hidden = view !== 'forecast'; element('horizon-label').hidden = view !== 'backtest';
  document.querySelectorAll<HTMLButtonElement>('[data-view]').forEach(button => { button.classList.toggle('active', button.dataset.view === view); button.setAttribute('aria-pressed',String(button.dataset.view === view)); });
  try {
    if (view === 'backtest') { const result = await api<Backtest>('/api/backtest?'+new URLSearchParams(params())); if (ticket !== serial) return; replay = result; offset = 0; renderReplay(); }
    else if (view === 'forecast') renderForecast();
    else await renderEvidence(ticket);
  } catch (e) { if (ticket === serial) error((e as Error).message); }
  finally { if (ticket === serial) busy(false); }
}
async function runForecast() {
  const ticket = ++serial; error(); busy(true); element<HTMLButtonElement>('predict').disabled = true;
  try {
    const p = params(); const result = await api<Forecast>('/api/forecast',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fold:p.fold,family:p.family,seed:Number(p.seed),origin:element<HTMLInputElement>('origin').value,scenario:select('scenario').value})});
    if (ticket !== serial) return; forecast = result; renderForecast();
  } catch (e) { if (ticket === serial) error((e as Error).message); }
  finally { element<HTMLButtonElement>('predict').disabled = false; if (ticket === serial) busy(false); }
}
async function start() {
  try {
    meta = await api<Meta>('/api/meta'); options('fold',meta.folds); options('node',meta.nodes); options('horizon',meta.horizons,v => `${v} hours`); options('scenario',meta.scenarios); options('seed',meta.models.tree);
    element<HTMLInputElement>('origin').value = meta.folds[0]+'T00:00'; select('horizon').value = String(meta.horizons.includes(24)?24:meta.horizons[0]);
    element('version').textContent = `Verified experiment ${meta.version.slice(0,12)} · local inference · no external API`;
    element('connection').textContent = '● Verified local experiment'; element('workspace').hidden = false;
    for (const id of ['fold','family','seed','node','horizon']) select(id).onchange = () => { if (id === 'family') options('seed',meta.models[select('family').value]); if (id === 'fold') element<HTMLInputElement>('origin').value = select('fold').value+'T00:00'; forecast = null; void refresh(); };
    const invalidate = () => { serial++; forecast = null; busy(false); error(); renderForecast(); };
    select('scenario').onchange = invalidate; element<HTMLInputElement>('origin').oninput = invalidate;
    document.querySelectorAll<HTMLButtonElement>('[data-view]').forEach(button => button.onclick = () => { view = button.dataset.view!; void refresh(); });
    element('predict').onclick = () => void runForecast(); await refresh();
  } catch (e) { error(`Could not load the experiment. ${(e as Error).message}`); busy(false); element('connection').textContent = 'Connection unavailable'; }
}
void start();
