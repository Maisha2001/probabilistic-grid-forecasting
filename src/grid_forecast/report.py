import html

import numpy as np


def render_report(report):
    rows = []
    for method in ("weekly_residual", "xgboost_quantile"):
        for horizon in report["config"]["horizons"]:
            records = [r for r in report["metrics"] if r["method"] == method and r["horizon"] == horizon]
            values = [np.mean([r[k] for r in records]) for k in ("mae_mw", "mase", "pinball_mw", "coverage_80", "interval_width_mw")]
            cells = "".join(f"<td>{value:.3f}</td>" for value in values)
            rows.append(f"<tr><td>{html.escape(method)}</td><td>{horizon}</td>{cells}</tr>")
    return """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Regional demand backtest</title><style>body{font:16px system-ui;margin:40px auto;max-width:1050px;padding:0 20px;color:#172c3d;background:#f5f8fb}h1{font-size:30px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #d6e1eb;text-align:right}td:first-child,th:first-child{text-align:left}.table{overflow:auto}p{line-height:1.6}</style>
<h1>Regional demand backtest</h1><p>Equal-weight averages over folds and six nodes, including the derived aggregate. An 80% interval should cover about 80% of observations. Scores are historical benchmark measurements, not a production guarantee.</p>
<div class="table"><table><tr><th>Model</th><th>Horizon (h)</th><th>MAE (MW)</th><th>MASE</th><th>Pinball (MW)</th><th>Coverage</th><th>Width (MW)</th></tr>""" + "".join(rows) + """</table></div>
<p>The source clock has no verified civil-time or UTC mapping. ALL_FIVE sums the five recorded OperationalLessIndustrial series, not official national demand. Forecasts at different nodes remain independent; marginal quantiles must not be added to obtain aggregate uncertainty.</p>
<p>Full per-node/fold metrics, peak errors, median-coherence errors, validation choices, hashes and runtime are in <a href="metrics.json">metrics.json</a>. Every evaluated prediction is in <a href="predictions.parquet">predictions.parquet</a>.</p></html>"""
