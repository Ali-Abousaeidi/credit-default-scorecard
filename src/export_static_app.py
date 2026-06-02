"""Export a standalone HTML scorecard app from generated model artifacts."""

from __future__ import annotations

import json
import re

import pandas as pd

from src.config import (
    BASE_ODDS,
    BASE_SCORE,
    CLEANING_METADATA_FILE,
    PDO,
    SCORECARD_FILE,
    STATIC_SCORECARD_APP_FILE,
)
from src.scorecard import scorecard_factor, scorecard_offset
from src.scoring import DISPLAY_LABELS, FEATURE_COLUMNS, LATE_PAYMENT_COLUMNS

DEFAULT_APPLICANT = {
    "RevolvingUtilizationOfUnsecuredLines": 0.35,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 5500,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
}

INTEGER_FEATURES = {
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
}

INTERVAL_RE = re.compile(r"^([\[(])([^,]+),\s*([^\])]+)([\])])$")


def _parse_bound(value: str) -> float | None:
    """Parse an interval bound from the scorecard table."""
    clean_value = value.strip()
    if clean_value in {"-inf", "inf"}:
        return None
    return float(clean_value)


def parse_attribute(attribute: str) -> dict[str, object]:
    """Convert a scorecard bin label into JavaScript-friendly interval metadata."""
    if attribute == "Missing":
        return {
            "label": attribute,
            "missing": True,
            "lower": None,
            "upper": None,
            "lowerInclusive": False,
            "upperInclusive": False,
        }

    match = INTERVAL_RE.match(attribute)
    if not match:
        raise ValueError(f"Unsupported scorecard attribute format: {attribute}")

    lower_bracket, lower, upper, upper_bracket = match.groups()
    return {
        "label": attribute,
        "missing": False,
        "lower": _parse_bound(lower),
        "upper": _parse_bound(upper),
        "lowerInclusive": lower_bracket == "[",
        "upperInclusive": upper_bracket == "]",
    }


def build_static_payload(scorecard: pd.DataFrame, cleaning_metadata: dict) -> dict[str, object]:
    """Build the embedded JSON payload consumed by the static browser app."""
    selected_features = list(dict.fromkeys(scorecard["characteristic"].tolist()))
    intercept = 0.0
    if not scorecard.empty:
        n_features = len(selected_features)
        row = scorecard.iloc[0]
        attribute_logit = (
            (scorecard_offset() / n_features) - float(row["points"])
        ) / scorecard_factor()
        intercept = (attribute_logit - float(row["coefficient"]) * float(row["woe"])) * n_features

    bins_by_feature: dict[str, list[dict[str, object]]] = {}
    for feature, feature_table in scorecard.groupby("characteristic", sort=False):
        bins = []
        for _, row in feature_table.sort_values("bin_order").iterrows():
            bin_data = parse_attribute(str(row["attribute"]))
            bin_data.update(
                {
                    "woe": float(row["woe"]),
                    "coefficient": float(row["coefficient"]),
                    "eventRate": float(row["event_rate"]),
                    "points": float(row["points"]),
                    "pointsRounded": int(row["points_rounded"]),
                }
            )
            bins.append(bin_data)
        bins_by_feature[feature] = bins

    features = []
    for feature in FEATURE_COLUMNS:
        features.append(
            {
                "name": feature,
                "label": DISPLAY_LABELS.get(feature, feature),
                "defaultValue": DEFAULT_APPLICANT[feature],
                "step": 1 if feature in INTEGER_FEATURES else (250 if feature == "MonthlyIncome" else 0.01),
                "selected": feature in selected_features,
            }
        )

    return {
        "baseScore": BASE_SCORE,
        "baseOdds": BASE_ODDS,
        "pdo": PDO,
        "factor": scorecard_factor(),
        "offset": scorecard_offset(),
        "intercept": intercept,
        "caps": cleaning_metadata["fitted_upper_caps"],
        "latePaymentColumns": LATE_PAYMENT_COLUMNS,
        "features": features,
        "selectedFeatures": selected_features,
        "binsByFeature": bins_by_feature,
    }


def build_html(payload: dict[str, object]) -> str:
    """Return a self-contained static HTML scorecard application."""
    payload_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Credit Default Scorecard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #657080;
      --line: #d8dee6;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --risk: #b42318;
      --safe: #17633a;
      --warn: #946200;
      --table: #f0f4f8;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}

    header {{
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}

    .topbar {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 22px 0 18px;
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
    }}

    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.15;
      font-weight: 700;
      letter-spacing: 0;
    }}

    .meta {{
      color: var(--muted);
      font-size: 13px;
      text-align: right;
    }}

    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 24px auto 40px;
      display: grid;
      grid-template-columns: minmax(280px, 360px) 1fr;
      gap: 18px;
      align-items: start;
    }}

    section, aside {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}

    .panel-head {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}

    h2 {{
      margin: 0;
      font-size: 16px;
      line-height: 1.25;
      letter-spacing: 0;
    }}

    .inputs {{
      padding: 14px 16px 16px;
      display: grid;
      gap: 12px;
    }}

    label {{
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}

    input {{
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      color: var(--ink);
      background: #fff;
      font-size: 15px;
    }}

    input:focus {{
      border-color: var(--accent);
      outline: 3px solid rgba(15, 118, 110, 0.16);
    }}

    .inactive {{
      opacity: 0.72;
    }}

    .tag {{
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      padding: 2px 8px;
      font-size: 11px;
      font-weight: 700;
      white-space: nowrap;
    }}

    .result-grid {{
      padding: 16px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}

    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px;
      min-height: 98px;
      display: grid;
      align-content: space-between;
      gap: 10px;
    }}

    .metric span {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}

    .metric strong {{
      font-size: 30px;
      line-height: 1;
      letter-spacing: 0;
    }}

    .band-low {{ color: var(--safe); }}
    .band-medium {{ color: var(--warn); }}
    .band-high {{ color: var(--risk); }}

    .tables {{
      display: grid;
      gap: 18px;
      margin-top: 18px;
    }}

    .table-wrap {{
      overflow-x: auto;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}

    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}

    th {{
      background: var(--table);
      color: #344054;
      font-size: 12px;
      font-weight: 700;
    }}

    td.numeric, th.numeric {{
      text-align: right;
    }}

    .footnote {{
      padding: 11px 16px 14px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }}

    @media (max-width: 860px) {{
      .topbar {{
        align-items: start;
        flex-direction: column;
      }}

      .meta {{
        text-align: left;
      }}

      main {{
        grid-template-columns: 1fr;
      }}

      .result-grid {{
        grid-template-columns: 1fr;
      }}

      th, td {{
        white-space: normal;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <h1>Credit Default Scorecard</h1>
      <div class="meta">Standalone HTML app | {BASE_SCORE} points at {BASE_ODDS:.0f}:1 odds | PDO {PDO}</div>
    </div>
  </header>

  <main>
    <aside>
      <div class="panel-head">
        <h2>Applicant Inputs</h2>
        <span class="tag">browser only</span>
      </div>
      <form class="inputs" id="applicant-form"></form>
      <div class="footnote">Fields marked "not selected" are retained from the project input schema but do not affect the champion scorecard.</div>
    </aside>

    <div>
      <section>
        <div class="panel-head">
          <h2>Score Result</h2>
          <span class="tag" id="selected-count"></span>
        </div>
        <div class="result-grid">
          <div class="metric">
            <span>Credit score</span>
            <strong id="score">--</strong>
          </div>
          <div class="metric">
            <span>Predicted default probability</span>
            <strong id="pd">--</strong>
          </div>
          <div class="metric">
            <span>Risk band</span>
            <strong id="band">--</strong>
          </div>
        </div>
      </section>

      <div class="tables">
        <section>
          <div class="panel-head">
            <h2>Top Reason Codes</h2>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Characteristic</th>
                  <th>Bin</th>
                  <th>Direction</th>
                  <th class="numeric">Contribution</th>
                  <th class="numeric">WoE</th>
                </tr>
              </thead>
              <tbody id="reason-body"></tbody>
            </table>
          </div>
        </section>

        <section>
          <div class="panel-head">
            <h2>Scorecard Points</h2>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Characteristic</th>
                  <th>Attribute</th>
                  <th class="numeric">Event rate</th>
                  <th class="numeric">WoE</th>
                  <th class="numeric">Points</th>
                </tr>
              </thead>
              <tbody id="points-body"></tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  </main>

  <script id="scorecard-data" type="application/json">{payload_json}</script>
  <script>
    const data = JSON.parse(document.getElementById("scorecard-data").textContent);
    const form = document.getElementById("applicant-form");
    const formatter = new Intl.NumberFormat("en-US", {{ maximumFractionDigits: 0 }});
    const pctFormatter = new Intl.NumberFormat("en-US", {{
      style: "percent",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }});

    function labelFor(featureName) {{
      const feature = data.features.find((item) => item.name === featureName);
      return feature ? feature.label : featureName;
    }}

    function renderInputs() {{
      for (const feature of data.features) {{
        const label = document.createElement("label");
        if (!feature.selected) {{
          label.classList.add("inactive");
        }}
        label.textContent = feature.label;

        if (!feature.selected) {{
          const tag = document.createElement("span");
          tag.className = "tag";
          tag.textContent = "not selected";
          label.appendChild(tag);
        }}

        const input = document.createElement("input");
        input.type = "number";
        input.name = feature.name;
        input.value = feature.defaultValue;
        input.min = 0;
        input.step = feature.step;
        input.inputMode = "decimal";
        input.addEventListener("input", scoreApplicant);
        label.appendChild(input);
        form.appendChild(label);
      }}
    }}

    function valueFor(featureName) {{
      const input = form.elements[featureName];
      const value = Number(input.value);
      if (!Number.isFinite(value)) {{
        return null;
      }}
      if (featureName === "age" && value <= 0) {{
        return null;
      }}
      if (data.latePaymentColumns.includes(featureName) && (value === 96 || value === 98)) {{
        return null;
      }}
      if (Object.prototype.hasOwnProperty.call(data.caps, featureName)) {{
        return Math.min(value, Number(data.caps[featureName]));
      }}
      return value;
    }}

    function matchesInterval(value, bin) {{
      if (bin.lower !== null) {{
        if (bin.lowerInclusive ? value < bin.lower : value <= bin.lower) {{
          return false;
        }}
      }}
      if (bin.upper !== null) {{
        if (bin.upperInclusive ? value > bin.upper : value >= bin.upper) {{
          return false;
        }}
      }}
      return true;
    }}

    function findBin(featureName, value) {{
      const bins = data.binsByFeature[featureName] || [];
      if (value === null) {{
        return bins.find((bin) => bin.missing) || bins[0];
      }}
      return bins.find((bin) => !bin.missing && matchesInterval(value, bin)) || bins[bins.length - 1];
    }}

    function bandFor(score) {{
      if (score >= 620) {{
        return ["Low risk", "band-low"];
      }}
      if (score >= 560) {{
        return ["Medium risk", "band-medium"];
      }}
      return ["High risk", "band-high"];
    }}

    function scoreApplicant() {{
      let linear = Number(data.intercept);
      const reasons = [];

      for (const featureName of data.selectedFeatures) {{
        const applicantValue = valueFor(featureName);
        const bin = findBin(featureName, applicantValue);
        const contribution = Number(bin.coefficient) * Number(bin.woe);
        linear += contribution;
        reasons.push({{
          characteristic: labelFor(featureName),
          attribute: bin.label,
          direction: contribution > 0 ? "raises PD" : "lowers PD",
          contribution,
          woe: Number(bin.woe)
        }});
      }}

      const pd = 1 / (1 + Math.exp(-linear));
      const score = Number(data.offset) - Number(data.factor) * linear;
      const [band, bandClass] = bandFor(score);

      document.getElementById("score").textContent = formatter.format(score);
      document.getElementById("pd").textContent = pctFormatter.format(pd);
      const bandElement = document.getElementById("band");
      bandElement.textContent = band;
      bandElement.className = bandClass;

      reasons.sort((left, right) => Math.abs(right.contribution) - Math.abs(left.contribution));
      document.getElementById("reason-body").innerHTML = reasons.slice(0, 5).map((reason) => `
        <tr>
          <td>${{reason.characteristic}}</td>
          <td>${{reason.attribute}}</td>
          <td>${{reason.direction}}</td>
          <td class="numeric">${{reason.contribution.toFixed(4)}}</td>
          <td class="numeric">${{reason.woe.toFixed(4)}}</td>
        </tr>
      `).join("");
    }}

    function renderPoints() {{
      const rows = [];
      for (const featureName of data.selectedFeatures) {{
        for (const bin of data.binsByFeature[featureName]) {{
          rows.push(`
            <tr>
              <td>${{labelFor(featureName)}}</td>
              <td>${{bin.label}}</td>
              <td class="numeric">${{pctFormatter.format(bin.eventRate)}}</td>
              <td class="numeric">${{Number(bin.woe).toFixed(4)}}</td>
              <td class="numeric">${{bin.pointsRounded}}</td>
            </tr>
          `);
        }}
      }}
      document.getElementById("points-body").innerHTML = rows.join("");
    }}

    renderInputs();
    renderPoints();
    document.getElementById("selected-count").textContent = `${{data.selectedFeatures.length}} characteristics`;
    scoreApplicant();
  </script>
</body>
</html>
"""


def main() -> None:
    """Write the standalone HTML scorecard app."""
    if not SCORECARD_FILE.exists():
        raise FileNotFoundError("Run `python -m src.scorecard` before exporting the static app.")
    if not CLEANING_METADATA_FILE.exists():
        raise FileNotFoundError("Run `python -m src.data_prep` before exporting the static app.")

    scorecard = pd.read_csv(SCORECARD_FILE)
    cleaning_metadata = json.loads(CLEANING_METADATA_FILE.read_text(encoding="utf-8"))
    payload = build_static_payload(scorecard, cleaning_metadata)

    STATIC_SCORECARD_APP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATIC_SCORECARD_APP_FILE.write_text(build_html(payload), encoding="utf-8")
    print(f"Wrote static scorecard app: {STATIC_SCORECARD_APP_FILE}")


if __name__ == "__main__":
    main()
