#!/usr/bin/env python3
"""Fetch GoatCounter daily statistics and render a cumulative SVG line chart."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MODULE_ROOT / "config.json"
DEFAULT_OUTPUT = MODULE_ROOT / "assets" / "dataset-visitors.svg"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = {
        "dataset_name",
        "tracking_path",
        "goatcounter_site",
        "chart_days",
        "timezone",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(missing)}")

    config["goatcounter_site"] = str(config["goatcounter_site"]).rstrip("/")
    config["chart_days"] = int(config["chart_days"])
    if config["chart_days"] < 7 or config["chart_days"] > 365:
        raise ValueError("chart_days must be between 7 and 365")

    tracking_path = str(config["tracking_path"])
    if not tracking_path.startswith("/"):
        tracking_path = "/" + tracking_path
    config["tracking_path"] = tracking_path
    return config


def fetch_stats(config: dict, token: str, start_day: date, end_day: date) -> dict:
    timezone = ZoneInfo(config["timezone"])
    start_at = datetime.combine(start_day, time.min, tzinfo=timezone)
    end_at = datetime.combine(end_day + timedelta(days=1), time.min, tzinfo=timezone)

    query = urlencode(
        [
            ("start", start_at.isoformat()),
            ("end", end_at.isoformat()),
            ("group", "day"),
            ("include_paths", config["tracking_path"]),
            ("path_by_name", "true"),
            ("limit", "100"),
        ]
    )
    url = f'{config["goatcounter_site"]}/api/v0/stats/hits?{query}'
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "GTR-dataset-stats/1.0",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GoatCounter API returned HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach GoatCounter API: {exc.reason}") from exc


def series_from_payload(
    payload: dict, tracking_path: str, start_day: date, end_day: date
) -> list[tuple[date, int]]:
    values: dict[date, int] = {}

    selected = None
    for hit in payload.get("hits", []):
        if hit.get("path") == tracking_path:
            selected = hit
            break

    if selected is None and len(payload.get("hits", [])) == 1:
        selected = payload["hits"][0]

    if selected:
        for item in selected.get("stats", []):
            day_text = item.get("day")
            if not day_text:
                continue
            try:
                day = date.fromisoformat(day_text[:10])
            except ValueError:
                continue
            values[day] = max(0, int(item.get("daily", 0) or 0))

    result: list[tuple[date, int]] = []
    current = start_day
    while current <= end_day:
        result.append((current, values.get(current, 0)))
        current += timedelta(days=1)
    return result


def cumulative_series(data: list[tuple[date, int]]) -> list[tuple[date, int]]:
    total = 0
    result: list[tuple[date, int]] = []
    for day, daily_value in data:
        total += max(0, daily_value)
        result.append((day, total))
    return result


def nice_ceiling(value: int) -> int:
    if value <= 1:
        return 1
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    return int(nice * magnitude)


def render_svg(
    data: list[tuple[date, int]],
    dataset_name: str,
    timezone_name: str,
    generated_at: datetime,
) -> str:
    width, height = 960, 360
    left, right, top, bottom = 76, 32, 86, 62
    plot_width = width - left - right
    plot_height = height - top - bottom

    values = [value for _, value in data]
    maximum = max(values, default=0)
    y_max = nice_ceiling(maximum)
    total = values[-1] if values else 0

    def x_position(index: int) -> float:
        if len(data) <= 1:
            return left + plot_width / 2
        return left + index * plot_width / (len(data) - 1)

    def y_position(value: int) -> float:
        return top + plot_height - (value / y_max) * plot_height

    points = [(x_position(i), y_position(value)) for i, (_, value) in enumerate(data)]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    area_points = (
        f"{left:.2f},{top + plot_height:.2f} "
        + polyline
        + f" {left + plot_width:.2f},{top + plot_height:.2f}"
    )

    y_ticks = []
    for index in range(5):
        value = round(y_max * index / 4)
        y = y_position(value)
        y_ticks.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" '
            f'x2="{left + plot_width}" y2="{y:.2f}"/>'
            f'<text class="axis-label" x="{left - 14}" y="{y + 4:.2f}" '
            f'text-anchor="end">{value}</text>'
        )

    label_count = min(6, len(data))
    label_indices = sorted(
        {
            round(index * (len(data) - 1) / max(1, label_count - 1))
            for index in range(label_count)
        }
    )
    x_labels = []
    for index in label_indices:
        day = data[index][0]
        x = x_position(index)
        x_labels.append(
            f'<text class="axis-label" x="{x:.2f}" y="{top + plot_height + 28}" '
            f'text-anchor="middle">{day.strftime("%m-%d")}</text>'
        )

    point_elements = []
    for (x, y), (_, value) in zip(points, data):
        point_elements.append(
            f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="3">'
            f'<title>{value} cumulative visits</title></circle>'
        )

    title = html.escape(f"{dataset_name} Dataset Access")
    subtitle = html.escape(
        f"Cumulative daily unique visits (approx.) - {len(data)} complete days"
    )
    generated = html.escape(
        f"Updated {generated_at.strftime('%Y-%m-%d %H:%M')} {timezone_name}"
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 {width} {height}"
     width="{width}" height="{height}"
     role="img" aria-labelledby="chart-title chart-desc"
     style="color-scheme: light dark">
  <title id="chart-title">{title}</title>
  <desc id="chart-desc">{subtitle}. Total visits shown: {total}.</desc>
  <style>
    .background {{ fill: Canvas; }}
    .text {{ fill: CanvasText; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .title {{ font-size: 24px; font-weight: 650; }}
    .subtitle {{ font-size: 13px; opacity: .68; }}
    .metric {{ font-size: 28px; font-weight: 650; }}
    .metric-label {{ font-size: 12px; opacity: .65; }}
    .grid {{ stroke: CanvasText; stroke-opacity: .12; stroke-width: 1; }}
    .axis {{ stroke: CanvasText; stroke-opacity: .32; stroke-width: 1; }}
    .axis-label {{ fill: CanvasText; fill-opacity: .62; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 11px; }}
    .area {{ fill: CanvasText; fill-opacity: .055; }}
    .line {{ fill: none; stroke: CanvasText; stroke-width: 2.7; stroke-linejoin: round; stroke-linecap: round; }}
    .point {{ fill: Canvas; stroke: CanvasText; stroke-width: 2; }}
    .footer {{ fill: CanvasText; fill-opacity: .52; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 10px; }}
  </style>

  <rect class="background" width="{width}" height="{height}" rx="12"/>
  <text class="text title" x="{left}" y="38">{title}</text>
  <text class="text subtitle" x="{left}" y="60">{subtitle}</text>

  <text class="text metric" x="{width - right}" y="38" text-anchor="end">{total}</text>
  <text class="text metric-label" x="{width - right}" y="58" text-anchor="end">cumulative visits</text>

  {''.join(y_ticks)}
  <line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>
  <line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>

  <polygon class="area" points="{area_points}"/>
  <polyline class="line" points="{polyline}"/>
  {''.join(point_elements)}
  {''.join(x_labels)}

  <text class="footer" x="{left}" y="{height - 18}">Source: GoatCounter - cumulative sum starts after the tracked link is enabled</text>
  <text class="footer" x="{width - right}" y="{height - 18}" text-anchor="end">{generated}</text>
</svg>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    token = os.environ.get("GOATCOUNTER_API_KEY", "").strip()
    if not token:
        print("GOATCOUNTER_API_KEY is not set.", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config)
        if "REPLACE-ME" in config["goatcounter_site"]:
            raise ValueError(
                "Replace goatcounter_site in dataset_stats/config.json first"
            )

        timezone = ZoneInfo(config["timezone"])
        now = datetime.now(timezone)
        end_day = now.date() - timedelta(days=1)
        start_day = end_day - timedelta(days=config["chart_days"] - 1)

        payload = fetch_stats(config, token, start_day, end_day)
        daily_data = series_from_payload(
            payload, config["tracking_path"], start_day, end_day
        )
        data = cumulative_series(daily_data)
        svg = render_svg(
            data,
            str(config["dataset_name"]),
            str(config["timezone"]),
            now,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(svg, encoding="utf-8")
        print(f"Updated chart: {args.output}")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Chart update failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
