#!/usr/bin/env python3
"""Build the small GitHub Pages site used for dataset access tracking."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MODULE_ROOT / "config.json"
DEFAULT_SITE = MODULE_ROOT / "site"
DEFAULT_ASSET = MODULE_ROOT / "assets" / "dataset-visitors.svg"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = {
        "dataset_name",
        "drive_url",
        "tracking_path",
        "goatcounter_site",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(missing)}")

    config["goatcounter_site"] = str(config["goatcounter_site"]).rstrip("/")
    tracking_path = str(config["tracking_path"])
    if not tracking_path.startswith("/"):
        tracking_path = "/" + tracking_path
    config["tracking_path"] = tracking_path
    return config


def render_template(text: str, config: dict) -> str:
    dataset_name = str(config["dataset_name"])
    drive_url = str(config["drive_url"])
    endpoint = f'{config["goatcounter_site"]}/count'

    replacements = {
        "{{DATASET_NAME_HTML}}": html.escape(dataset_name, quote=True),
        "{{DATASET_NAME_JS}}": json.dumps(dataset_name, ensure_ascii=False),
        "{{DRIVE_URL_HTML}}": html.escape(drive_url, quote=True),
        "{{DRIVE_URL_JS}}": json.dumps(drive_url),
        "{{TRACKING_PATH_JS}}": json.dumps(config["tracking_path"]),
        "{{GOATCOUNTER_ENDPOINT_HTML}}": html.escape(endpoint, quote=True),
    }
    for marker, value in replacements.items():
        text = text.replace(marker, value)
    return text


def build(config_path: Path, output: Path) -> None:
    config = load_config(config_path)

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for source in DEFAULT_SITE.rglob("*"):
        relative = source.relative_to(DEFAULT_SITE)
        destination = output / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == ".html":
            rendered = render_template(source.read_text(encoding="utf-8"), config)
            destination.write_text(rendered, encoding="utf-8")
        else:
            shutil.copy2(source, destination)

    shutil.copy2(DEFAULT_ASSET, output / "dataset-visitors.svg")
    (output / ".nojekyll").write_text("", encoding="utf-8")

    if "REPLACE-ME" in config["goatcounter_site"]:
        print(
            "::warning::Edit dataset_stats/config.json and replace "
            "https://REPLACE-ME.goatcounter.com before publishing.",
            file=sys.stderr,
        )

    print(f"Built GitHub Pages site at: {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("_site"))
    args = parser.parse_args()

    try:
        build(args.config, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
