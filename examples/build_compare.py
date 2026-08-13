"""Regenerate compare.md from results.json. Run after adding new results."""
import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results.json"
COMPARE = Path(__file__).parent / "compare.md"


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main():
    records = json.loads(RESULTS.read_text())
    columns = [f"{record['example']} / {record['method']}" for record in records]
    metrics = list(dict.fromkeys(key for record in records for key in record if key not in ("example", "method")))

    lines = [
        "# Method comparison",
        "",
        "| metric | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join("---" for _ in columns) + " |",
    ]
    for metric in metrics:
        row = [fmt(record.get(metric, "")) for record in records]
        lines.append(f"| {metric} | " + " | ".join(row) + " |")

    COMPARE.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(metrics)} metrics x {len(columns)} columns to {COMPARE}")


if __name__ == "__main__":
    main()
