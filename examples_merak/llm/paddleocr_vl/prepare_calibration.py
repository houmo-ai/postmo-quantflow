# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Render held-out-independent OCR calibration pages with exact numeric labels."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    font = ImageFont.truetype(args.font, 20)
    records = []
    for sample in range(args.samples):
        image = Image.new("RGB", (896, 896), "white")
        draw = ImageDraw.Draw(image)
        title = f"Transaction register {sample + 1:03d}"
        draw.text((38, 34), title, font=font, fill="black")
        headers = ["Date", "Reference", "Debit", "Credit", "Balance"]
        rows = []
        balance = rng.randint(1000000, 10000000)
        for row in range(16):
            amount = rng.randint(1, 99999)
            credit = rng.choice([True, False])
            balance += amount if credit else -amount
            rows.append([f"2026-{sample % 12 + 1:02d}-{row + 1:02d}", f"TX{sample:02d}{row:04d}",
                         "0.00" if credit else f"{amount / 100:.2f}",
                         f"{amount / 100:.2f}" if credit else "0.00", f"{balance / 100:.2f}"])
        x_positions = [38, 225, 402, 560, 714, 862]
        for row_index, row in enumerate([headers, *rows]):
            y = 98 + row_index * 41
            draw.line((38, y - 7, 862, y - 7), fill="black", width=1)
            for column, text in enumerate(row):
                draw.text((x_positions[column] + 5, y), text, font=font, fill="black")
        for x in x_positions:
            draw.line((x, 91, x, 791), fill="black", width=1)
        draw.line((38, 791, 862, 791), fill="black", width=1)
        note = f"Prepared by clerk {sample % 7 + 1}. Currency: USD. Rows: 16."
        draw.text((38, 827), note, font=font, fill="black")
        image_path = output / f"ledger_{sample:03d}.png"
        image.save(image_path)
        # The native 1.6 model emits OTSL; HTML is a service-side conversion.
        # These rendered tables have no empty or merged cells.
        table = "".join("".join(f"<fcel>{value}" for value in row) + "<nl>" for row in [headers, *rows])
        records.append({"image": image_path.name, "prompt": "Table Recognition:", "answer": table,
                        "answer_format": "otsl",
                        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
    (output / "calibration.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (output / "provenance.json").write_text(json.dumps({
        "source": "deterministically rendered numeric ledgers", "seed": args.seed,
        "samples": args.samples, "font": args.font, "held_out_issue_images_used": False,
        "answer_format": "otsl", "table_shape": [17, 5], "merged_cells": False,
        "limitation": "English printed tables and numeric text; does not represent handwriting or all languages",
    }, indent=2) + "\n")
    print(output / "calibration.jsonl")


if __name__ == "__main__":
    main()
