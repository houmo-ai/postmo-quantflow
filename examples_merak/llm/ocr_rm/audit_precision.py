# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Reject SEFP h0 in every OCR RM component and the SDK fused GDR path."""

import argparse
import json
from pathlib import Path


def main():
    from xhmodel_merak.model_artifact.validation import read_manifest
    from xhmodel_merak.xh_llm.models.ocr_rm.precision import audit_sefp_h1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, manifest = read_manifest(args.artifact_dir)
    if manifest["model_id"] != "ocr-rm-2b-value":
        parser.error("Expected an OCR RM artifact")
    report = audit_sefp_h1(root / manifest["artifact"]["metadata_file"])
    report.update(artifact_dir=str(root), deployment_precision=manifest["parameters"]["precision"]["deployment"])
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "graphs": list(report["graphs"])}))


if __name__ == "__main__":
    main()
