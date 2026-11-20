#!/usr/bin/env python3
"""
DQM CRV Anomaly Dataset Matrix Generator

Pulls metrics from the DQM MCP server (http://mu2eaigpvm01:8000/list) and writes the
run/subrun x metric matrix CSV consumed by the pipeline.

Usage:
  python -m mu2e_tranad.export_from_dqm
  python -m mu2e_tranad.export_from_dqm --source crvreco,kpp_crv,file,005-001 --output out.csv
"""

import argparse
import sys

from .dqm_mcp import DEFAULT_URL, DqmMcpError, fetch_matrix

SOURCE_STR = "crvreco,kpp_crv,file,004-000"
OUTPUT_FILE = "dqm_crv_anomaly_dataset_004-000.csv"


def main():
    parser = argparse.ArgumentParser(description="Export a DQM source to a metric-matrix CSV via the DQM MCP server")
    parser.add_argument("--source", default=SOURCE_STR,
                        help=f"process,stream,aggregation,version (default: {SOURCE_STR})")
    parser.add_argument("--group", default="CRV", help="DQM value group (groupx) to export (default: CRV)")
    parser.add_argument("--output", default=OUTPUT_FILE, help=f"Output CSV (default: {OUTPUT_FILE})")
    parser.add_argument("--after", help="Only intervals starting after this ISO UTC time")
    parser.add_argument("--before", help="Only intervals ending before this ISO UTC time")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"DQM MCP endpoint (default: {DEFAULT_URL})")
    args = parser.parse_args()

    try:
        df = fetch_matrix(args.source, groupx=args.group, url=args.url,
                          start_time_after_iso_utc=args.after,
                          end_time_before_iso_utc=args.before)
    except DqmMcpError as e:
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(1)

    df.to_csv(args.output, index=False)
    print(f"✓ Anomaly dataset saved to {args.output}")


if __name__ == "__main__":
    main()
