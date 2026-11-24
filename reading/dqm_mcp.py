"""
DQM MCP Client

Fetches Mu2e DQM metrics from the DQM MCP server (see http://mu2eaigpvm01:8000/list)
and assembles them into the wide run/subrun x metric matrix used by the pipeline.

Only uses the standard library, so it works in any Python >= 3.8; pandas is only
needed for fetch_matrix().
"""

import itertools
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

DEFAULT_URL = os.environ.get("DQM_MCP_URL", "http://mu2eaigpvm01.fnal.gov:8001/mcp")
PROTOCOL_VERSION = "2025-03-26"

# Large enough to cover the full history of any source
ALL_HISTORY_DAYS = 100000
# The server caps `limit` at 1000 rows per call; larger requests are silently truncated
PAGE_SIZE = 1000
# Max metric rows the server scans per query; its default (2000) truncates long sources
SCAN_LIMIT = 1000000


class DqmMcpError(RuntimeError):
    pass


class DqmMcpClient:
    """Minimal JSON-RPC client for the streamable-HTTP DQM MCP server."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 120.0):
        self.url = url
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self._ids = itertools.count(1)  # thread-safe id generator
        self._initialize()

    def _post(self, payload: dict) -> Optional[dict]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            self.session_id = resp.headers.get("mcp-session-id", self.session_id)
            body = resp.read().decode()

        if "id" not in payload:  # notification, no response expected
            return None

        # Server may answer with plain JSON or a server-sent-events stream
        if resp.headers.get("Content-Type", "").startswith("text/event-stream"):
            for line in body.splitlines():
                if line.startswith("data:"):
                    msg = json.loads(line[5:].strip())
                    if msg.get("id") == payload["id"]:
                        return msg
            raise DqmMcpError(f"No response for request {payload['id']} in event stream")
        return json.loads(body)

    def _request(self, method: str, params: Optional[dict] = None) -> dict:
        msg = self._post({"jsonrpc": "2.0", "id": next(self._ids), "method": method,
                          "params": params or {}})
        if "error" in msg:
            raise DqmMcpError(f"{method} failed: {msg['error']}")
        return msg["result"]

    def _initialize(self) -> None:
        self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "mu2e-tranad", "version": "0.2.0"},
        })
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call_tool(self, name: str, **arguments) -> dict:
        """Call an MCP tool and return its structured JSON output."""
        args = {k: v for k, v in arguments.items() if v is not None}
        result = self._request("tools/call", {"name": name, "arguments": args})
        if result.get("isError"):
            text = " ".join(c.get("text", "") for c in result.get("content", []))
            raise DqmMcpError(f"Tool {name} returned an error: {text}")
        if "structuredContent" in result:
            return result["structuredContent"]
        return json.loads(result["content"][0]["text"])

    # ------------------------------------------------------------------
    # Convenience wrappers around the DQM tools
    # ------------------------------------------------------------------

    def list_sources(self, process=None, stream=None, aggregation=None, version=None) -> List[dict]:
        return self.call_tool("list_sources", process=process, stream=stream,
                              aggregation=aggregation, version=version, limit=1000)["results"]

    def list_values(self, groupx=None, subgroup=None, namex=None) -> List[dict]:
        rows, offset = [], 0
        while True:
            page = self.call_tool("list_values", groupx=groupx, subgroup=subgroup,
                                  namex=namex, limit=PAGE_SIZE, offset=offset)["results"]
            rows.extend(page)
            if len(page) < PAGE_SIZE:
                return rows
            offset += PAGE_SIZE

    def query_metrics(self, sid: int, vid: int, metric_table: str = "numbers",
                      recent_days: int = ALL_HISTORY_DAYS, **filters) -> List[dict]:
        """Full timeline of one variable (vid) for one source (sid), all pages."""
        rows, offset = [], 0
        while True:
            out = self.call_tool("query_metrics", sid=sid, vid=vid, metric_table=metric_table,
                                 recent_days=recent_days, limit=PAGE_SIZE, offset=offset,
                                 expand_ids=False, scan_limit=SCAN_LIMIT, **filters)
            if not out["counts"].get("scan_complete", True):
                raise DqmMcpError(f"Incomplete scan for sid={sid} vid={vid} (raise SCAN_LIMIT "
                                  f"or narrow the time range): {out['counts']}")
            page = out["results"]
            rows.extend(page)
            total = out["counts"].get("total_rows")
            if not page or (total is not None and len(rows) >= total) or \
                    (total is None and len(page) < PAGE_SIZE):
                return rows
            offset += len(page)

    def resolve_source(self, source_str: str) -> dict:
        """Resolve 'process,stream,aggregation,version' (as used by DqmTool) to a source row."""
        parts = [p.strip() or None for p in source_str.split(",")]
        parts += [None] * (4 - len(parts))
        process, stream, aggregation, version = parts[:4]
        matches = self.list_sources(process, stream, aggregation, version)
        if len(matches) != 1:
            found = [f"{m['process']},{m['stream']},{m['aggregation']},{m['version']}"
                     for m in matches]
            raise DqmMcpError(f"Source '{source_str}' matched {len(matches)} sources: {found}")
        return matches[0]


def fetch_matrix(source_str: str,
                 groupx: str = "CRV",
                 url: str = DEFAULT_URL,
                 workers: int = 8,
                 start_time_after_iso_utc: Optional[str] = None,
                 end_time_before_iso_utc: Optional[str] = None,
                 verbose: bool = True):
    """
    Build the run/subrun x metric DataFrame for one DQM source.

    Columns: run, subrun, timestamp, then one column per metric named
    '{groupx}_{subgroup}_{namex}' -- the same layout export_from_dqm.py has always written.
    """
    import pandas as pd

    client = DqmMcpClient(url)
    source = client.resolve_source(source_str)
    sid = source["sid"]
    values = client.list_values(groupx=groupx)
    if verbose:
        print(f"Source sid={sid} ({source_str}); {len(values)} {groupx} metrics in catalog")

    def fetch(value: dict) -> Tuple[dict, List[dict]]:
        return value, client.query_metrics(
            sid, value["vid"],
            start_time_after_iso_utc=start_time_after_iso_utc,
            end_time_before_iso_utc=end_time_before_iso_utc,
        )

    matrix: Dict[Tuple[int, int], dict] = {}
    features: List[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for value, rows in pool.map(fetch, values):
            feat = f"{value['groupx']}_{value['subgroup']}_{value['namex']}"
            if verbose:
                print(f"  [{value['vid']:>4}] {feat:<55} {len(rows):>6} rows")
            if not rows:
                continue
            features.append(feat)
            for r in rows:
                # Older servers nest the interval under "interval"; newer ones return it flat
                iv = r.get("interval", r)
                key = (iv["start_run"], iv["start_subrun"])
                entry = matrix.setdefault(key, {
                    "run": iv["start_run"],
                    "subrun": iv["start_subrun"],
                    "timestamp": iv.get("start_time") or "",
                })
                entry[feat] = r["valuex"]

    if not matrix:
        raise DqmMcpError(f"No {groupx} metrics found for source '{source_str}'")

    df = pd.DataFrame([matrix[k] for k in sorted(matrix)],
                      columns=["run", "subrun", "timestamp"] + features)
    if verbose:
        print(f"✓ Matrix: {len(df)} run/subrun states x {len(features)} features")
    return df
