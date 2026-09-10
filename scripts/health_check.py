#!/usr/bin/env python3
"""
scripts/health_check.py
========================
Standalone health check script for deployment pipelines and load balancers.

USAGE:
  # Basic check (returns exit code 0=healthy, 1=unhealthy)
  python scripts/health_check.py

  # Check a specific host/port
  python scripts/health_check.py --host api.example.com --port 8000

  # Detailed output
  python scripts/health_check.py --verbose

  # Used in Docker HEALTHCHECK:
  HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python scripts/health_check.py --host localhost || exit 1

  # Used in Kubernetes liveness probe:
  livenessProbe:
    httpGet:
      path: /health
      port: 8000
    initialDelaySeconds: 30
    periodSeconds: 10

EXIT CODES:
  0 — All components healthy
  1 — Critical component(s) down
  2 — Degraded (non-critical components down, API still serving)
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from typing import Any


def check_health(host: str, port: int, timeout: int = 10) -> dict[str, Any]:
    """Call the /health endpoint and return parsed response."""
    url = f"http://{host}:{port}/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "health-check/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return {"http_status": resp.status, "body": data, "error": None}
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = {}
        return {"http_status": e.code, "body": body, "error": str(e)}
    except urllib.error.URLError as e:
        return {"http_status": 0, "body": {}, "error": str(e.reason)}
    except Exception as e:
        return {"http_status": 0, "body": {}, "error": str(e)}


def print_status(label: str, ok: bool, detail: str = "") -> None:
    icon = "✅" if ok else "❌"
    line = f"  {icon} {label}"
    if detail:
        line += f" — {detail}"
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM Platform health check")
    parser.add_argument("--host",    default="localhost",   help="API host")
    parser.add_argument("--port",    default=8000, type=int, help="API port")
    parser.add_argument("--timeout", default=10,  type=int, help="Request timeout (s)")
    parser.add_argument("--verbose", action="store_true",   help="Verbose output")
    args = parser.parse_args()

    print(f"Checking LLM Platform health at {args.host}:{args.port}...")

    result = check_health(args.host, args.port, args.timeout)

    if result["error"]:
        print(f"  ❌ Cannot reach API: {result['error']}")
        return 1

    body = result["body"]
    http_status = result["http_status"]

    overall_status = body.get("status", "unknown")
    components     = body.get("components", {})
    version        = body.get("version", "unknown")
    environment    = body.get("environment", "unknown")
    uptime         = body.get("uptime_seconds", 0)

    print(f"\nPlatform: {version} ({environment})")
    print(f"Uptime:   {uptime:.0f}s ({uptime/3600:.1f}h)")
    print(f"Status:   {overall_status.upper()}")
    print("\nComponents:")

    # Critical components — platform is down without them
    critical = {"postgres", "redis"}
    all_critical_ok = True
    any_down = False

    for name, is_ok in components.items():
        is_critical = name in critical
        detail = "CRITICAL" if is_critical and not is_ok else ""
        print_status(name, is_ok, detail)
        if not is_ok:
            any_down = True
            if is_critical:
                all_critical_ok = False

    if args.verbose:
        print(f"\nHTTP Status: {http_status}")
        print(f"Raw response: {json.dumps(body, indent=2)}")

    print()

    # Determine exit code
    if not all_critical_ok:
        print("❌ UNHEALTHY — critical component(s) down")
        return 1
    elif any_down:
        print("⚠️  DEGRADED — non-critical component(s) down")
        return 2
    else:
        print("✅ HEALTHY — all components operational")
        return 0


if __name__ == "__main__":
    sys.exit(main())