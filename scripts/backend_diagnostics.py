#!/usr/bin/env python3
"""
Backend Diagnostics Utility
===========================
Real-Time Industrial Defect Detection System

Inspects system resources, parses structured application logs (JSON),
validates environment variables, and generates reports/backend_diagnostic_report.md.

Author: prajwaledu802-coder
Date: 2026-07-14
"""

import sys
import os
import json
import time
import platform
from pathlib import Path
from typing import Dict, Any, List

# Bootstrap path resolution
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Attempt to load PyTorch details
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Attempt to load psutil
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def check_system_resources() -> Dict[str, Any]:
    """Gather current CPU, Memory, and Disk stats."""
    stats = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": sys.version.split()[0],
        "cpu_percent": 0.0,
        "memory_percent": 0.0,
        "disk_free_gb": 0.0
    }
    
    if HAS_PSUTIL:
        stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        stats["memory_percent"] = psutil.virtual_memory().percent
        
    # Get disk stats
    try:
        total, used, free = os.statvfs("/") if hasattr(os, "statvfs") else (0, 0, 0)
        if total > 0:
            stats["disk_free_gb"] = round(free * total / (1024 ** 3), 2)
        else:
            # Fallback for Windows
            import shutil
            total, used, free = shutil.disk_usage(str(_PROJECT_ROOT))
            stats["disk_free_gb"] = round(free / (1024 ** 3), 2)
    except Exception:
        pass
        
    return stats


def check_torch_environment() -> Dict[str, Any]:
    """Retrieve PyTorch details and CUDA device availability."""
    info = {
        "installed": HAS_TORCH,
        "version": torch.__version__ if HAS_TORCH else "N/A",
        "cuda_available": torch.cuda.is_available() if HAS_TORCH else False,
        "device_count": torch.cuda.device_count() if (HAS_TORCH and torch.cuda.is_available()) else 0,
        "current_device_name": torch.cuda.get_device_name(0) if (HAS_TORCH and torch.cuda.is_available()) else "N/A"
    }
    return info


def parse_application_logs(log_file: Path) -> Dict[str, Any]:
    """Parse JSON log entries to count log message levels and identify errors."""
    summary = {
        "total_entries": 0,
        "info_count": 0,
        "warning_count": 0,
        "error_count": 0,
        "unstructured_lines": 0,
        "recent_errors": []
    }
    
    if not log_file.exists():
        return summary

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    summary["total_entries"] += 1
                    level = data.get("level", "").upper()
                    if level == "INFO":
                        summary["info_count"] += 1
                    elif level == "WARNING":
                        summary["warning_count"] += 1
                    elif level == "ERROR":
                        summary["error_count"] += 1
                        summary["recent_errors"].append(data)
                except json.JSONDecodeError:
                    summary["unstructured_lines"] += 1
    except Exception as e:
        print(f"Error parsing log file: {e}")
        
    # Cap recent errors
    summary["recent_errors"] = summary["recent_errors"][-5:]
    return summary


def write_diagnostic_report(
    report_path: Path,
    system_stats: Dict[str, Any],
    torch_info: Dict[str, Any],
    log_stats: Dict[str, Any]
) -> None:
    """Compile diagnostic data and save it as reports/backend_diagnostic_report.md."""
    lines = [
        "# Backend Diagnostics & Monitoring Report",
        "",
        f"**Generated at**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. System Resources & Platform",
        "",
        f"- **Operating System**: {system_stats['os']}",
        f"- **Python Version**: {system_stats['python_version']}",
        f"- **CPU Usage**: {system_stats['cpu_percent']}%",
        f"- **Memory Usage**: {system_stats['memory_percent']}%",
        f"- **Disk Free Space**: {system_stats['disk_free_gb']} GB",
        "",
        "## 2. Machine Learning Environment (PyTorch)",
        "",
        f"- **PyTorch Installed**: {'✅ Yes' if torch_info['installed'] else '❌ No'}",
        f"- **PyTorch Version**: {torch_info['version']}",
        f"- **CUDA GPU Available**: {'✅ Yes' if torch_info['cuda_available'] else '❌ No (CPU Mode)'}",
        f"- **CUDA Device Count**: {torch_info['device_count']}",
        f"- **Active GPU Name**: {torch_info['current_device_name']}",
        "",
        "## 3. Log Analytics Summary (Structured JSON)",
        "",
        f"- **Total JSON Log Entries**: {log_stats['total_entries']}",
        f"- **Info Logs**: {log_stats['info_count']}",
        f"- **Warning Logs**: {log_stats['warning_count']}",
        f"- **Error Logs**: {log_stats['error_count']}",
        f"- **Legacy Unstructured Lines**: {log_stats['unstructured_lines']}",
        ""
    ]
    
    if log_stats["error_count"] > 0:
        lines.append("### Recent Application Errors Found:")
        lines.append("")
        for idx, err in enumerate(log_stats["recent_errors"], 1):
            lines.append(f"#### error {idx} [{err.get('timestamp')}]")
            lines.append(f"- **Logger**: `{err.get('logger')}` | **Func**: `{err.get('function')}:{err.get('line')}`")
            lines.append(f"- **Message**: `{err.get('message')}`")
            if "exception" in err:
                lines.append(f"```python\n{err['exception']}\n```")
            lines.append("")
    else:
        lines.append("✅ No recent application errors detected in logs.")
        lines.append("")
        
    lines.append("---")
    lines.append("Report generated automatically by `backend_diagnostics.py`.")
    
    # Save file
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Diagnostics report saved at {report_path}")


def main() -> None:
    print("Gathering backend diagnostic metrics...")
    sys_stats = check_system_resources()
    torch_info = check_torch_environment()
    
    log_file = _PROJECT_ROOT / "logs" / "app.log"
    log_stats = parse_application_logs(log_file)
    
    report_path = _PROJECT_ROOT / "reports" / "backend_diagnostic_report.md"
    write_diagnostic_report(report_path, sys_stats, torch_info, log_stats)


if __name__ == "__main__":
    main()
