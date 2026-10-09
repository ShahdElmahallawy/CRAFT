"""
orchestrator.py — Parser → Generator → Execute pipeline steps.

Owns the three per-query steps; called by scripts/run_craft.py.
  1. run_parser   : NL query → staged input.json
  2. run_generator: staged input.json → generated.py  (LLM fills template)
  3. run_execute  : generated.py → output.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
for _p in [str(ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from craft.generator.agent import LocalLLM, flatten_parser_output
from craft.parser.agent import parse as parser_parse
from craft.parser.tools import CatalogResolver

GENERATOR_AGENT  = ROOT / "craft" / "generator" / "agent.py"
GENERATOR_OUTPUTS = ROOT / "outputs" / "craft" / "generator"
CRAFT_TOOLS_DIR  = ROOT / "craft" / "generator" / "tools"


# Step 1: Parse

def run_parser(
    query: Dict[str, Any],
    *,
    work_dir: Path,
    gpkg: str,
    llm: LocalLLM,
    resolver: CatalogResolver,
    problem_text: str,
    G=None,
    stage_name: Optional[str] = None,
) -> Path:
    """
    Parse NL query → flat structured JSON → write to work_dir.

    Falls back to the query's own src/dst fields when the parser doesn't
    resolve them. Returns the path to the staged JSON file.
    Raises on parse failure.
    """
    query_id = query.get("query_id", "unknown")
    nl_query = query.get("nl_query", "")

    parsed = parser_parse(
        query_id, nl_query,
        llm=llm, resolver=resolver, problem_text=problem_text, G=G,
    )
    parsed.pop("_parse_meta", None)

    if not parsed.get("src"):
        parsed["src"] = query.get("src")
    if not parsed.get("dst"):
        parsed["dst"] = query.get("dst")

    parsed["gpkg"] = gpkg
    parsed.setdefault("template_file", query.get("template_file"))

    flat = flatten_parser_output(parsed)

    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / f"{stage_name or query_id}.json"
    out_path.write_text(json.dumps(flat, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


# Step 2: Generate

def run_generator(
    scratch_id: str,
    input_file: str,
    model: str,
    timeout_s: float,
    python_exe: str,
) -> Dict[str, Any]:
    """
    Call generator agent to fill the Dijkstra template from input_file.
    Writes generated.py + llm_meta.json under GENERATOR_OUTPUTS/{scratch_id}_output/.
    Returns {"ok": bool, "error": str|None, "elapsed_s": float}.
    """
    cmd = [
        python_exe, str(GENERATOR_AGENT),
        "--query", scratch_id,
        "--input-file", input_file,
        "--model", model,
        "--timeout-s", str(timeout_s),
    ]
    _prompt_override = os.environ.get("CRAFT_PROMPT_FILE")
    if _prompt_override:
        cmd += ["--problem-file", _prompt_override]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s + 10, cwd=str(ROOT),
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            return {"ok": False, "error": proc.stderr.strip() or proc.stdout.strip(), "elapsed_s": elapsed}
        return {"ok": True, "error": None, "elapsed_s": elapsed}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "generate timeout", "elapsed_s": time.perf_counter() - t0}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "elapsed_s": time.perf_counter() - t0}


# Step 3: Execute

def run_execute(
    scratch_id: str,
    timeout_s: float,
    python_exe: str,
) -> Dict[str, Any]:
    """
    Execute generated.py for scratch_id. Writes output.json in the same dir.
    Returns {"ok": bool, "error": str|None, "elapsed_s": float}.
    """
    generated = GENERATOR_OUTPUTS / f"{scratch_id}_output" / "generated.py"
    if not generated.exists():
        return {"ok": False, "error": f"generated.py not found: {generated}", "elapsed_s": 0.0}

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), str(CRAFT_TOOLS_DIR), existing]))

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [python_exe, str(generated)],
            capture_output=True, text=True,
            timeout=timeout_s, cwd=str(generated.parent), env=env,
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            return {"ok": False, "error": proc.stderr.strip() or proc.stdout.strip(), "elapsed_s": elapsed}
        return {"ok": True, "error": None, "elapsed_s": elapsed}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "execute timeout", "elapsed_s": time.perf_counter() - t0}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "elapsed_s": time.perf_counter() - t0}
    