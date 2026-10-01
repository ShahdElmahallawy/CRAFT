"""
Generator agent — asks the LLM to fill in the four method bodies (State.start,
Algorithm.expand/dominates/is_goal) that turn tools/template.py into a runnable solver.

Takes either flat input ({src, dst, max_battery, ...}) or parser output
({src, dst, objective, constraints: {...}}) — parser format gets flattened first.

Usage:
  python craft/generator/agent.py --query T001_v01
  python craft/generator/agent.py --query T001_v01 --input-file path/to/parsed.json
"""

import argparse
import ast
import json
import os
import shutil
import sys
import textwrap
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = Path(__file__).resolve().parent
TOOLS_DIR = SHARED_ROOT / "tools"
for _p in [str(REPO_ROOT), str(SHARED_ROOT), str(TOOLS_DIR)]:
    if _p not in sys.path:
        sys.path.append(_p)

from craft.generator.tools.result import save_result  # noqa: E402


TEMPLATE_PATH = TOOLS_DIR / "template.py"
PROBLEM_PATH = SHARED_ROOT / "prompt.md"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "craft" / "generator"


def read_env(key: str) -> str:
    env_candidates = [
        REPO_ROOT / ".env",
        REPO_ROOT / "algorithms+llm" / ".env",
    ]
    for env_path in env_candidates:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            k, v = text.split("=", 1)
            if k.strip() == key:
                return v.strip().strip("'").strip('"')
    return ""


def resolve_input_fields_path(query_name: str) -> Path:
    candidates = [
        REPO_ROOT / "queries" / f"{query_name}.json",
        AGENTS_ROOT / "queries" / f"{query_name}.json",
        SHARED_ROOT / f"{query_name}.json",
        SHARED_ROOT / "queries" / f"{query_name}.json",
        REPO_ROOT / "benchmark" / "socal_queries" / "set1_city_focused" / f"{query_name}.json",
    ]
    benchmark_root = REPO_ROOT / "benchmark"
    if benchmark_root.exists():
        for folder in sorted(benchmark_root.iterdir()):
            if not folder.is_dir():
                continue
            candidates.append(folder / "queries" / f"{query_name}.json")
            candidates.append(folder / f"{query_name}.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_text = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Input file not found for query={query_name}. Tried:\n{candidate_text}")


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse JSON out of an LLM response, stripping markdown fences or stray text if needed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner_lines = lines[1:]
        if inner_lines and inner_lines[-1].strip() == "```":
            inner_lines = inner_lines[:-1]
        stripped = "\n".join(inner_lines).strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for search_text in (text, stripped):
        start = search_text.find("{")
        if start == -1:
            continue
        try:
            obj, _ = decoder.raw_decode(search_text, start)
            return obj
        except json.JSONDecodeError:
            continue

    return json.loads(text)


class LocalLLM:
    def __init__(self, model: str, timeout_s: float = 120.0, debug: bool = False) -> None:
        is_gemini = "gemini" in model.lower()
        if is_gemini:
            self.base_url = (
                os.getenv("GEMINI_BASE_URL", "").strip()
                or read_env("GEMINI_BASE_URL")
                or os.getenv("OPENAI_BASE_URL", "").strip()
                or read_env("OPENAI_BASE_URL")
            )
            self.api_key = (
                os.getenv("GEMINI_API_KEY", "").strip()
                or read_env("GEMINI_API_KEY")
                or os.getenv("OPENAI_API_KEY", "").strip()
                or read_env("OPENAI_API_KEY")
            )
        else:
            self.base_url = (
                os.getenv("OPENAI_BASE_URL", "").strip()
                or read_env("OPENAI_BASE_URL")
            )
            self.api_key = (
                os.getenv("OPENAI_API_KEY", "").strip()
                or read_env("OPENAI_API_KEY")
            )
        self.model = model
        self.timeout_s = timeout_s
        self.debug = debug
        self.enabled = bool(self.base_url and self.model)
        self.last_usage: Dict[str, Any] = {}
        self.last_latency_s: float = 0.0

    def ask_json(self, instruction: str, payload: Dict[str, object]) -> Dict[str, object]:
        if not self.enabled:
            raise RuntimeError("Missing provider base URL or model")
        url = self.base_url.rstrip("/") + "/chat/completions"
        body: Dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ],
        }
        if os.environ.get("QWEN_NO_THINKING"):
            body["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
        }

        max_attempts = max(1, int(os.environ.get("LLM_MAX_RETRIES", "10")))
        raw = None
        t0 = time.perf_counter()
        for attempt in range(1, max_attempts + 1):
            req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == max_attempts:
                    raise RuntimeError(
                        f"LLM request failed at {url} (HTTP {exc.code})"
                        + (f" after {attempt} attempts." if retryable else ".")
                    ) from exc
                time.sleep(self._retry_wait(exc, attempt))
            except urllib.error.URLError as exc:
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"LLM request failed at {url}. Check that the model server is running and reachable."
                    ) from exc
                time.sleep(min(2 ** attempt, 30.0))
        parsed = json.loads(raw)
        self.last_usage = parsed.get("usage", {})
        self.last_latency_s = round(time.perf_counter() - t0, 3)
        text = parsed["choices"][0]["message"]["content"]
        if self.debug:
            print(f"[debug] received response from model={self.model}")
        return _extract_json(text)

    @staticmethod
    def _retry_wait(exc: "urllib.error.HTTPError", attempt: int) -> float:
        """Seconds to wait before retrying a 429/5xx, honoring server hints."""
        headers = getattr(exc, "headers", None)
        if headers is not None:
            ra = headers.get("Retry-After")
            if ra:
                try:
                    return min(float(ra) + 0.5, 60.0)
                except ValueError:
                    pass
        try:
            body = exc.read().decode("utf-8", "ignore")
        except Exception:
            body = ""
        m = re.search(r"try again in ([\d.]+)\s*(ms|s)", body)
        if m:
            val = float(m.group(1))
            secs = val / 1000.0 if m.group(2) == "ms" else val
            return min(secs + 0.5, 60.0)
        return min(2 ** attempt, 30.0)


def method_wrapper(name: str, body: str) -> str:
    return "def _tmp(self, a=None, b=None, state=None):\n" + textwrap.indent(body.rstrip() + "\n", "    ")


def normalize_body(body: str) -> str:
    text = body.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return textwrap.dedent(text).strip()


def validate_body(name: str, body: str) -> None:
    body = normalize_body(body)
    if not body.strip():
        raise ValueError(f"{name} body is empty")
    lowered = body.lower()
    banned_tokens = ["import ", "open(", "exec(", "eval(", "urllib", "requests", "__import__"]
    if any(token in lowered for token in banned_tokens):
        raise ValueError(f"{name} body contains banned tokens")
    try:
        tree = ast.parse(method_wrapper(name, body))
    except SyntaxError as exc:
        raise ValueError(f"{name} body is invalid Python: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if isinstance(node, ast.FunctionDef) and node.name == "_tmp":
                continue
            raise ValueError(f"{name} body contains disallowed definitions")


def indent_body(body: str) -> str:
    return textwrap.indent(normalize_body(body), "        ")


def validate_state_start_init_body(body: str) -> None:
    body = normalize_body(body)
    if not body.strip():
        raise ValueError("state_start_init_body is empty")
    lowered = body.lower()
    banned_tokens = ["import ", "open(", "exec(", "eval(", "urllib", "requests", "__import__"]
    if any(token in lowered for token in banned_tokens):
        raise ValueError("state_start_init_body contains banned tokens")
    if "self." in lowered:
        raise ValueError("state_start_init_body must not use self; use init_context and cls")
    # init_context can go unused if constraints are all constants (e.g. poi_idx=0)
    wrapped = "def _tmp(origin, init_context):\n" + textwrap.indent(body.rstrip() + "\n", "    ")
    try:
        tree = ast.parse(wrapped)
    except SyntaxError as exc:
        raise ValueError(f"state_start_init_body is invalid Python: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if isinstance(node, ast.FunctionDef) and node.name == "_tmp":
                continue
            raise ValueError("state_start_init_body contains disallowed definitions")


def build_instruction() -> str:
    return (
        "You generate one State start snippet and three method bodies for a fixed Python template. "
        "Return JSON only with keys: state_start_init_body, dominates_body, is_goal_body, expand_body. "
        "State class is already provided in template with fields self.node, self.prev, self.constraints, self._objective (exposed via property objective). "
        "You decide initial objective and initial constraints content in start(origin, init_context). "
        "CRITICAL: state_start_init_body runs inside a @classmethod — there is NO self variable. "
        "Read all values from init_context using init_context['key'] and ONLY keys that exist in input_fields. "
        "Never write self.anything inside state_start_init_body. Use only origin, init_context, and cls. "
        "CRITICAL: expand must iterate graph edges using NetworkX MultiDiGraph out_edges API, e.g. "
        "for _, neighbor, _, data in self.G.out_edges(state.node, keys=True, data=True). "
        "Do not use self.G[state.node].items(). "
        "Each value must be raw Python statements only, not markdown. "
        "Do not include method signatures. Do not add imports. Do not add helper functions. "
        "Use object attributes exactly as provided in the problem; do not invent alternative attribute names. "
        "Use only self attributes and provided helper functions in method bodies. "
        "For EV charging, charging stations are POIs in self.type_nodes.get('charging_station', set()). "
        "Use 'neighbor in self.type_nodes.get(\"charging_station\", set())' to detect charger nodes. "
        "Do not use self.charger_nodes — it does not exist. "
        "Infer the algorithm logic from the problem description and constraints. "
        "All constraints in input_fields are hard requirements — satisfy every one. "
        "If nl_query is present, use it only to disambiguate the objective (time vs distance) "
        "or understand priority ordering among competing goals — never to drop constraints. "
        "Do not copy a canned solution unless it follows from the provided information."
    )


def _load_poi_schema(repo_root: Path) -> Dict[str, object]:
    """Load poi_attributes from schema.json; return empty dict if not found."""
    for candidate in [
        repo_root / "data" / "socal" / "schema.json",
    ]:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8")).get("poi_attributes", {})
            except Exception:
                pass
    return {}


def build_payload(problem_text: str, input_fields: Dict[str, object]) -> Dict[str, object]:
    # surface nl_query at the top level so the LLM sees it for priorities/nuances
    nl_query = None
    clean_input = {}
    if isinstance(input_fields, dict):
        nl_query = input_fields.get("nl_query")
        clean_input = {k: v for k, v in input_fields.items() if k != "nl_query"}
    else:
        clean_input = input_fields

    payload: Dict[str, object] = {
        "task": "Fill State objective/start snippets and three Algorithm method bodies",
        "problem_description": problem_text,
    }
    if nl_query:
        payload["nl_query"] = nl_query
        payload["nl_query_note"] = (
            "Read nl_query to understand the user's intent. "
            "All constraints in input_fields are hard requirements — never skip or downgrade them. "
            "Use nl_query only to disambiguate the objective (time vs distance) "
            "or understand priority ordering among competing goals."
        )
    payload["input_fields"] = clean_input
    poi_schema = _load_poi_schema(REPO_ROOT)
    if poi_schema:
        payload["poi_attribute_schema"] = poi_schema
        payload["poi_attribute_schema_note"] = (
            "poi_attribute_schema describes the POI columns available in self.pois_gdf. "
            "Each entry has a 'kind' (static_filter or state_dependent), optional 'sample_values', "
            "and for state_dependent columns an 'activation_field' that names the query field that enables the check. "
            "If a state_dependent column's activation_field is present in input_fields, implement that check in expand()."
        )
    payload["input_context_rule"] = (
        "Use only fields present in input_fields for objective and constraints. "
        "Do not assume missing fields exist."
    )
    payload["function_contracts"] = {
        "state_start_init_body": {
            "placement": "inside @classmethod def start(origin, init_context):",
            "must_return": "State(origin, None, constraints_dict, objective)",
            "runtime_contract": [
                "constraints must be a dict containing resource/progress variables inferred from the problem",
                "objective must be initialized from start-of-search information",
            ],
        },
        "dominates": {
            "signature": "def dominates(self, a: State, b: State) -> bool:",
            "must_return": "bool",
            "purpose": "Return whether state a should dominate and prune state b.",
        },
        "is_goal": {
            "signature": "def is_goal(self, state: State) -> bool:",
            "must_return": "bool",
            "purpose": "Return whether state is a valid solution state.",
        },
        "expand": {
            "signature": "def expand(self, state: State) -> Iterable[State]:",
            "must_yield": "State instances",
            "purpose": "Yield successor states that respect all constraints.",
            "runtime_contract": [
                "Use self.G.out_edges(state.node, keys=True, data=True) to iterate edges",
                "data is a single edge-attribute dict compatible with edge_weight(data, ...)",
                "Do not use self.G[state.node].items()",
            ],
        },
    }
    payload["output_schema_example"] = {
        "state_start_init_body": "constraints = {...}\\nobjective = 0.0\\nreturn cls(origin, None, constraints, objective)",
        "dominates_body": "return <python boolean logic using a and b>",
        "is_goal_body": "return <python boolean logic using state>",
        "expand_body": "for ...:\n    ...\n    yield State(...)",
    }
    return payload


def _query_meta(input_fields: object) -> Dict[str, object]:
    if isinstance(input_fields, dict):
        return input_fields
    if isinstance(input_fields, list) and input_fields and isinstance(input_fields[0], dict):
        return input_fields[0]
    return {}


def flatten_parser_output(input_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Un-nest parser output ({constraints: {...}}) into the flat shape the template expects.
    Values already arrive in base units (meters, seconds) from normalize_constraints, so this
    just moves keys up a level. Already-flat input passes through untouched."""
    if "constraints" not in input_fields:
        return input_fields  # already flat

    flat = {k: v for k, v in input_fields.items() if k not in ("constraints", "_parse_meta")}
    for k, v in input_fields.get("constraints", {}).items():
        flat[k] = v
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate algorithm logic for a named query.")
    parser.add_argument("--query", required=True, help="Query name (e.g. query1). Writes output to {query}_output/")
    parser.add_argument("--input-file", default=None, help="Explicit path to input JSON (skips path resolution)")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "Qwen/Qwen3-32B-AWQ"), help="LLM model")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="LLM timeout")
    parser.add_argument("--problem-file", default=None, help="Problem description markdown file (default: problem.md)")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    args = parser.parse_args()

    query_name = args.query
    input_path = Path(args.input_file) if args.input_file else resolve_input_fields_path(query_name)
    output_dir = OUTPUTS_DIR / f"{query_name}_output"
    out_path = output_dir / "generated.py"
    raw_response_path = output_dir / "raw_response.json"
    summary_out_path = OUTPUTS_DIR / f"{query_name}_output.json"

    problem_path = Path(args.problem_file) if args.problem_file else PROBLEM_PATH

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    problem_text = problem_path.read_text(encoding="utf-8")
    input_fields_raw = json.loads(input_path.read_text(encoding="utf-8"))

    if isinstance(input_fields_raw, list):
        input_fields = input_fields_raw  # benchmark array format, pass as-is
    else:
        input_fields = flatten_parser_output(input_fields_raw)

    output_dir.mkdir(exist_ok=True)
    (output_dir / "input.json").write_text(
        json.dumps(input_fields, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    llm = LocalLLM(model=args.model, timeout_s=args.timeout_s, debug=args.debug)
    response = llm.ask_json(build_instruction(), build_payload(problem_text, input_fields))

    (output_dir / "llm_meta.json").write_text(
        json.dumps({"model": args.model, "latency_s": llm.last_latency_s, "tokens": llm.last_usage}, indent=2),
        encoding="utf-8",
    )
    raw_response_path.write_text(json.dumps(response, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.debug:
        print(f"[debug] raw response saved to {raw_response_path}")

    required_keys = [
        "state_start_init_body",
        "dominates_body",
        "is_goal_body",
        "expand_body",
    ]
    for key in required_keys:
        if key not in response or not isinstance(response[key], str):
            raise RuntimeError(f"LLM response missing required string key: {key}")

    response["state_start_init_body"] = normalize_body(response["state_start_init_body"])
    response["dominates_body"] = normalize_body(response["dominates_body"])
    response["is_goal_body"] = normalize_body(response["is_goal_body"])
    response["expand_body"] = normalize_body(response["expand_body"])

    validate_state_start_init_body(response["state_start_init_body"])
    validate_body("dominates", response["dominates_body"])
    validate_body("is_goal", response["is_goal_body"])
    validate_body("expand", response["expand_body"])

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    generated = template
    generated = generated.replace(
        "        __STATE_START_INIT_BODY__", indent_body(response["state_start_init_body"])
    )
    generated = generated.replace("        __DOMINATES_BODY__", indent_body(response["dominates_body"]))
    generated = generated.replace("        __IS_GOAL_BODY__", indent_body(response["is_goal_body"]))
    generated = generated.replace("        __EXPAND_BODY__", indent_body(response["expand_body"]))

    ast.parse(generated)
    out_path.write_text(generated, encoding="utf-8")

    query_meta = _query_meta(input_fields)
    result_doc = {
        "query_name": query_name,
        "status": "generated",
        "template_file": query_meta.get("template_file"),
        "template_id": query_meta.get("template_id"),
        "algorithm": query_meta.get("algorithm"),
        "input_file": str(input_path),
        "generated_file": str(out_path),
        "raw_response_file": str(raw_response_path),
    }
    save_result(result_doc, str(summary_out_path))

    print(f"Generated file written to: {out_path}")
    print(f"Raw response saved to:     {raw_response_path}")
    print(f"Input copied to:           {output_dir / 'input.json'}")
    print(f"Summary result saved to:   {summary_out_path}")


if __name__ == "__main__":
    main()
