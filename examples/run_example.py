"""
Runs ONE query through the full CRAFT pipeline (parse -> generate -> execute)
and saves the result where you can actually look at it.

Use this to try the system out before running anything at benchmark scale.

Usage:
  python examples/run_example.py
  python examples/run_example.py --query "Drive from (33.83, -117.95) to (33.85, -117.86), stop at a pharmacy."
  python examples/run_example.py --query-file my_query.json
  python examples/run_example.py --model gpt-4.1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from craft.generator.agent import LocalLLM
from craft.parser.tools import CatalogResolver
from craft.generator.tools.common_handlers import build_graph_from_gpkg
from craft.generator.tools.result import enrich_output, load_output
from craft.orchestrator import run_parser, run_generator, run_execute, GENERATOR_OUTPUTS

DEFAULT_GPKG = ROOT / "data" / "socal" / "road_network.gpkg"
DEFAULT_QUERY_FILE = Path(__file__).resolve().parent / "example_query.json"
PARSER_PROMPT = ROOT / "craft" / "parser" / "prompt.md"
OUT_DIR = Path(__file__).resolve().parent / "output"


def _weight_key_for(flat: dict) -> str:
    """Mirrors template.py's own objective->weight_key mapping, so the
    enriched objective_value matches what the search actually optimized."""
    objective = str(flat.get("objective", "")).strip().lower()
    if objective == "minimize_distance":
        return "length"
    if objective == "minimize_travel_time":
        return "travel_time"
    return flat.get("weight_key", "travel_time")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=None, help="A natural-language query string")
    parser.add_argument("--query-file", default=None,
                        help=f"JSON file with {{query_id, nl_query}} (default: {DEFAULT_QUERY_FILE.name})")
    parser.add_argument("--query-id", default="example_query", help="Used only with --query")
    parser.add_argument("--gpkg", default=str(DEFAULT_GPKG), help="Path to road_network.gpkg")
    parser.add_argument("--model", default="gemini-3.1-flash-lite", help="LLM model for parsing + generation")
    parser.add_argument("--timeout-generate", type=float, default=120.0)
    parser.add_argument("--timeout-execute", type=float, default=120.0)
    args = parser.parse_args()

    if args.query:
        query = {"query_id": args.query_id, "nl_query": args.query}
    else:
        query_file = Path(args.query_file) if args.query_file else DEFAULT_QUERY_FILE
        query = json.loads(query_file.read_text(encoding="utf-8"))

    gpkg_path = str(Path(args.gpkg).resolve())
    if not Path(gpkg_path).exists():
        sys.exit(
            f"GPKG not found: {gpkg_path}\n"
            "Download the dataset and place it there — see examples/README.md."
        )

    query_id = query["query_id"]
    print(f"Query [{query_id}]: {query['nl_query']}\n")

    print("Loading graph and POIs ...")
    G = build_graph_from_gpkg(gpkg_path)
    import geopandas as gpd
    pois_gdf = gpd.read_file(gpkg_path, layer="pois")
    print(f"  {G.number_of_nodes():,} nodes, {len(pois_gdf):,} POIs\n")

    resolver = CatalogResolver(gpkg_path=gpkg_path)
    llm = LocalLLM(model=args.model, timeout_s=args.timeout_generate)
    if not llm.enabled:
        sys.exit(
            "No LLM endpoint configured. Set GEMINI_API_KEY/GEMINI_BASE_URL (or OPENAI_*) "
            "in a .env file at the project root — see examples/README.md."
        )

    print("Step 1/3: parsing ...")
    with tempfile.TemporaryDirectory() as staging:
        staged_path = run_parser(
            query, work_dir=Path(staging), gpkg=gpkg_path,
            llm=llm, resolver=resolver,
            problem_text=PARSER_PROMPT.read_text(encoding="utf-8"),
            G=G,
        )
        flat = json.loads(staged_path.read_text(encoding="utf-8"))
        print(f"  src={flat.get('src')}  dst={flat.get('dst')}  objective={flat.get('objective')}\n")

        print("Step 2/3: generating (LLM fills the Dijkstra template) ...")
        gen_result = run_generator(
            scratch_id=query_id, input_file=str(staged_path),
            model=args.model, timeout_s=args.timeout_generate, python_exe=sys.executable,
        )
    if not gen_result["ok"]:
        sys.exit(f"Generation failed: {gen_result['error']}")
    print(f"  done in {gen_result['elapsed_s']:.1f}s\n")

    print("Step 3/3: executing the generated search ...")
    exec_result = run_execute(
        scratch_id=query_id, timeout_s=args.timeout_execute, python_exe=sys.executable,
    )
    if not exec_result["ok"]:
        sys.exit(f"Execution failed: {exec_result['error']}")
    print(f"  done in {exec_result['elapsed_s']:.1f}s\n")

    raw_dir = GENERATOR_OUTPUTS / f"{query_id}_output"
    enrich_output(raw_dir, G, pois_gdf, _weight_key_for(flat))
    result = load_output(str(raw_dir / "output.json"))

    dest_dir = OUT_DIR / query_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in ("input.json", "generated.py", "output.json", "llm_meta.json",
                 "raw_response.json", "output.gpkg"):
        src = raw_dir / name
        if src.exists():
            shutil.copy2(src, dest_dir / name)

    print("=" * 60)
    if result.get("route") is None:
        print("No feasible route found.")
    else:
        stops = result.get("poi_stops", [])
        print(f"Route found: {len(result['route'])} nodes, {len(stops)} POI stop(s)")
        print(f"Objective ({result.get('weight_key')}): {result.get('objective')}")
        for s in stops:
            print(f"  - {s.get('type')}: {s.get('name') or '(unnamed)'}")
    print(f"\nFull output saved to: {dest_dir}")


if __name__ == "__main__":
    main()
