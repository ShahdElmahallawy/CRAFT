"""Stop-level: for every FILTERED stop (structured/description; plain stops skipped),
did CRAFT's route satisfy the gold filter, and if not, why.  -> results/set3_stop_level.xlsx"""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import geopandas as gpd
from evaluation import constraint_checker as cc
from craft.generator.tools.common_handlers import build_graph_from_gpkg

GPKG = str(ROOT/"data/socal/road_network.gpkg")
MODELS = ["gemini-3_1-flash-lite", "gpt-4_1", "openai_gpt-oss-20b"]
gold = cc.load_set3_gold(); recs = list(gold.values())
oracle = {r["query_id"]: r for r in json.loads((ROOT/"evaluation/optimal/results/set3_oracle.json").read_text())}
G = build_graph_from_gpkg(GPKG); pois = gpd.read_file(GPKG, layer="pois")
caches = cc.build_poi_caches(G, pois, poi_types=cc.SET3_POI_TYPES); type_nodes = caches[0]
fc = cc.build_set3_filter_cache(G, pois, recs); qe = cc.build_set3_query_embeddings(recs)

def parser_stop_ok(po, g, i):
    """None if parser fine for stop i, else reason."""
    pt = [t.lower() for t in (po.get("poi_types") or [])]
    if i >= len(pt) or pt[i] != cc.normalize_type(g["poi_types"][i]): return "wrong_poi_type"
    pf = po.get("poi_filters") or []
    x = pf[i] if i < len(pf) else {}
    if isinstance(x, str): x = {"description": x}
    if not isinstance(x, dict) or not x: return "filter_dropped"
    gf = g["poi_filters"][i]
    if any(str(x.get(c, "")).lower() == str(v).lower() for c, v in gf.items()): return None
    if "description" in x and "description" not in gf: return "column_swap"
    return "filter_altered"

rows = []
for model in MODELS:
    for qid, g in gold.items():
        p = ROOT/"outputs/craft"/model/"ordered+desc"/f"{qid}.json"
        d = json.loads(p.read_text()); po = d.get("parser_output") or {}
        m = d.get("metrics") or {}; res = d.get("result") or {}
        route = res.get("route") if (isinstance(res, dict) and m.get("ran")) else None
        route_c = [n for n in (cc.coerce(G, x) for x in (route or [])) if n is not None]
        gtypes = [cc.normalize_type(t) for t in g["poi_types"]]
        pf = cc._set3_parser_filters(po)
        ptypes = [cc.normalize_type(t) for t in (po.get("poi_types") or [])]
        cands = []
        for i, c in enumerate(gtypes):
            nodes = set(type_nodes.get(c, set()))
            if i < len(ptypes) and ptypes[i] == c and i < len(pf) and pf[i]:
                nodes = cc._set3_narrow_by_filter(nodes, pf[i], c, fc, qe)
            cands.append(nodes)
        claims = cc._set3_claim(route_c, gtypes, cands) if route_c else [None]*len(gtypes)
        err = None
        if not m.get("ran"): err = "crash_or_not_run"
        for i, stop in enumerate(g["stops"]):
            kind = stop["filter_kind"]
            if kind not in ("structured", "description"): continue
            v = cc._set3_stop_filter_ok(claims[i], stop, fc, qe) if claims[i] is not None else None
            pr = parser_stop_ok(po, g, i) if po else "parser_failed"
            ostat = oracle[qid]["status"]
            if claims[i] is not None and v is True: out, cause = "SATISFIED", ""
            else:
                if claims[i] is None:
                    out = "NO_ROUTE" if not route_c else "STOP_NOT_REACHED"
                elif v is False: out = "FALLBACK_wrong_poi"
                else: out = "UNVERIFIABLE_no_label"
                if ostat != "optimal": cause = "no_matching_POI_reachable (oracle)"
                elif pr: cause = f"parser:{pr}"
                elif not m.get("ran"): cause = "none: codegen/execute crash"
                elif out == "NO_ROUTE": cause = "none: search returned no route"
                elif out == "STOP_NOT_REACHED": cause = "none: route skipped this stop"
                else: cause = "none: parser correct, route ignored filter"
            rows.append(dict(model=model, query_id=qid, stop=i+1, kind=kind, poi_type=stop["poi_type"],
                             filter=f'{stop.get("column")}={stop.get("value")}' if kind=="structured" else stop.get("description"),
                             outcome=out, cause=cause, parser_stop=pr or "ok", oracle_query_status=ostat))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(pd.crosstab(df.model, df.outcome, margins=True))
print(pd.crosstab([df.model, df.kind], df.outcome))
bad = df[df.outcome != "SATISFIED"]
print(pd.crosstab(bad.cause, bad.model, margins=True))
with pd.ExcelWriter(ROOT/"results/set3_stop_level.xlsx") as w:
    pd.crosstab(df.model, df.outcome).to_excel(w, sheet_name="Outcome")
    pd.crosstab(bad.cause, bad.model).to_excel(w, sheet_name="Why not satisfied")
    df.to_excel(w, sheet_name="Per-Stop", index=False)
