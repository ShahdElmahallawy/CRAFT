"""Joins results/craft_set3.xlsx with evaluation/optimal/results/set3_oracle.json and
attributes every non-solved query to a cause. -> results/set3_oracle_gap.xlsx"""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
oracle = {r["query_id"]: r for r in json.loads((ROOT/"evaluation/optimal/results/set3_oracle.json").read_text())}
gold = {r["query_id"]: r for r in json.loads((ROOT/"benchmark/socal_queries/set3_descriptions/set3_meta.json").read_text())}
df = pd.read_excel(ROOT/"results/craft_set3.xlsx", "Results", header=2)
MD = {"gemini-3_1-flash-lite": "gemini-3_1-flash-lite", "gpt-4_1": "gpt-4_1", "openai_gpt-oss-20b": "openai_gpt-oss-20b"}

def parser_diag(model, qid):
    p = ROOT/"outputs/craft"/model/"ordered+desc"/f"{qid}.json"
    if not p.exists(): return "no_record"
    po = (json.loads(p.read_text()).get("parser_output") or {})
    g = gold[qid]
    if not po: return "parser_failed"
    if [t.lower() for t in po.get("poi_types", [])] != [t.lower() for t in g["poi_types"]]: return "wrong_poi_types"
    pf = po.get("poi_filters") or []
    issues = []
    for i, gf in enumerate(g["poi_filters"]):
        if not gf: continue
        x = pf[i] if i < len(pf) and isinstance(pf[i], dict) else (({"description": pf[i]}) if i < len(pf) and isinstance(pf[i], str) else {})
        if not x: issues.append("filter_dropped")
        elif any(str(x.get(c, "")).lower() == str(v).lower() for c, v in gf.items()): pass
        elif "description" in x and "description" not in gf: issues.append("column_swap")
        else: issues.append("filter_altered")
    return "ok" if not issues else "+".join(sorted(set(issues)))

rows = []
for _, r in df.iterrows():
    o = oracle[r.query_id]; feas_o = o["status"] == "optimal"
    if r.solved_correctly is True or r.solved_correctly == True:
        cause = "SOLVED" + ("_optimal" if feas_o and abs(r.objective_value - o["optimal_distance_m"]) < 1 else "_suboptimal")
    elif not feas_o: cause = "UNSOLVABLE_no_matching_route"
    elif r.error_stage not in ("none", None) and not pd.isna(r.error_stage): cause = f"CRAFT_fail_{r.error_stage}"
    elif not r.feasible: cause = "CRAFT_ran_no_route"
    elif r.poi_filters_respected == False: cause = "CRAFT_route_wrong_filter"
    elif r.distance_limits_met == False: cause = "CRAFT_route_violates_distance"
    else: cause = "CRAFT_route_other_violation"
    pd_ = parser_diag(r.model, r.query_id) if cause.startswith("CRAFT") else ""
    gap = (r.objective_value - o["optimal_distance_m"]) / o["optimal_distance_m"] * 100 if cause.startswith("SOLVED") and feas_o else None
    rows.append(dict(query_id=r.query_id, model=r.model, oracle_status=o["status"], oracle_optimal_m=o.get("optimal_distance_m"),
                     candidates_per_slot=str(o["n_candidates_per_slot"]), craft_solved=bool(r.solved_correctly), craft_objective_m=r.objective_value,
                     gap_pct=gap, cause=cause, parser_diag=pd_))
out = pd.DataFrame(rows)
summ = out.groupby(["model", "cause"]).size().unstack(fill_value=0)
diag = out[out.cause.str.startswith("CRAFT")].groupby(["model", "cause", "parser_diag"]).size().rename("n").reset_index()
with pd.ExcelWriter(ROOT/"results/set3_oracle_gap.xlsx") as w:
    summ.to_excel(w, sheet_name="Summary"); diag.to_excel(w, sheet_name="Failure x Parser", index=False); out.to_excel(w, sheet_name="Per-Query", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(summ.T); print(diag.to_string())
