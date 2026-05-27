"""Push existing export JSONs back into Label Studio as annotations."""

import json
import requests
from pathlib import Path

API_KEY = "c726dc3f9c729652f3316343963f4feade22bcc0"
LS_URL  = "http://localhost:8080"
HEADERS = {"Authorization": f"Token {API_KEY}"}

EXPORT_TO_TASK = {
    "data/export/ResidentialTenancyDatabases_FinalReport.json": 4,
    "data/export/Review_of_the_bail_act_report.json":           5,
    "data/export/medicinal_cannabis-export.json":               6,
    "data/export/VLRC_Recklessness_Report_fnl_Parl.json":       7,
}

for export_path, task_id in EXPORT_TO_TASK.items():
    path = Path(export_path)
    data = None
    for enc in ("utf-8-sig", "utf-16", "utf-8"):
        try:
            data = json.loads(path.read_text(encoding=enc))
            break
        except Exception:
            continue
    if data is None:
        print(f"  {path.name}: could not decode — skipping")
        continue

    if isinstance(data, dict):
        data = [data]

    task = max(data, key=lambda t: sum(len(a.get("result", [])) for a in t.get("annotations", [])))
    valid = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    ann = max(valid, key=lambda a: len(a.get("result", [])), default=None)

    if not ann:
        print(f"  {path.name}: no annotations — skipping")
        continue

    result = ann.get("result", [])
    resp = requests.post(
        f"{LS_URL}/api/tasks/{task_id}/annotations/",
        headers=HEADERS,
        json={"result": result, "ground_truth": False},
    )
    if resp.ok:
        print(f"  Task {task_id} ({path.name}): pushed {len(result)} regions")
    else:
        print(f"  Task {task_id} ERROR {resp.status_code}: {resp.text[:200]}")

print("\nDone.")
