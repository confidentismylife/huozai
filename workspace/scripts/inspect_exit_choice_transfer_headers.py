"""Read-only header inventory; no scores, labels, fitting or model requests."""
import json
from pathlib import Path
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
path = Path("C:/Users/摩天轮大盗/Downloads/Exit choice.xlsx")
wb = openpyxl.load_workbook(path, read_only=True, data_only=False)
summary = []
for ws in wb.worksheets:
    if "EC" not in ws.title:
        continue
    rows = list(ws.iter_rows(values_only=True))
    for i, row in enumerate(rows):
        starts = [j for j, cell in enumerate(row) if isinstance(cell, str) and cell.strip().lower() == "dist"]
        if len(starts) != 3:
            continue
        previous = rows[i-1]
        summary.append({"sheet": ws.title, "header_row": i+1,
                        "previous_row_first_three": [str(x) if x is not None else None for x in previous[:3]],
                        "exit_labels": [previous[j] for j in starts], "starts": starts})
wb.close()
print(json.dumps(summary, ensure_ascii=False, indent=2))
