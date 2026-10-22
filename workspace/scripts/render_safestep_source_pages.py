"""Render selected original-paper pages for map, equation and table inspection."""
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('C:/Users/摩天轮大盗/Desktop/火灾行人轨迹疏散论文呢/1-s2.0-S2352710225033698-main.pdf')
DEST = ROOT / 'outputs/safestep_reproduction_20261014/source_audit/pages'
DEST.mkdir(exist_ok=True)
with pymupdf.open(SOURCE) as doc:
    for number in [4, 5, 6, 8, 9, 10, 11, 13, 14]:
        path = DEST / f'pdf_page_{number:02d}.png'
        doc[number-1].get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6)).save(path)
        print(path.name)
