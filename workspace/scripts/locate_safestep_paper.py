"""Locate the requested local paper by PDF content, preserving source files."""
import hashlib
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PAPERS = Path('C:/Users/摩天轮大盗/Desktop/火灾行人轨迹疏散论文呢')
OUT = ROOT / 'outputs/safestep_reproduction_20261014/source_audit'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = []
    for path in sorted(PAPERS.glob('*.pdf')):
        with pymupdf.open(path) as doc:
            first = '\n'.join(doc[i].get_text() for i in range(min(2, len(doc))))
            found = []
            for i in range(len(doc)):
                text = doc[i].get_text()
                if 'safestep' in text.lower().replace(' ', '').replace('-', ''):
                    found.append(i + 1)
            row = {'path': str(path), 'pages': len(doc), 'metadata_title': doc.metadata.get('title'),
                'first_page_excerpt': first[:1600], 'safestep_pdf_pages': found}
            if found:
                extracted = '\n\n'.join(f'===== PDF PAGE {i+1} =====\n' + doc[i].get_text() for i in range(len(doc)))
                destination = OUT / (path.stem + '.txt')
                destination.write_text(extracted, encoding='utf-8')
                row.update(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), extracted_text=str(destination))
            inventory.append(row)
            print(json.dumps({'file': path.name, 'pages': len(doc), 'safestep_pages': found,
                'title': doc.metadata.get('title')}, ensure_ascii=True), flush=True)
    (OUT / 'local_pdf_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
