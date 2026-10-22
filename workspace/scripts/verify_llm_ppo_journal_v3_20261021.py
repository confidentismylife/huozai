"""Check the final manuscript package against its frozen local sources."""
from pathlib import Path
import csv
import hashlib
import json
import re
from zipfile import ZipFile
from lxml import etree
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/llm_ppo_paper_v3_20261021'
OLD=ROOT/'artifacts/llm_ppo_paper_v2_20261021'
MD=OUT/'固定LLM建议与PPO疏散控制_期刊式改写稿_v3.md'
DOCX=MD.with_suffix('.docx')
PDF=ROOT/'output/pdf'/MD.with_suffix('.pdf').name
source=MD.read_text(encoding='utf-8')
checks={}

def check(name,condition):
    checks[name]=bool(condition)
    if not condition: raise AssertionError(name)

def read_csv(name):
    with (OLD/'data'/name).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

check('seven_numbered_sections',len(re.findall(r'^## [1-7] ',source,re.M))==7)
images=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',source)
check('five_images_resolve',len(images)==5 and all((OUT/p).is_file() for p in images))
check('three_numbered_equations',len(re.findall(r'\\tag\{[123]\}',source))==3)

blocks=[]
current=[]
for line in source.splitlines()+['']:
    if line.startswith('|'):
        row=[x.strip() for x in line.strip('|').split('|')]
        if not all(re.fullmatch(r':?-+:?',x) for x in row): current.append(row)
    elif current:
        blocks.append(current); current=[]
check('five_body_tables',len(blocks)==5)

names={'trained_llm_advice_ppo':'LLM建议PPO','trained_pure_ppo':'独立纯PPO','trained_rule_advice_ppo':'规则建议PPO'}
absolute={r[0]:r for r in blocks[2][1:]}
for r in read_csv('primary_absolute_outcomes.csv'):
    expected=[names[r['method']],r['episodes'],f"{float(r['incapacitation_percent']):.4f}",f"{float(r['mean_cumulative_fed']):.6f}",f"{float(r['episode_end_seconds']):.2f}",f"{float(r['cumulative_congestion']):.4f}"]
    check('absolute_'+r['method'],absolute[names[r['method']]]==expected)

attainment={r[0]:r for r in blocks[3][1:]}
all90={r['method']:r for r in read_csv('endpoint_attainment_summary.csv') if r['scope']=='all' and r['threshold_percent']=='90'}
primary90=[r for r in read_csv('endpoint_attainment_summary.csv') if r['scope']=='calibrated_logit/none' and r['threshold_percent']=='90']
check('three_primary_attainment_rows',len(primary90)==3)
for r in primary90:
    full=all90[r['method']]
    expected=[names[r['method']],full['not_reached']+'/'+full['episodes'],r['reached']+'/'+r['episodes'],f"{float(r['attainment_percent']):.2f}",f"{float(r['conditional_mean_seconds']):.2f}"]
    check('attainment_'+r['method'],attainment[names[r['method']]]==expected)

for i,r in enumerate(read_csv('prediction_without_identity_metrics.csv')):
    row=blocks[4][i+1]
    check('prediction_'+str(i),row[2:]==[f"{float(r['roc_auc']):.4f}",f"［{float(r['roc_auc_ci95_low']):.4f}，{float(r['roc_auc_ci95_high']):.4f}］",f"{float(r['brier_improvement_percent']):.2f}".replace('-','−')])

normalized=source.replace('−','-')
selected={'LLM建议PPO−独立纯PPO','LLM建议PPO−规则建议PPO'}
for r in read_csv('论文主表_修订.csv'):
    if r['比较'] not in selected or r['指标'] not in {'失能率','人均累计FED','回合终止时间'}: continue
    decimals={'失能率':4,'人均累计FED':6,'回合终止时间':2}[r['指标']]
    for key in ['效应值','95%CI下限','95%CI上限']:
        expected=f"{float(r[key]):.{decimals}f}"
        check('paired_'+r['比较']+'_'+r['指标']+'_'+key,expected in normalized)

evidence=json.loads((OUT/'evidence_and_layout_record.json').read_text(encoding='utf-8'))
for path,digest in evidence['source_documents'].items():
    check('unchanged_'+path,hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest)
check('manuscript_hash_current',hashlib.sha256(MD.read_bytes()).hexdigest()==evidence['manuscript_sha256'])
for path in images:
    p=OUT/path
    check('unchanged_figure_'+p.name,hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256((OLD/'figures'/p.name).read_bytes()).hexdigest())

ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
with ZipFile(DOCX) as z:
    xml=etree.fromstring(z.read('word/document.xml'))
    check('docx_six_tables_including_abstract',len(xml.findall('.//w:tbl',ns))==6)
    check('docx_three_native_math_equations',len(xml.findall('.//m:oMath',ns))==3)
    check('docx_five_embedded_figures',len(xml.findall('.//wp:inline',ns))==5)
    check('docx_no_raw_latex',b'\\tag{' not in z.read('word/document.xml'))

pdf=PdfReader(PDF)
check('pdf_ten_pages',len(pdf.pages)==10)
pdftext='\n'.join(p.extract_text() or '' for p in pdf.pages)
check('pdf_no_replacement_glyph','\ufffd' not in pdftext)
check('pdf_contains_final_reference','2108.13264' in pdftext)
check('ten_final_page_images_exist',all((ROOT/'tmp/pdfs/v3_qa'/f'final-{n:02d}.png').is_file() for n in range(1,11)))

record={
    'status':'passed_assistant_checks_requires_author_review',
    'automated_checks':checks,
    'passed_count':len(checks),
    'final_pdf_pages':len(pdf.pages),
    'visual_inspection':{'by':'AI assistant','pages':list(range(1,11)),'render':'Microsoft Word PDF export then Poppler PNG rendering','result':'No observed clipping or overlap; captions and scientific table rows retained together.','builtin_renderer':'Attempted; unavailable because LibreOffice soffice.exe is missing.','human_verified':False},
    'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [MD,DOCX,PDF]},
    'new_training':0,'new_simulation_episodes':0,'new_llm_calls':0,
}
(OUT/'validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':record['status'],'checks':len(checks),'pdf_pages':len(pdf.pages)},ensure_ascii=True))
