"""Read-only numerical, source-preservation and document-layout QA."""
from pathlib import Path
from collections import Counter, defaultdict
from statistics import fmean
import csv
import hashlib
import json
import re
import sys
import unicodedata
import zipfile
import xml.etree.ElementTree as ET

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont
from build_full_history_report import split_table_row, markdown_link

ROOT=Path(__file__).resolve().parents[1]
NAME='LLM_RL火灾疏散研究全程整合报告_2026-10-08'
ASSETS=ROOT/'artifacts/integrated_report_assets_20261008'
FULL=ROOT/'outputs/repeat_override_confirmation_full_20261007T144941461333Z'
CF=ROOT/'outputs/repeat_override_confirmation_continuations_20261007T145327415464Z'
ANALYSIS=ROOT/'outputs/repeat_override_confirmation_analysis_20261007T145724470047Z'
WNS='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv_read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def normalize(s): return re.sub(r'\s+','',unicodedata.normalize('NFKC',s))


def list_numbering_audit(markdown, document, numbering):
    intended=[];in_code=False
    for line in markdown.splitlines():
        stripped=line.strip()
        if stripped.startswith(('```','~~~')):in_code=not in_code;continue
        if in_code:continue
        match=re.match(r'^(\d+)[.)]\s+(.+)$',stripped)
        if match:intended.append((int(match[1]),normalize(plain(match[2]))))
    abstracts={n.get(WNS+'abstractNumId'):n for n in numbering.findall(WNS+'abstractNum')}
    instances={n.get(WNS+'numId'):n for n in numbering.findall(WNS+'num')}
    seen=Counter();actual=[]
    for paragraph in document.iter(WNS+'p'):
        props=paragraph.find(WNS+'pPr/'+WNS+'numPr')
        if props is None:continue
        ref=props.find(WNS+'numId')
        if ref is None or ref.get(WNS+'val')=='0':continue
        ident=ref.get(WNS+'val');level=props.find(WNS+'ilvl')
        ilvl=level.get(WNS+'val') if level is not None else '0'
        instance=instances[ident];abstract=abstracts[instance.find(WNS+'abstractNumId').get(WNS+'val')]
        start=None
        for override in instance.findall(WNS+'lvlOverride'):
            if override.get(WNS+'ilvl')==ilvl:
                start=override.find(WNS+'startOverride')
        if start is None:
            start=next(n for n in abstract.findall(WNS+'lvl') if n.get(WNS+'ilvl')==ilvl).find(WNS+'start')
        ordinal=int(start.get(WNS+'val'))+seen[ident,ilvl];seen[ident,ilvl]+=1
        body=''.join(t.text or '' for t in paragraph.iter(WNS+'t'))
        actual.append((ordinal,normalize(body)))
    assert actual==intended,[(i,a,b) for i,(a,b) in enumerate(zip(actual,intended)) if a!=b][:5]
    return dict(items_checked=len(actual),independent_lists=len(seen),source_numbers_preserved=True)


def plain(s):
    out=[];i=0
    while i<len(s):
        if s[i]=='\\' and i+1<len(s) and s[i+1] in '\\`*[]|': out.append(s[i+1]);i+=2;continue
        is_image=s[i:i+2]=='!['
        item=markdown_link(s,i+int(is_image))
        if item: out.append(item[0]);i=item[2];continue
        if s[i]=='`':
            marker=re.match(r'`+',s[i:])[0];end=s.find(marker,i+len(marker))
            if end>=0: out.append(s[i+len(marker):end]);i=end+len(marker);continue
        if s[i]=='*': i+=1;continue
        out.append(s[i]);i+=1
    return ''.join(out)


def labels(inc,fed):
    if inc<=-.001:return 'beneficial'
    if inc>=.001:return 'harmful'
    if fed<=-.001:return 'beneficial'
    if fed>=.001:return 'harmful'
    return 'small_or_neutral'


def numeric_audit():
    rows=csv_read(FULL/'episodes.csv');cfs=csv_read(CF/'terminal_rollouts.csv');a=read(ANALYSIS/'summary.json')
    assert len(rows)==2080 and len(cfs)==2480
    idx={r['case_id']:r for r in rows};assert len(idx)==len(rows)
    assert Counter(r['policy'] for r in rows)==Counter(no_advice=400,original_hint=400,preserve_always=400,first_only=400,public_consensus=400,pure_rule=80)
    paired=[]
    for effect in a['paired_effects']:
        pairs=[]
        for r in rows:
            if effect['candidate']=='pure_rule':
                if r['policy']!=effect['baseline'] or r['source_mode']!=effect['mode']:continue
                parts=r['case_id'].split('|');parts[3]='0';parts[-1]='pure_rule'
                candidate=idx['|'.join(parts)]
                pairs.append({m:float(candidate[m])-float(r[m]) for m in effect['mean_differences']})
                continue
            if r['policy']!=effect['candidate'] or r['source_mode']!=effect['mode']:continue
            cid=r['case_id'].rsplit('|',1)[0]+'|'+effect['baseline']
            if effect['baseline']=='pure_rule':
                parts=cid.split('|');parts[3]='0';cid='|'.join(parts)
            base=idx[cid];pairs.append({m:float(r[m])-float(base[m]) for m in effect['mean_differences']})
        assert len(pairs)==effect['paired_comparisons']
        computed={m:fmean(p[m] for p in pairs) for m in effect['mean_differences']}
        assert all(abs(computed[m]-v)<1e-12 for m,v in effect['mean_differences'].items())
        paired.append(dict(mode=effect['mode'],candidate=effect['candidate'],baseline=effect['baseline'],pairs=len(pairs),mean_differences=computed))
    states=[json.loads(s) for s in (FULL/'selected_states.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(states)==248
    values={(r['state_id'],int(r['forced_action']),int(r['future_seed'])):r for r in cfs if r['future_policy']=='common_ppo'}
    confusion=defaultdict(Counter)
    for state in states:
        if state['policy']!='preserve_always':continue
        seeds=sorted({int(r['future_seed']) for r in cfs if r['state_id']==state['state_id'] and r['future_policy']=='common_ppo'})
        deltas=[]
        for seed in seeds:
            cand=values[state['state_id'],state['candidate_action'],seed];base=values[state['state_id'],state['base_action'],seed]
            deltas.append((float(cand['incapacitation_rate'])-float(base['incapacitation_rate']),float(cand['mean_cumulative_fed'])-float(base['mean_cumulative_fed'])))
        label=labels(fmean(d[0] for d in deltas),fmean(d[1] for d in deltas))
        subset='repeat_offers' if state['offers_before']>=1 else 'first_offers'
        for gate,accept in [('first_only',state['first_only_accept']),('public_consensus',state['consensus_accept'])]:
            confusion[state['source_mode'],gate,subset][('accept_' if accept else 'reject_')+label]+=1
    for r in a['shadow_confusion']:
        if r['label']=='local_class' and r['subset'] in ('first_offers','repeat_offers'):
            assert dict(confusion[r['mode'],r['gate'],r['subset']])==r['counts']
    instant='qwen14b_instant_cache'
    repeat=confusion[instant,'first_only','repeat_offers']
    assert repeat==Counter(reject_harmful=13,reject_beneficial=5,reject_small_or_neutral=6)
    actual_steps=sum(1 for _ in (FULL/'actual_step_traces.jsonl').open(encoding='utf-8'))
    assert actual_steps==102632==a['actual_control_steps']
    frozen=[]
    for p in (FULL,CF):
        manifest=read(p/'completion_manifest.json')
        for path,h in manifest['files'].items():assert sha(ROOT/path)==h
        frozen.append(dict(path=str(p),files_checked=len(manifest['files'])))
    return dict(episodes=len(rows),terminal_counterfactuals=len(cfs),selected_states=len(states),actual_control_steps=actual_steps,
                full_paired_groups_recomputed=len(paired),first_repeat_shadow_groups_recomputed=len(confusion),repeat_labels=dict(repeat),paired_effects=paired,completion_hashes=frozen)


def run():
    md=ROOT/'reports'/(NAME+'.md');docx=ROOT/'artifacts'/(NAME+'.docx');pdf=ROOT/'artifacts'/(NAME+'.pdf')
    text=md.read_text(encoding='utf-8');chapters=re.findall(r'(?m)^## (\d+)\.',text)
    assert chapters==list(map(str,range(1,41)))
    assert '<!--' not in text and 'EVIDENCE_INDEX' not in text
    targets=[]
    for line in text.splitlines():
        i=0
        while i<len(line):
            item=markdown_link(line,i)
            if item: targets.append(item[1].strip('<>'));i=item[2]
            else:i+=1
    local=sorted({re.sub(r':\d+$','',t) for t in targets if re.match(r'^[A-Za-z]:[/\\]',t)})
    missing=[t for t in local if not Path(t).exists()];assert not missing,missing
    before=read(ASSETS/'preservation_before.json')['records']
    changed=[p for p,h in before.items() if sha(ROOT/p)!=h];assert not changed,changed
    sources=read(ASSETS/'assembly_manifest.json')['post_oct2_source_reports']
    assert all(sha(ROOT/'reports'/r['source'])==r['sha256'] for r in sources)
    with zipfile.ZipFile(docx) as archive:
        xml=ET.fromstring(archive.read('word/document.xml'));word=''.join(t.text or '' for t in xml.iter(WNS+'t'))
        numbering=ET.fromstring(archive.read('word/numbering.xml'))
        imgs=len(list(xml.iter(WNS+'drawing')))
        toc=[t.text for t in xml.iter(WNS+'instrText') if ' TOC ' in (t.text or '')]
    assert toc and '目录将在PDF导出前' not in word
    lists=list_numbering_audit(text,xml,numbering)
    expected=[];in_code=False
    for line in text.splitlines():
        if line.startswith('```'):in_code=not in_code;continue
        if not line.strip() or line.startswith('# ') and not in_code:continue
        if in_code:expected.append(line);continue
        if line.startswith('|'):
            row=split_table_row(line)
            if not all(re.fullmatch(r':?-{2,}:?',x) for x in row):expected.extend(plain(x) for x in row)
        else:expected.append(plain(re.sub(r'^(?:#{2,6} |[-+*] |\d+[.)] |> )','',line)))
    nw=normalize(word)
    word_missing=[s for s in expected if normalize(s) not in nw]
    assert not word_missing,word_missing[:5]
    d=fitz.open(pdf)
    body='\n'.join(''.join(b[4] for b in p.get_text('blocks') if b[1]>=49 and b[3]<=p.rect.height-50) for p in d)
    npdf=normalize(body)
    pdf_missing=[s for s in expected if normalize(s) not in npdf]
    empty=[i+1 for i,p in enumerate(d) if not p.get_text().strip() and not p.get_images()]
    outside=[]
    for i,p in enumerate(d):
        for b in p.get_text('blocks'):
            if b[0]<-1 or b[1]<-1 or b[2]>p.rect.width+1 or b[3]>p.rect.height+1: outside.append(dict(page=i+1,bbox=list(b[:4])))
    (ASSETS/'pdf_extracted_text.txt').write_text(body,encoding='utf-8')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    contacts=[]
    for start in range(0,len(d),12):
        sheet=Image.new('RGB',(1280,1200),'#e9edf0');draw=ImageDraw.Draw(sheet)
        for i in range(start,min(start+12,len(d))):
            pix=d[i].get_pixmap(matrix=fitz.Matrix(.6,.6),alpha=False);im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);im.thumbnail((302,360))
            slot=i-start;x=10+(slot%4)*320;y=8+(slot//4)*400
            draw.text((x,y),f'第{i+1}页',font=font,fill='#23364a');sheet.paste(im,(x,y+28))
        target=ASSETS/f'contact_{start//12+1:02d}.png';sheet.save(target);contacts.append(str(target))
    detailed=[]
    markers=['###','即时模式：24次','再次覆盖门控','37.1 为什么','失能配对差','分阶段实测','全新家庭的轻量','同时看拦截与误拒']
    select={0,1,len(d)-1}
    for i,p in enumerate(d):
        if any(normalize(m) in normalize(p.get_text()) for m in markers):select.add(i)
    for i in sorted(select):
        target=ASSETS/f'review_{i+1:03d}.png';d[i].get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False).save(target);detailed.append(str(target))
    numbers=numeric_audit()
    result=dict(source_characters=len(text),chinese_characters=len(re.findall(r'[\u4e00-\u9fff]',text)),chapters=40,local_links_checked=len(local),
        preserved_existing_files=len(before),preserved_files_changed=changed,post_oct2_sources_checked=len(sources),
        docx_tables=len(list(xml.iter(WNS+'tbl'))),docx_images=imgs,word_toc_fields=len(toc),ordered_lists=lists,expected_text_units=len(expected),
        missing_word_units=word_missing,missing_pdf_units=pdf_missing,pdf_pages=len(d),empty_pdf_pages=empty,outside_pdf_text=outside,
        replacement_characters=body.count('\ufffd'),contact_sheets=contacts,detailed_pages=detailed,numeric_audit=numbers,
        artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in (md,docx,pdf)],
        new_simulations=0,new_fits=0,new_ppo_updates=0,new_llm_requests=0,reserved_final_scenarios_read=0,
        scope='Preservation and document QA plus recent raw counts/paired means/confusion recheck; not a rerun of every historical analysis.')
    (ASSETS/'report_qa_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('numeric_audit','artifacts','contact_sheets','detailed_pages','missing_pdf_units','missing_word_units')},ensure_ascii=False),flush=True)
    print(json.dumps(dict(pdf_missing_count=len(pdf_missing),word_missing_count=len(word_missing),first_pdf_missing=pdf_missing[:5]),ensure_ascii=False),flush=True)
    assert not pdf_missing and not empty and not outside and not result['replacement_characters']


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8');run()
