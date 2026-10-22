"""Audit downloaded Bristol context-choice and Zenodo field trajectories.

No model fitting, outcome-driven decoding, trajectory intention inference,
author contact, or confirmation performance. Input fields and outcomes export
to separate files with processed CSV source-row provenance.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/dynamic_information_search_20261013'
OUTPUT=SOURCE/'audit'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def number(value):return float(value.strip().replace(',','.'))


def bristol():
    folder=SOURCE/'bristol_context_data/3kn82h2aamsfw2eplor4v2ewk0'
    path=folder/'processed_data/processed_data.CSV'
    original=list(csv.DictReader(path.open(encoding='utf-8-sig')))
    cases=[];inputs=[];outcomes=[]
    for source_row,row in enumerate(original,2):
        left,right,knowledge,choice=(int(row[k]) for k in ['left_exit','right_exit','knowledge','exit_choice'])
        if left<0 or right<0 or left+right==0 or knowledge not in [0,1,2,3] or choice not in [1,2]:
            raise ValueError(f'Invalid condition or outcome at processed row {source_row}')
        start,finish=float(row['start_time']),float(row['finish_time'])
        if not math.isfinite(start+finish) or finish<=start:raise ValueError('Invalid time values')
        ambiguous=left==right and knowledge in [1,2]
        if knowledge==0:information='no_map';nearer='unknown'
        elif knowledge==3:information='map_equal_final_exit_utility';nearer='equal'
        elif ambiguous:information='biased_map_direction_unresolved';nearer='unresolved'
        else:
            majority='left' if left>right else 'right'
            minority='right' if left>right else 'left'
            nearer=majority if knowledge==1 else minority
            information='map_biased_final_exit_utility'
        key=f'context2023_csvrow_{source_row}'
        observation=dict(assigned_left_crowd=left,assigned_right_crowd=right,
            map_information=information,nearer_to_final_exit=nearer,
            crowd_state_design='busy' if max(left,right)>=20 else 'free',
            counts_semantics='Experimental assigned numbers, not timestamped measured queue lengths')
        inputs.append(dict(case_key=key,eligible_context_benchmark=not ambiguous,observation=observation))
        outcome=dict(case_key=key,source_csv_row=source_row,observed_exit=choice,
                     source_file='processed_data/processed_data.CSV')
        outcomes.append(outcome)
        cases.append({**outcome,**observation,'knowledge_code':knowledge,
                      'eligible_context_benchmark':not ambiguous,
                      'exclusion_reason':'balanced_crowd_biased_map_direction_unresolved' if ambiguous else None})
    with (OUTPUT/'bristol_context_inputs.jsonl').open('w',encoding='utf-8') as f:
        for row in inputs:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    write_csv(OUTPUT/'bristol_context_outcomes.csv',outcomes)
    write_csv(OUTPUT/'bristol_context_cases.csv',cases)
    conditions=[]
    for condition in sorted({(r['left_exit'],r['right_exit'],r['knowledge']) for r in original},key=lambda x:tuple(int(y) for y in x)):
        selected=[r for r in original if (r['left_exit'],r['right_exit'],r['knowledge'])==condition]
        conditions.append(dict(left_count=int(condition[0]),right_count=int(condition[1]),knowledge=int(condition[2]),records=len(selected),
                               ambiguity=int(condition[0])==int(condition[1]) and condition[2] in ['1','2']))
    write_csv(OUTPUT/'bristol_condition_counts.csv',conditions)
    raw_folders=[p for p in (folder/'raw_data').iterdir() if p.is_dir()]
    raw_files=[p for p in (folder/'raw_data').rglob('*') if p.is_file() and p.suffix.lower()=='.txt']
    raw_summary=[];all_zero=0;total_motion=0;invalid_motion=0
    for f in raw_files:
        lines=f.read_text(encoding='utf-8-sig').splitlines()
        if len(lines)<6:raise ValueError('Missing raw header')
        header=[number(s) for s in lines[:6]]
        nonzero_rows=0;frames=[]
        for line in lines[6:]:
            vals=[number(x) for x in line.strip().strip('()').split(', ')]
            if len(vals)!=4:invalid_motion+=1;continue
            total_motion+=1
            nonzero_rows+=any(abs(v)>1e-9 for v in vals[:3])
            frames.append(vals[3])
        all_zero+=nonzero_rows==0
        # Keep exact source paths locally for provenance; not sent to predictors.
        raw_summary.append(dict(source_relative_file=str(f.relative_to(folder)),source_sha256=digest(f),
            start=header[0],knowledge=header[1],assigned_left=header[2],assigned_right=header[3],finish=header[4],choice=header[5],
            motion_rows=len(frames),nonzero_coordinate_rows=nonzero_rows,
            all_zero_coordinates=nonzero_rows==0,
            time_non_increasing_pairs=sum(b<=a for a,b in zip(frames,frames[1:])),
            zero_time_rows=sum(t==0 for t in frames)))
    write_csv(OUTPUT/'bristol_raw_file_quality.csv',raw_summary)
    ages=[int(r['age']) for r in original if r['age'] not in ['0','NA','']]
    result=dict(dataset_doi='10.5523/bris.3kn82h2aamsfw2eplor4v2ewk0',paper_doi='10.1016/j.ssci.2022.105919',
        processed_records=len(original),columns=list(original[0]),source_participant_design='between-subjects; one experiment per participant per manuscript',
        processed_participant_ids_present=False,independent_identity_verified_from_processed_csv=False,
        conditions=len(conditions),left_choices=sum(r['exit_choice']=='1' for r in original),right_choices=sum(r['exit_choice']=='2' for r in original),
        ambiguous_records=sum(not r['eligible_context_benchmark'] for r in cases),eligible_records=sum(r['eligible_context_benchmark'] for r in cases),
        clear_no_or_equal_map_records=sum(r['knowledge'] in ['0','3'] for r in original),
        raw_participant_folders=len(raw_folders),raw_data_files=len(raw_files),folders_without_raw_txt=len(raw_folders)-len(raw_files),
        raw_motion_rows=total_motion,raw_unparsed_motion_rows=invalid_motion,raw_all_zero_coordinate_files=all_zero,
        raw_duplicate_or_reset_time_pairs=sum(r['time_non_increasing_pairs'] for r in raw_summary),
        observed_age_range= [min(ages),max(ages)],age_missing_zero=sum(r['age']=='0' for r in original),
        demographic_policy='Not predictor inputs; zero age is not interpreted as an actual age.',
        phase_semantics='Exit choice outcome with experimentally assigned scene context; not dynamic target-switch or staff-guidance labels.',
        raw_motion_semantics_pending='README calls movement rows training phase; paper says training not recorded and choice phase recorded. Fixed-length buffers require validity rules before trajectory use.',
        known_ambiguity='README defines codes 1/2 relative to majority/minority; direction cannot be resolved for equal crowds using those statements alone. Do not use outcomes to infer direction.',
        source_sha256={'archive':digest(SOURCE/'3kn82h2aamsfw2eplor4v2ewk0.zip'),'processed_csv':digest(path),'readme':digest(folder/'README.txt')})
    return result


def zenodo():
    metadata=json.loads((SOURCE/'zenodo_curl.json').read_text(encoding='utf-8'))
    results=[]
    for item in metadata['files']:
        if not item['key'].endswith('.txt'):continue
        path=SOURCE/item['key'];data=path.read_bytes()
        if hashlib.md5(data).hexdigest()!=item['checksum'].split(':')[1] or len(data)!=item['size']:
            raise ValueError('Zenodo source checksum mismatch')
        tracks=defaultdict(list);rows=0
        for line in data.decode().splitlines():
            if not line.strip() or line.startswith('#'):continue
            values=line.split()
            if len(values)!=5:raise ValueError('Expected 5 trajectory columns')
            local_id,frame=int(values[0]),int(values[1]);coords=[float(v) for v in values[2:]]
            if not all(math.isfinite(v) for v in coords):raise ValueError('Invalid coordinates')
            tracks[local_id].append(frame);rows+=1
        results.append(dict(file=item['key'],rows=rows,track_file_keys=len(tracks),fps_from_header=25,
            frame_gap_pairs=sum(b-a>1 for v in tracks.values() for a,b in zip(v,v[1:])),
            duplicate_or_reverse_pairs=sum(b<=a for v in tracks.values() for a,b in zip(v,v[1:])),
            md5_verified=True,sha256=digest(path)))
    return dict(dataset_doi='10.5281/zenodo.3365464',files=results,trajectory_rows=sum(r['rows'] for r in results),
        track_file_keys=sum(r['track_file_keys'] for r in results),cross_file_people_identity_verified=False,
        expected_repository_files=len(metadata['files']),downloaded_trajectory_files=len(results),video_downloaded=False,
        geometry_xml_in_repository_manifest=any(r['key']=='geometry.xml' for r in metadata['files']),
        outcome_columns_present=False,initial_queue_and_guidance_labels_present=False,
        scope='Field trajectory supplement; not a verified source of VR decision records despite dataset title.')


def main():
    OUTPUT.mkdir(exist_ok=True)
    archive=SOURCE/'3kn82h2aamsfw2eplor4v2ewk0.zip'
    download_log=json.loads((SOURCE/'bristol_download_log.json').read_text(encoding='utf-8'))
    registered=next(r for r in download_log if r['file']==archive.name)
    assert archive.stat().st_size==registered['bytes']==2971358
    assert digest(archive)==registered['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in ['README.txt','processed_data/processed_data.CSV']:
            full='3kn82h2aamsfw2eplor4v2ewk0/'+name
            extracted=SOURCE/'bristol_context_data'/full
            assert extracted.read_bytes()==z.read(full)
    result={'bristol_context':bristol(),'zenodo_field':zenodo(),'model_fits':0,'LLM_requests':0,'PPO_updates':0,
            'status':'downloaded_and_audited_inputs_outcomes_separated_no_predictions'}
    save_json(OUTPUT/'intake_summary.json',result)
    inputs=[json.loads(line) for line in (OUTPUT/'bristol_context_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in inputs:
        assert set(row)=={'case_key','eligible_context_benchmark','observation'}
        assert not {'exit_choice','observed_exit','finish_time','gender','age'} & set(row['observation'])
        if row['eligible_context_benchmark']:
            assert row['observation']['nearer_to_final_exit']!='unresolved'
    save_json(OUTPUT/'verification.json',dict(all_547_processed_rows_accounted_for=len(inputs)==547,
        input_outcome_separation_verified=True,ambiguous_codes_not_decoded_from_outcomes=True,
        all_four_zenodo_MD5_and_byte_checks_passed=True,
        bristol_zip_crc_and_extracted_readme_csv_checks_passed=True,no_predictive_evaluation=True))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
