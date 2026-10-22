"""Offline source, geometry and explanatory-bound audit of completed FDS case.

No input repair, solver invocation, retraining or fitting. Stair probes are
candidate direction lines from the existing mapping, not recovered exit XYZ.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from analyze_safestep_fds_smokecheck import parse_smv
from audit_safestep450_cached_contributions import integral

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/safestep450_source_boundary_audit_20261017'
CASE = ROOT / 'outputs/safestep_continuous450_20261017'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def numeric(body, key, count):
    match = re.search(rf'\b{key}\s*=\s*([+\-\d.eE,\s]+)', body)
    assert match, (key, body)
    values = [float(v.strip()) for v in match.group(1).strip().rstrip(',').split(',')]
    assert len(values) == count, values
    return values


def string(body, key):
    match = re.search(rf"\b{key}\s*=\s*'([^']*)'", body)
    return match.group(1) if match else None


def contains(xyz, xb, eps=1e-8):
    return all(xb[2*a]-eps <= xyz[a] <= xb[2*a+1]+eps for a in range(3))


def main():
    assert not OUT.exists(), 'Keep prior audit immutable'
    protected = {}
    for directory in ('safestep_continuous450_20261017', 'safestep_continuous450_review_20261017'):
        manifest = json.loads((ROOT/'outputs'/directory/'completion_manifest.json').read_text('utf-8'))
        protected.update(manifest['artifact_sha256'])
    for rel, expected in protected.items():
        assert sha(ROOT/rel) == expected, rel
    paths = {
        'input': CASE/'run/hotel_continuous450.fds',
        'smv': CASE/'run/hotel_continuous450.smv',
        'mapping': ROOT/'outputs/safestep_hotel_fds_candidate_20261016/region_mapping_candidate.csv',
        'routes': CASE/'frozen_routes.json',
        'components': ROOT/'outputs/safestep450_cached_contribution_audit_20261017/results.json',
        'native': CASE/'native_gas_and_FER_long.csv',
        'manuscript': ROOT/'outputs/safestep_paper_numerics_20261015/nist_author_manuscript_text.txt',
        'figure11': ROOT/'outputs/safestep_paper_numerics_20261015/source_figures/image11.png',
        'figure12': ROOT/'outputs/safestep_paper_numerics_20261015/source_figures/image12.png',
        'equations': ROOT/'outputs/safestep_hotel_fds_candidate_20261016/manuscript_equation_tokens.json',
        'guide': ROOT/'outputs/safestep_hotel_fds_smokecheck_20261016/FDS_6.11.1_local_guide_text.txt',
        'HCN_reference': ROOT/'outputs/safestep_physics_time_contract_20261016/materials/nistir4989_text.txt',
        'author_main': ROOT/'outputs/safestep_paper_numerics_20261015/author_repository/main.py',
        'template': ROOT/'outputs/safestep_gap_material_search_20261016/materials/example_commercial.fds',
    }
    before = {str(p.relative_to(ROOT)):sha(p) for p in paths.values()}
    text = paths['input'].read_text('utf-8')
    blocks = re.findall(r'&([A-Z][A-Z0-9_]*)\b([^/]*?)/', text, flags=re.S|re.I)
    meshes = [dict(ID=string(b,'ID'), XB=numeric(b,'XB',6), IJK=[int(v) for v in numeric(b,'IJK',3)])
              for k,b in blocks if k.upper() == 'MESH']
    vents = [dict(ID=string(b,'ID'), SURF_ID=string(b,'SURF_ID'), XB=numeric(b,'XB',6))
             for k,b in blocks if k.upper() == 'VENT']
    assert len(meshes) == 6 and len(vents) == 1 and vents[0]['SURF_ID'] == 'OPEN'
    geometry, events = parse_smv(paths['smv'])
    assert len(geometry) == len(meshes)
    mapping = {r['region']:r for r in rows(paths['mapping'])}
    device_bodies = {string(b,'ID'):b for k,b in blocks if k.upper() == 'DEVC'}
    inspected = []
    for region in ('R5','R14','R15','R16','R39','R40','R41','R42','R43'):
        m = mapping[region]
        xyz = [float(m[k]) for k in ('x_m','y_m','z_m')]
        for gas in ('CO','CO2','HCN'):
            identifier = f'SS_R{int(region[1:]):02d}_{gas}'
            assert numeric(device_bodies[identifier],'XYZ',3) == xyz
        observations = []
        for index,mesh in enumerate(meshes,1):
            if not contains(xyz,mesh['XB']):
                continue
            neighbors = []
            for a in range(3):
                step = (mesh['XB'][2*a+1]-mesh['XB'][2*a])/mesh['IJK'][a]
                q = (xyz[a]-mesh['XB'][2*a])/step
                near = round(q)
                choices = (near,near+1) if abs(q-near)<1e-7 else (math.floor(q)+1,)
                neighbors.append([v for v in choices if 1<=v<=mesh['IJK'][a]])
            cells = list(itertools.product(*neighbors))
            matches = []
            for cell in cells:
                for obst in geometry[index-1]:
                    b=obst['grid_bounds']
                    if all(b[2*a]<cell[a]<=b[2*a+1] for a in range(3)):
                        matches.append(dict(cell=cell,local_obst=obst['local_OBST_index']))
            observations.append(dict(mesh_ID=mesh['ID'], incident_cells=cells,
                                     initial_volume_obstacle_matches=matches))
        assert observations and not any(o['initial_volume_obstacle_matches'] for o in observations)
        inspected.append(dict(region=region,requested_XYZ_m=xyz,template_anchor=m['template_anchor_ID'],
                              observations=observations, exact_author_XY_verified=False,
                              height_2m_paper_supported=True,
                              limitation='All incident cells tested for volume obstructions; exact solver-selected cell remains unresolved.'))
    # Exit directions follow the public map; these are diagnostic rays, not door centres.
    probes = [dict(exit='EXIT_A', source_region='R41', face_XYZ_m=[11.2,21.1,1.0],
                   beyond_XYZ_m=[11.4,21.1,1.0], host_mesh='LEFT', direction='+x'),
              dict(exit='EXIT_B', source_region='R14', face_XYZ_m=[28.0,26.3,1.0],
                   beyond_XYZ_m=[28.0,26.1,1.0], host_mesh='TOP', direction='-y')]
    for p in probes:
        p['adjacent_meshes_beyond_candidate_face']=[m['ID'] for m in meshes if contains(p['beyond_XYZ_m'],m['XB'])]
        p['explicit_vents_covering_candidate_face']=[v['ID'] for v in vents if contains(p['face_XYZ_m'],v['XB'])]
        p['status']='NO_ADJACENT_MESH_OR_EXPLICIT_OPEN_AT_CANDIDATE_STAIR_DIRECTION'
        assert not p['adjacent_meshes_beyond_candidate_face'] and not p['explicit_vents_covering_candidate_face']
        p['limit']='Exact stair doorway position, dimensions and operating state are not recovered; this does not establish the author boundary was wrong.'
    components=json.loads(paths['components'].read_text('utf-8'))['route_results']
    bounds=[]
    for r in components:
        bounds.append(dict(route=r['route'], archived_same_primary=r['archived'],
                           unchanged_HCN_term_FED=r['HCN'],
                           lower_bound_if_only_CO_V_over_D_is_changed=r['HCN']/r['archived']))
    native=rows(paths['native'])
    grouped={region:[r for r in native if r['region']==region] for region in ('R5','R43','R42','R41')}
    frozen=json.loads(paths['routes'].read_text('utf-8'))
    alt=next(r for r in frozen['routes'] if r['id']=='hotel_position1_alternative')
    prepeak=0.0
    for s in alt['segments']:
        start=s['start_fire_s'];end=min(s['end_fire_s'],326.0)
        if start>=end:continue
        g=grouped[s['region']]
        prepeak+=integral([float(r['fire_time_s']) for r in g],[float(r['FER_total_per_s']) for r in g],start,end)
    old=next(r for r in components if r['route']==alt['id'])
    prepeak_result=dict(route=alt['id'], FED_before_326s=prepeak,
                        fraction_of_candidate_total=prepeak/old['total'],
                        before_peak_over_archived_entire_route=prepeak/old['archived'],
                        interpretation='Only changing the post-326s plateau cannot explain dose already accumulated before 326s, conditional on same pre-326s input.')
    locators=[]
    for key, patterns in {
        'manuscript':['two exits to the stairwells','All doors and windows were initially closed',
                      'height of 2.0','outside Room 24 (R42)','per minute'],
        'guide':['is a solid wall','HOLE has no effect','external boundary'],
        'HCN_reference':['0.007 by eq.(3) and 0.005','Since flashover appears'],
        'author_main':['pd.read_excel','time_frame=300'],
    }.items():
        lines=paths[key].read_text('utf-8').splitlines()
        for needle in patterns:
            found=[i+1 for i,line in enumerate(lines) if needle in line]
            if needle!='external boundary': assert found,(key,needle)
            locators.append(dict(source=key,search=needle,line_numbers=found))
    result=dict(status='SOURCE_BOUNDARY_AND_EXPLANATORY_BOUNDS_COMPLETE',
                mesh_count=len(meshes),vent_inventory=vents,inspected_regions=inspected,stair_direction_probes=probes,
                physiology_CO_only_bounds=bounds,postpeak_only_bound=prepeak_result,source_locators=locators,
                manual_figure_inspection=dict(figure11='source_figures/image11.png',figure12='source_figures/image12.png',
                    observations=['R42 outside Room24','EXIT_A adjacent to R41, leads to Stair A',
                                  'EXIT_B adjacent to R14, leads to Stair B','Region order R39-R43 is compatible with candidate mapping'],
                    not_recovered=['surveyed region-centre XYZ','stair opening geometry and state']),
                decision=dict(evidence_backed_physics_patch_ready=False,original_upstream_reproduction_verified=False,
                              next_run='No new run created: exact original HCN and stair ventilation remain missing.',
                              avoid=['fit HCN to final FED','equate stair entrance with ambient OPEN','increase RL budget to fix physics']),
                solver_runs=0,training_updates=0,LLM_requests=0,final48_read=False)
    for p in paths.values():assert sha(p)==before[str(p.relative_to(ROOT))]
    for rel,expected in protected.items():assert sha(ROOT/rel)==expected,rel
    OUT.mkdir()
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'source_manifest.json').write_text(json.dumps(dict(input_sha256=before,protected_prior_sha256=protected,
         protected_prior_files_verified=len(protected)),ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'region_device_evidence.csv').open('w',encoding='utf-8-sig',newline='') as f:
        flat=[dict(region=r['region'],XYZ_m=json.dumps(r['requested_XYZ_m']),
                   template_anchor=r['template_anchor'],height_evidence='PAPER_2M',
                   XY_evidence='TEMPLATE_PLUS_PAPER_TOPOLOGY_NOT_EXACT_AUTHOR_XY',
                   initial_incident_cells_clear=True) for r in inspected]
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps(dict(status=result['status'],regions_checked=len(inspected),vent_count=len(vents),
                         bounds=bounds,prepeak=prepeak_result,prior_files_verified=len(protected)),ensure_ascii=True))


if __name__=='__main__':main()
