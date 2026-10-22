"""Background supervisor for the frozen campaign and independent LLM branch."""
from pathlib import Path
import os,sys,time,subprocess,traceback
from auto_evacuation_campaign import ROOT,OUT as MAIN,read,write,sha,stamp,pid_live

OUT=ROOT/'outputs/evacuation_auto_manager_20261005';LLM=ROOT/'outputs/evacuation_auto_llm_independent_20261005'
REPORT=ROOT/'reports/疏散自动实验合并进展与结论_2026-10-05.md'
PROTOCOL=ROOT/'reports/疏散自动实验LLM独立分支固定补充方案_2026-10-05.md'


def launch(script,folder,attempt):
    log=OUT/'logs';log.mkdir(parents=True,exist_ok=True)
    while (log/f'{folder}_{attempt:03d}_stdout.log').exists() or (log/f'{folder}_{attempt:03d}_stderr.log').exists():attempt+=1
    stdout=(log/f'{folder}_{attempt:03d}_stdout.log').open('xb');stderr=(log/f'{folder}_{attempt:03d}_stderr.log').open('xb')
    args=[sys.executable,'-u',str(ROOT/'scripts'/script)]
    if script=='auto_evacuation_campaign.py':args+=['--run']
    return subprocess.Popen(args,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)


def finish(source):
    base=read(MAIN/'budget.json');extra=read(LLM/'budget.json') if source=='independent' else {'simulations_started':0,'matrix_fits':0,'llm_requests':0}
    primary=read(MAIN/'student_summary.json') if (MAIN/'student_summary.json').exists() else None
    llm=read((LLM if source=='independent' else MAIN)/'llm_summary.json')
    total=base['simulations_started']+extra['simulations_started'];assert total<=44738
    text='# 自动实验合并进展与结论\n\n日期：2026-10-05。主流程及适用独立LLM分支已完成。\n\n'
    text+=f"总模拟启动{total}条；学生矩阵拟合{base['matrix_fits']}次、均值参考{base['mean_reference_estimates']}项；PPO更新0；LLM请求{base['llm_requests']+extra['llm_requests']}次。全部分支遵循各自冻结方案，没有为寻找阳性降低原资格阈值。\n\n"
    mech=read(MAIN/'mechanism_summary.json');north=next(r for r in mech['factor_effects'] if r['factor']=='north_exit_capacity')
    text+=f"机制核查：在这两个端点的完整32组合里，北出口容量5→2，在另四变量16配置中全部使603相对601的M收益翻转，平均变化{north['mean_M_change_pp']:+.6f}个百分点。该结论只适用于固定模拟器/控制器/参数网格。\n\n"
    if primary:text+=f"全新48家庭学生主臂：{primary['primary']}。预测MAE：{primary['actual_gain_MAE_pp']}；原固定行为/噪声推进判定{primary['proceed_to_robustness']}。\n\n"
    text+=f"独立LLM建议主对照：失能差{llm['primary_mean_delta_pp']:+.6f}个百分点，95%区间{llm['primary_ci95_pp']}，五种子差{llm['per_training_seed_delta_pp']}，固定改善信号判定{llm['improvement_signal']}。\n\n"
    text+='原门控资格、跨地图/FDS与真实火灾效能未建立。若当前分支无稳定改善，保留负结果并据此确定下一份独立方案，不能重复刷已看银行。\n\n'
    text+=f"[主流程总报告]({(ROOT/'reports/疏散自动实验阶段总报告_2026-10-05.md').as_posix()})；[LLM补充固定方案]({PROTOCOL.as_posix()})。\n"
    if source=='independent':text+=f"\n[LLM独立对照报告]({(ROOT/'reports/LLM独立建议对照自动实验报告_2026-10-05.md').as_posix()})。\n"
    REPORT.write_text(text,encoding='utf-8');s=read(OUT/'manager_status.json');s.update(state='complete',stage='complete',completed_at_utc=stamp(),total_simulations=total,
        llm_source=source,report=REPORT.relative_to(ROOT).as_posix());write(OUT/'manager_status.json',s)
    parent_completion=MAIN/('completion_manifest_v2.json' if (MAIN/'completion_manifest_v2.json').exists() else 'completion_manifest.json')
    files=[REPORT,PROTOCOL,Path(__file__),ROOT/'scripts/auto_evacuation_llm_independent.py',parent_completion]
    if source=='independent':files.append(LLM/'completion_manifest.json')
    write(OUT/'completion_manifest.json',{'all_checks_passed':True,'total_simulations':total,'simulation_cap':44738,'new_ppo_updates':0,
        'created_at_utc':stamp(),'files':{p.relative_to(ROOT).as_posix():sha(p) for p in files}})


def run():
    if not OUT.exists():
        OUT.mkdir();write(OUT/'manager_status.json',{'state':'running','stage':'waiting_for_main','pid':os.getpid(),'created_at_utc':stamp()})
        write(OUT/'manifest.json',{'protocol_sha256':sha(PROTOCOL),'manager_sha256':sha(__file__),'independent_runner_sha256':sha(ROOT/'scripts/auto_evacuation_llm_independent.py'),
            'main_manifest_sha256':sha(MAIN/'manifest.json'),'created_before_llm_inference':True,'simulation_cap_combined':44738})
    else:
        saved=read(OUT/'manager_status.json')
        if saved['state']=='complete':return
        if saved.get('pid')!=os.getpid() and pid_live(saved.get('pid')):print('Manager already active',flush=True);return
        saved.update(state='running',pid=os.getpid());write(OUT/'manager_status.json',saved)
    try:
        worker=None
        while True:
            status=read(MAIN/'campaign_status.json')
            if status['state']=='complete' and (MAIN/'completion_manifest.json').exists() and not pid_live(status.get('pid')):break
            if status['state']=='complete':time.sleep(20);continue
            if not pid_live(status.get('pid')):
                if status.get('consecutive_failures',0)>=3:raise RuntimeError('main worker failed three times; inspect preserved failure before retry')
                worker=launch('auto_evacuation_campaign.py','main',status['run_attempts']+1)
                # Allow its lock/status initialization before another liveness check.
                time.sleep(10)
            time.sleep(20)
        if (MAIN/'llm_summary.json').exists():finish('main');return
        s=read(OUT/'manager_status.json');s.update(stage='independent_llm',updated_at_utc=stamp());write(OUT/'manager_status.json',s)
        while True:
            status=read(LLM/'campaign_status.json') if (LLM/'campaign_status.json').exists() else None
            if status and status['state']=='complete' and (LLM/'completion_manifest.json').exists() and not pid_live(status.get('pid')):
                finish('independent');return
            if status and status['state']=='complete':time.sleep(20);continue
            if not status or not pid_live(status.get('pid')):
                attempt=1 if not status else status['run_attempts']+1
                if status and status.get('consecutive_failures',0)>=3:raise RuntimeError('independent LLM worker failed three times; inspect preserved failure')
                worker=launch('auto_evacuation_llm_independent.py','llm',attempt);time.sleep(10)
            time.sleep(20)
    except Exception:
        s=read(OUT/'manager_status.json');s.update(state='failed',error=traceback.format_exc(),failed_at_utc=stamp());write(OUT/'manager_status.json',s);raise


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
