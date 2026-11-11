"""Hourly progress check for the NJ148 Colab runs. Reads only file names, timestamps and 'step' lines of the logs.

Writes MyDrive/nj148/runs/STATUS.txt every hour and stops when all jobs are done (or after 24 hours).
Run it from a Colab terminal (Drive mounted), or on the Mac with the Drive for desktop path as the first argument:
    nohup python3 /content/drive/MyDrive/nj148/check_status.py > /dev/null 2>&1 &
    nohup python3 check_status.py "$HOME/Library/CloudStorage/GoogleDrive-<account>/My Drive/nj148/runs_gemma3_1b" gemma > /dev/null 2>&1 &
"""
import glob
import os
import re
import sys
import time
from datetime import datetime, timezone

RUNS = sys.argv[1] if len(sys.argv) > 1 else '/content/drive/MyDrive/nj148/runs'   # on the Mac: the Drive for desktop path
MODE = sys.argv[2] if len(sys.argv) > 2 else 'main'   # main, ablations, gemma, e3x or bert
# mode: (experiments, checkpoint folder, run-name suffix in predictions and logs)
MODES = {'main': (('E1', 'E2', 'E3'), 'checkpoints/official', ''), 'ablations': (('E8', 'E9', 'E10'), 'checkpoints/official', ''),
         'gemma': (('E1', 'E2', 'E3', 'E8', 'E9', 'E10'), 'checkpoints/official/gemma3_1b', ''),
         'e3x': (('E1', 'E2', 'E3'), 'checkpoints/official/llama_extraction', ''), 'bert': (('E1', 'E3'), 'checkpoints/official/bert_it', '-BERT'),
         'bert2': (('E1', 'E2', 'E3', 'E8', 'E9', 'E10'), 'checkpoints/official/bert_it_v2', '-BERT')}
EXPERIMENTS, CKPT, SUFFIX = MODES[MODE]
SEEDS = (42, 43, 44) if MODE == 'bert2' else (42,)
JOBS = [(e, k, s) for s in SEEDS for e in EXPERIMENTS for k in range(1, 6)]
INTERVAL_S, MAX_HOURS, STALL_MIN = 3600, 24, 90


def job_state(exp, fold, SEED):
    trained = os.path.exists(f'{RUNS}/{CKPT}/{exp}/fold_{fold}/seed_{SEED}/training_manifest.json')
    inferred = bool(glob.glob(f'{RUNS}/predictions_secure/{exp}{SUFFIX}__*__seed{SEED}/fold_{fold}/prediction_manifest.json'))
    log = f'{RUNS}/logs/{exp}{SUFFIX}_fold{fold}_seed{SEED}.log'
    step, age = '', None
    if os.path.exists(log):
        age = (time.time() - os.path.getmtime(log)) / 60
        with open(log, errors='replace') as f:
            steps = re.findall(r'step (\d+)/(\d+) loss [\d.]+ elapsed_min [\d.]+(?: eta_min ([\d.]+))?', f.read())
        if steps:
            s, total, eta = steps[-1]
            step = f'step {s}/{total}' + (f', eta {eta} min' if eta else '')
    if inferred:
        return 'done', age
    if trained:
        return 'training done, inference running or pending', age
    if step:
        return f'training {step}', age
    return ('started' if age is not None else 'queued'), age


def report():
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    rows, done, recent = [], 0, []
    for exp, fold, seed in JOBS:
        state, age = job_state(exp, fold, seed)
        done += state == 'done'
        if age is not None and state != 'done':
            recent.append(age)
        rows.append(f'{exp} fold {fold} seed {seed}: {state}' + (f' (log updated {age:.0f} min ago)' if age is not None and state != 'done' else ''))
    lines = [f'NJ148 status at {now}: {done}/{len(JOBS)} jobs done']
    if done == len(JOBS):
        lines.append('ALL DONE. Release both Colab servers to stop using compute units.')
    elif not recent or min(recent) > STALL_MIN:
        lines.append(f'WARNING: no unfinished job has written to its log for over {STALL_MIN} min. A session may have dropped:'
                     ' rerun cells 1-4 and 7 there.')
    lines += [''] + rows
    with open(f'{RUNS}/STATUS.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return done == len(JOBS)


if __name__ == '__main__':
    started = time.time()
    while True:
        if report() or time.time() - started > MAX_HOURS * 3600:
            break
        time.sleep(INTERVAL_S)
