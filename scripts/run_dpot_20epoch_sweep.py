#!/usr/bin/env python
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path


TRAIN_PATHS = [
    'ns2d_fno_1e-5',
    'ns2d_fno_1e-3',
    'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2',
    'swe_pdb',
    'dr_pdb',
    'cfdbench',
]
NTRAIN_LIST = [1000, 1000, 9000, 900, 900, 9000]


def add_list_arg(cmd, name, values):
    cmd.append(f'--{name}')
    cmd.extend(str(v) for v in values)


def run_and_log(cmd, log_file):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open('w') as f:
        f.write('$ ' + ' '.join(cmd) + '\n\n')
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def find_log_dir(prefix, start_time):
    candidates = [Path(p) for p in glob.glob(f'logs/{prefix}*') if Path(p).is_dir()]
    candidates = [p for p in candidates if p.stat().st_mtime >= start_time - 2]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_text(path):
    if path and path.exists():
        return path.read_text(errors='replace')
    return ''


def parse_train_log(text):
    params = None
    match = re.search(r'Total Trainable Params:\s*(\d+)', text)
    if match:
        params = int(match.group(1))

    epoch_times = [float(x) for x in re.findall(r'epoch\s+\d+,\s+time\s+([0-9.]+)', text)]
    last_epoch = None
    epoch_matches = re.findall(
        r'epoch\s+(\d+),.*?train l2 step\s+([0-9.eE+-]+)\s+train l2 full\s+([0-9.eE+-]+),'
        r'\s+test l2 step\s+([^,]+(?:,\s*[^,]+)*?)\s+test l2 full\s+([^,]+(?:,\s*[^,]+)*?),\s+time train avg\s+([0-9.eE+-]+).*?test\s+([0-9.eE+-]+)',
        text,
    )
    if epoch_matches:
        last_epoch = epoch_matches[-1]

    return {
        'params': params,
        'train_time_total_s': sum(epoch_times) if epoch_times else None,
        'epochs_seen': len(epoch_times),
        'last_train_l2_step': float(last_epoch[1]) if last_epoch else None,
        'last_train_l2_full': float(last_epoch[2]) if last_epoch else None,
        'last_epoch_test_time_s': float(last_epoch[6]) if last_epoch else None,
    }


def parse_eval_log(text):
    metrics = {}
    for name, value in re.findall(r'^([A-Za-z0-9_\-]+):\s+([0-9.eE+-]+)', text, flags=re.MULTILINE):
        metrics[name] = float(value)

    rollout = re.search(r'Total rollout time:\s*([0-9.eE+-]+)s', text)
    samples = re.search(r'Total samples:\s*(\d+)', text)
    avg = re.search(r'Avg rollout time per sample:\s*([0-9.eE+-]+)s', text)
    return {
        'eval_total_rollout_s': float(rollout.group(1)) if rollout else None,
        'eval_total_samples': int(samples.group(1)) if samples else None,
        'eval_avg_rollout_s_per_sample': float(avg.group(1)) if avg else None,
        'dataset_metrics': metrics,
    }


def write_results(rows, csv_path, md_path):
    dataset_names = TRAIN_PATHS
    fieldnames = [
        'experiment',
        'mode',
        'freq_loss_weight',
        'spatial_loss_weight',
        'params',
        'epochs_seen',
        'train_time_total_s',
        'eval_total_rollout_s',
        'eval_avg_rollout_s_per_sample',
        'train_log',
        'eval_log',
    ] + dataset_names

    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {k: row.get(k) for k in fieldnames}
            for name in dataset_names:
                flat[name] = row.get('dataset_metrics', {}).get(name)
            writer.writerow(flat)

    with md_path.open('w') as f:
        f.write('# DPOT 20 Epoch Sweep Results\n\n')
        f.write('| experiment | mode | freq weight | params | train time (s) | eval rollout (s) | eval/sample (s) |')
        for name in dataset_names:
            f.write(f' {name} |')
        f.write('\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|' + '---:|' * len(dataset_names) + '\n')
        for row in rows:
            f.write(f"| {row['experiment']} | {row['mode']} | {row['freq_loss_weight']} | ")
            f.write(f"{row.get('params') or ''} | {fmt(row.get('train_time_total_s'))} | ")
            f.write(f"{fmt(row.get('eval_total_rollout_s'))} | {fmt(row.get('eval_avg_rollout_s_per_sample'))} |")
            for name in dataset_names:
                f.write(f" {fmt(row.get('dataset_metrics', {}).get(name))} |")
            f.write('\n')


def fmt(value):
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def build_common_args(args, t_bundle, include_eval_args=False):
    cmd = [
        '--model', 'DPOT',
        '--dataset', 'ns2d',
        '--res', '128',
        '--modes', '32',
        '--width', '512',
        '--out_layer_dim', '32',
        '--lr', '0.001',
        '--lr_method', 'cycle',
        '--epochs', str(args.epochs),
        '--warmup_epochs', str(args.warmup_epochs),
        '--noise_scale', '0.0005',
        '--T_ar', '10',
        '--T_bundle', str(t_bundle),
        '--normalize', '0',
        '--patch_size', '8',
        '--mlp_ratio', '1',
        '--n_blocks', '4',
        '--n_layers', '4',
        '--beta1', '0.9',
        '--beta2', '0.9',
        '--batch_size', str(args.batch_size),
        '--gpu', args.gpu,
    ]
    if include_eval_args:
        cmd.extend(['--n_class', str(len(TRAIN_PATHS))])
    add_list_arg(cmd, 'train_paths', TRAIN_PATHS)
    add_list_arg(cmd, 'test_paths', TRAIN_PATHS)
    add_list_arg(cmd, 'ntrain_list', NTRAIN_LIST)
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--warmup_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--output_csv', default='dpot_20epoch_sweep_results.csv')
    parser.add_argument('--output_md', default='dpot_20epoch_sweep_results.md')
    parser.add_argument('--only', nargs='*', default=None,
                        help='Optional experiment names to run, e.g. ar semi_freq_0.5')
    args = parser.parse_args()

    experiments = [('ar', 'AR', None)]
    experiments.extend((f'semi_freq_{w:.1f}', 'semi_freq', w) for w in [i / 10 for i in range(1, 11)])
    if args.only:
        wanted = set(args.only)
        experiments = [exp for exp in experiments if exp[0] in wanted]

    csv_path = Path(args.output_csv)
    md_path = Path(args.output_md)
    rows = []

    for exp_name, mode, freq_weight in experiments:
        t_bundle = 1 if mode == 'AR' else 10
        train_script = 'train_temporal.py' if mode == 'AR' else 'train_temporal_non.py'
        eval_script = 'evaluate.py' if mode == 'AR' else 'evaluate_non.py'
        log_prefix = f'sweep20_{exp_name}'

        train_cmd = [sys.executable, train_script]
        train_cmd.extend(build_common_args(args, t_bundle))
        train_cmd.extend(['--use_writer', '--log_path', log_prefix, '--comment', ''])
        if freq_weight is not None:
            train_cmd.extend(['--freq_loss_weight', str(freq_weight)])

        start_time = time.time()
        outer_log = Path('logs') / f'{log_prefix}_driver_train.log'
        train_rc = run_and_log(train_cmd, outer_log)
        log_dir = find_log_dir(log_prefix, start_time)
        train_log = log_dir / 'logs.txt' if log_dir else outer_log
        model_path = log_dir / 'model.pth' if log_dir else None

        row = {
            'experiment': exp_name,
            'mode': mode,
            'freq_loss_weight': freq_weight if freq_weight is not None else 0.0,
            'spatial_loss_weight': 1.0 - freq_weight if freq_weight is not None else 1.0,
            'train_returncode': train_rc,
            'train_log': str(train_log),
        }
        row.update(parse_train_log(read_text(train_log)))

        eval_log = Path('logs') / f'{log_prefix}_evaluate.log'
        if train_rc == 0 and model_path and model_path.exists():
            eval_cmd = [sys.executable, eval_script]
            eval_cmd.extend(build_common_args(args, t_bundle, include_eval_args=True))
            eval_cmd.extend(['--resume_path', str(model_path)])
            eval_rc = run_and_log(eval_cmd, eval_log)
            row['eval_returncode'] = eval_rc
            row['eval_log'] = str(eval_log)
            row.update(parse_eval_log(read_text(eval_log)))
        else:
            row['eval_returncode'] = None
            row['eval_log'] = str(eval_log)
            row['dataset_metrics'] = {}

        rows.append(row)
        write_results(rows, csv_path, md_path)


if __name__ == '__main__':
    main()
