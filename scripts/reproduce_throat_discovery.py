"""Run the unchanged discovery protocol into fresh directories, preserving release evidence."""
import argparse
from pathlib import Path
import sys
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_throat_discovery as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    args = parser.parse_args()
    runner.OUT = args.output.resolve()
    runner.DATA = args.data.resolve()
    if runner.OUT == runner.DATA or runner.OUT in runner.DATA.parents or runner.DATA in runner.OUT.parents:
        raise ValueError('Use separate output and data directories')
    if runner.OUT.exists() or runner.DATA.exists():
        raise ValueError('Reproduction requires two fresh directories')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        runner.freeze()
        runner.select()
        runner.evaluate()
    from scripts import build_discovery_report as report
    report.OUT = runner.OUT
    report.DATA = runner.DATA
    report.main()


if __name__ == '__main__':
    main()
