"""Re-record every mined seed's policy-death game at tail-90 into crisis/death_games/.

The existing logs death games are tail-60, too short for the D-15..D-85 crisis band
the MCTS-corrections generator needs. batch_record replays each seed to natural
death (batched) and keeps the last 90 plies. Resumable (skips already-recorded).

    PYTHONPATH=. python scripts/record_crisis.py
"""
import os, sys, glob, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from alphatrain.evaluate import load_model
from scripts.batch_record import batch_record

MODEL = 'alphatrain/data/pillar3b_epoch_20.pt'
OUT = 'crisis/death_games'
TAIL = 90


def main():
    seeds = sorted({int(re.search(r'_(\d+)\.json$', f).group(1))
                    for f in glob.glob('logs/mine_*.json')})
    dev = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')
    net, _ = load_model(MODEL, dev, fp16=(dev.type != 'cpu'))
    dtype = next(net.parameters()).dtype
    os.makedirs(OUT, exist_ok=True)
    print(f"re-recording {len(seeds)} mined seeds -> {OUT}/ at tail={TAIL}, "
          f"device={dev} dtype={dtype}", flush=True)
    batch_record(net, dev, dtype, seeds, OUT, MODEL, batch=128, tail=TAIL,
                 skip_existing=True)


if __name__ == '__main__':
    main()
