"""Dump per-window coupling/fused margins (attended - competing) for val and test splits of a
CoupleFlow run, at several window lengths, with trial ids and window start (time order).
    python scripts/results/dump_margins.py <run_dir> [<run_dir> ...]
Writes <run_dir>/conf_margins.npz."""
import sys
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupleflow import paths  # noqa: E402
from coupleflow.data import load_dataset, split_fold, eval_windows, build_eval_batch
from coupleflow.evaluate import encode, score, _oth, SR
from coupleflow.trainer import build_model, streams_for, feature_banks

for rd in map(Path, sys.argv[1:]):
    cfg = OmegaConf.load(rd / 'config.yaml')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(rd / 'best.pt', map_location=dev, weights_only=False)
    sn, gn = streams_for(cfg, str(cfg.training.dataset))
    data = load_dataset(str(paths.DATA), str(cfg.training.dataset),
                        drop_frontal=bool(cfg.data.drop_frontal), stream_name=sn, groups_name=gn)
    assert data['subjects'] == ck['subjects']
    data['cand_idx'] = list(cfg.data.get('cand_idx') or [])
    sp = split_fold(data, int(cfg.fold), int(cfg.n_folds), int(cfg.model.margin),
                    val_mode=str(cfg.data.val_mode), val_frac=float(cfg.data.val_frac),
                    tail_min_s=float(cfg.data.get('tail_min_s', 200.0)),
                    mode=str(cfg.data.get('split_mode', 'stimulus')),
                    seed=int(cfg.get('split_seed', 42)))
    bank_dims = [int(data['trials'][0][f'env_att_{bk}'].shape[0]) for bk in feature_banks(cfg)]
    model = build_model(cfg, len(data['subjects']), bank_dims,
                        n_channels=int(data['trials'][0]['eeg'].shape[0])).to(dev)
    model.load_state_dict(ck['model']); model.eval()
    cs = 'competing' if 'competing' in list(cfg.eval.cand_sets) else 'same_talker'
    out = {'cand_set': np.array(cs)}
    for split in ('val', 'test'):
        cells = ((1.0, 1.0), (2.0, 2.0), (5.0, 5.0))
        if cs == 'competing':           # overlap analyses exist for two-talker data only
            cells += ((1.0, 0.5), (1.0, 0.25), (5.0, 1.0))
        for ws, hop in cells:
            win = int(round(ws * SR)); M = model.margin; h = int(round(hop * SR)); L = win + 2 * M
            wins = [{'trial': sg.trial, 'start': st} for sg in sp[split]
                    for st in range(sg.start, sg.end - L + 1, h)]
            b = build_eval_batch(data, wins, win, M, cs, seed=0,  # same candidates for every seed
                                 glimpse=bool(getattr(model, 'glimpse', False)),
                                 banks=list(getattr(model, 'banks', [])),
                                 n_glimpse=int(getattr(model, 'n_glimpse', 0)),
                                 extra_feats=getattr(model, 'extra_feats', []))
            enc = encode(model, b, dev)
            sc = score(model, enc['e'], enc['a'], enc['side_logits'], b['side'])
            k = f'{split}_{ws:g}s' + ('' if hop == ws else f'_h{hop:g}')
            out[f'{k}__trial'] = np.asarray(b['trial']).astype(int)
            out[f'{k}__subject'] = b['subj'].numpy().astype(int)
            out[f'{k}__start'] = np.array([w['start'] for w in b['wins']])
            for r in sc:
                out[f'{k}__{r}'] = (sc[r][:, 0] - _oth(sc[r])).cpu().numpy()
            print(rd.name, k, len(wins), {r: float((out[f'{k}__{r}'] > 0).mean()) for r in sc})
    np.savez_compressed(rd / 'conf_margins.npz', **out)
