
# © 2025 Haonan Wang. All rights reserved.
# Contact: hnwang@tongji.edu.cn
# This code is provided for academic and research purposes only.
# Redistribution or modification without explicit permission is prohibited.

"""
ZODPDA_HICZ_Heterogeneity_Compressed_SaveAdv.py
-----------------------------------------------
- Runs 10 heterogeneity experiments using the COMPRESSED solver: ZO_DPDA_HICZ.
- During training: print EVERY 10 iterations -> overall / attack / distortion (pass log_every=10).
- Capture stdout to file and parse to CSV: iter, overall, attack, distortion.
- SAVE ONLY adversarial images (after adding the learned delta), NOT the delta.
  File name format: "adv-{adv_label}_orig-{orig_label}_agent-{i}_idx-{k}.png".
  Total saved images per experiment = agent_num * nFunc_per_agent.

Assumptions:
- Utils.save_img(img) expects img ∈ [-0.5, 0.5] (maps via (img + 0.5) * 255).
  Our adversarial images are in (-1,1), so we save adv_imgs / 2.0.
"""

import os, sys, io, re, argparse, contextlib
import numpy as np
import tensorflow as tf

# --- Make local project modules importable regardless of CWD ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'models'))
sys.path.insert(0, os.path.join(BASE_DIR, 'optimization_methods'))

from setup_mnist import MNIST, MNISTModel
from SysManager import SYS_MANAGER
from graph_generator import random_graph, metropolis_hastings, laplacian_matrix
import Utils as util
import ObjectiveFunc_HICZ as ObjectiveFunc
from ZO_DPDA_HICZ import ZO_DPDA_HICZ

# tf.random.set_seed(2025)
# np.random.seed(2025)

class Tee(contextlib.AbstractContextManager):
    """Tee stdout to both an internal buffer and the original stream."""
    def __init__(self):
        self.buffer = io.StringIO()
        self._orig = sys.stdout
    def __enter__(self):
        sys.stdout = self
        return self
    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._orig
    def write(self, data):
        self._orig.write(data)
        self.buffer.write(data)
    def flush(self):
        self._orig.flush()
        self.buffer.flush()
    def getvalue(self):
        return self.buffer.getvalue()

def make_agent_labels(label_set, agent_num):
    """Cycle label_set to length agent_num."""
    return [ int(label_set[i % len(label_set)]) for i in range(agent_num) ]

def labels_to_str(label_set):
    return "-".join(str(x) for x in label_set)

def apply_delta_tanh_space(x, delta):
    """
    CW-style addition in atanh-space:
      x_adv = tanh(atanh(x) + delta)
    Assume x ∈ (-1,1); clip for numerical stability.
    """
    x = np.clip(x, -0.999, 0.999)
    return np.tanh(np.arctanh(x) + delta)

def predict_labels(model, imgs):
    """Return integer labels for a batch of images (NHWC)."""
    logits = model.model(imgs, training=False).numpy()
    return np.argmax(logits, axis=1).astype(int)

def run_one_experiment(MGR, data, model, label_set, base_save_path):
    agent_num = int(MGR.parSet['agent_num'])
    nFunc_per_agent = int(MGR.parSet['nFunc_per_agent'])
    target_labels = make_agent_labels(label_set, agent_num)

    # Folder per experiment
    exp_name = f"labels_{labels_to_str(sorted(set(label_set)))}"
    save_dir = os.path.join(base_save_path, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # Record config
    MGR.Add_Parameter('target_labels', target_labels)
    MGR.Add_Parameter('save_path', save_dir)

    # Build per-agent objectives and initial deltas; also keep the images/labels used
    objfunc_list, delImgAT_Init_list, agents_imgs = [], [], []
    for i in range(agent_num):
        target_label = int(target_labels[i])
        MGR_i = SYS_MANAGER()
        for k in MGR.parSet:
            MGR_i.Add_Parameter(k, MGR.parSet[k])
        MGR_i.Add_Parameter('target_label', target_label)
        MGR_i.Add_Parameter('agent_id', i)

        # Only the target digit images for training on this agent
        origImgs, origLabels, _ = util.generate_attack_data_set_specific(
            data, model, MGR_i, target_label, nFunc_per_agent
        )
        objfunc = ObjectiveFunc.OBJFUNC_HICZ(MGR_i, model, origImgs, origLabels, agent_id=i)
        objfunc_list.append(objfunc)
        delImgAT_Init_list.append(np.zeros(origImgs[0].shape, dtype=np.float32))
        agents_imgs.append((origImgs, origLabels))

    # Communication graph
    adj = random_graph(agent_num, p=0.4, seed=None, link_type='undirected')
    w = metropolis_hastings(adj)
    lap = laplacian_matrix(w)

    # Meta + log
    MGR.Add_Parameter('eta', MGR.parSet['alpha'] / delImgAT_Init_list[0].size)
    MGR.Log_MetaData()

    # === Train with tee stdout; print every 10 iterations ===
    log_path = os.path.join(save_dir, "train_log.txt")
    csv_path = os.path.join(save_dir, "iter_metrics.csv")
    with Tee() as tee, open(log_path, "w", encoding="utf-8") as logf:
        del_list = ZO_DPDA_HICZ(
            delImgAT_Init_list, MGR, objfunc_list, lap,
            plot_every=10**9, log_every=10, ckpt_every=10**9,
            ckpt_path=os.path.join(save_dir, "hicz_ckpt")
        )
        # persist logs
        text = tee.getvalue()
        logf.write(text)

    # Parse iterations -> CSV (best-effort regex)
    # Accept lines that contain "Avg Overall" OR "Overall", and "Attack", and ("Dist" OR "L2").
    pat = re.compile(r'(?:Avg\s*Overall|Overall)\s*[:=]\s*([0-9.+-eE]+).*?(?:Attack)\s*[:=]\s*([0-9.+-eE]+).*?(?:Dist|L2)\s*[:=]\s*([0-9.+-eE]+)')
    rows = [("iter","overall","attack","distortion")]
    iter_idx = 0
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            rows.append((iter_idx, f"{float(m.group(1)):.6f}", f"{float(m.group(2)):.6f}", f"{float(m.group(3)):.6f}"))
            iter_idx += 10  # because we set log_every=10
    with open(csv_path, "w", encoding="utf-8") as cf:
        for r in rows:
            cf.write(",".join(map(str, r)) + "\n")

    # === SAVE ONLY ADVERSARIAL IMAGES ===
    # For each agent, for each of its used training images:
    #   x_adv = tanh(atanh(x) + delta_i), then save x_adv/2.0 with filename including adv/orig labels.
    total_saved = 0
    for i, (imgs, labels) in enumerate(agents_imgs):
        delta_i = del_list[i]  # [H,W,C] tanh-space delta
        orig_pred = predict_labels(model, imgs)
        adv_imgs = apply_delta_tanh_space(imgs, delta_i)
        adv_pred = predict_labels(model, adv_imgs)

        save_agent_dir = os.path.join(save_dir, f"agent_{i}_adv_examples")
        os.makedirs(save_agent_dir, exist_ok=True)
        for k in range(len(imgs)):
            fn = os.path.join(save_agent_dir, f"adv-{int(adv_pred[k])}_orig-{int(orig_pred[k])}_agent-{i}_idx-{k}.png")
            util.save_img(adv_imgs[k] / 2.0, fn)  # map (-1,1) -> [-0.5,0.5] for Utils.save_img
            total_saved += 1

    # Sanity record
    with open(os.path.join(save_dir, "saved_count.txt"), "w") as f:
        f.write(f"Saved adversarial images: {total_saved} (expected {agent_num * nFunc_per_agent})\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-agent_num', type=int, default=10)
    parser.add_argument('-nFunc_per_agent', type=int, default=3)
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-alpha', type=float, default=1.0)
    parser.add_argument('-M', type=int, default=50)
    parser.add_argument('-nStage', type=int, default=100)
    parser.add_argument('-const', type=float, default=1.5)
    parser.add_argument('-mu', type=float, default=0.01)
    parser.add_argument('-rv_dist', default='UnitSphere')
    parser.add_argument('-q', type=int, default=1, help="ZO directions (not compression)")
    parser.add_argument('-l_bits', type=int, default=4, help="compression bit-width for HICZ")
    parser.add_argument('-save_base', default='HICZ_Heterogeneity_Compressed_SaveAdv')
    args = vars(parser.parse_args())

    # Global manager
    MGR = SYS_MANAGER()
    for par in args:
        if par != 'save_base':
            MGR.Add_Parameter(par, args[par])

    base_save_path = args['save_base']
    os.makedirs(base_save_path, exist_ok=True)

    # Load data/model once
    data, model = MNIST(), MNISTModel(restore="models/mnist", use_log=True)

    # 10 heterogeneity experiments
    experiments = [
        [2, 4, 6, 7, 9],
        [0, 5, 6, 7, 8]
    ]

    print("=== Heterogeneity sweep (Compressed HICZ, save adv images) begins ===")
    for label_set in experiments:
        print(f"\n>>> Running experiment with labels: {label_set}")
        run_one_experiment(MGR, data, model, label_set, base_save_path)

    print("\nAll experiments finished. Results saved under:", base_save_path)

if __name__ == "__main__":
    main()
