# © 2025 Haonan Wang. All rights reserved.
# Contact: hnwang@tongji.edu.cn
# This code is provided for academic and research purposes only.
# Redistribution or modification without explicit permission is prohibited.

# No_Sign_HICZ.py 

import os, json, pickle, numpy as np, matplotlib.pyplot as plt, faulthandler
faulthandler.enable()

def _is_finite_array(x):
    """Check if all elements in the array are finite numbers (not NaN/Inf)"""
    import numpy as np
    return np.all(np.isfinite(x))

def _checkpoint_save(path, t, x, v, y, z, q, rng_state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Arrays stored as .npz; random number states stored separately as .pkl (to avoid shape errors in np.savez)
    np.savez_compressed(path + ".npz", t=t, x=x, v=v, y=y, z=z, q=q)
    with open(path + ".rng.pkl", "wb") as f:
        pickle.dump(rng_state, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path + ".meta.json", "w") as f:
        json.dump({"t": int(t)}, f)

def _checkpoint_save(path, t, x, v, y, z, q, rng_state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # array -> npz
    np.savez_compressed(path + ".npz", t=t, x=x, v=v, y=y, z=z, q=q)
    # Random State -> Separate .pkl File (Avoids irregular shape error in np.savez)
    with open(path + ".rng.pkl", "wb") as f:
        pickle.dump(rng_state, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Optional Metadata
    with open(path + ".meta.json", "w") as f:
        json.dump({"t": int(t)}, f)

def _checkpoint_load(path):
    npz = path + ".npz"
    if not os.path.exists(npz):
        return None
    data = np.load(npz, allow_pickle=True)
    # Backward Compatibility: Old checkpoints can continue without rng.pkl
    rng_state = None
    rng_pkl = path + ".rng.pkl"
    if os.path.exists(rng_pkl):
        with open(rng_pkl, "rb") as f:
            rng_state = pickle.load(f)
    return {
        "t": int(data["t"]),
        "x": data["x"], "v": data["v"], "y": data["y"], "z": data["z"], "q": data["q"],
        "rng_state": rng_state,
    }


def Norm_sign(x):
    """
    Norm-sign
    vec = (||x||_inf / 2) * sign(x)
    """
    x = np.asarray(x)
    if x.size == 0:
        return x
    x_norm_inf = np.linalg.norm(x, ord=np.inf)
    if (not np.isfinite(x_norm_inf)) or x_norm_inf == 0.0:
        return np.zeros_like(x)
    return (x_norm_inf / 2.0) * np.sign(x)


def Unbiased_l_bits_quantizer(x, l):
    d = len(x)
    if d == 0: return x
    x_norm_inf = np.linalg.norm(x, ord=np.inf)
    if x_norm_inf == 0: return np.zeros_like(x)
    ptb = np.random.rand(d)
    scale = x_norm_inf / (2 ** (l - 1))
    normalized = np.abs(x) / x_norm_inf
    quantized = np.floor((2 ** (l - 1)) * normalized + ptb)
    return scale * np.sign(x) * quantized

def No_Sign_HICZ(delImgAT_Init_list, MGR, objfunc_list, Lap,
                 plot_every=1000, log_every=10, ckpt_every=200, ckpt_path=None):
    # —— Original Built-in Hyperparameters (Unchanged) ——
    alpha0, beta0, gamma0, omega0 = 0.028, 2.0, 3.0, 0.1
    agent_num = len(objfunc_list)
    dim = int(np.prod(delImgAT_Init_list[0].shape))
    l_bits = int(MGR.parSet.get('l_bits', 4))
    T = int(MGR.parSet['nStage'])

    x = np.zeros((agent_num, dim), dtype=np.float32)
    for i in range(agent_num):
        x[i] = delImgAT_Init_list[i].reshape(-1).astype(np.float32)
    v = np.zeros_like(x); y = np.zeros_like(x); z = np.zeros_like(x); q = np.zeros_like(x)

    # Checkpoint Resumption
    save_dir = MGR.parSet['save_path']
    ckpt_path = ckpt_path or os.path.join(save_dir, "hicz_ckpt")
    state = _checkpoint_load(ckpt_path)
    start_t = 0
    if state is not None:
        print(f"[Resume] Load checkpoint at iter {state['t']}")
        x, v, y, z, q = state["x"], state["v"], state["y"], state["z"], state["q"]
        if state["rng_state"] is not None:
            np.random.set_state(state["rng_state"])
        start_t = int(state["t"]) + 1


    best_Loss = [1e10] * agent_num
    best_del = [init.copy() for init in delImgAT_Init_list]

    it_hist = []
    loss_overall_hist = [[] for _ in range(agent_num)]
    loss_dist_hist = [[] for _ in range(agent_num)]
    loss_attack_hist = [[] for _ in range(agent_num)]
    best_loss_hist = [[] for _ in range(agent_num)]

    avg_overall_hist, avg_attack_hist, avg_dist_hist, avg_best_hist = [], [], [], []
    best_avg_overall = float('inf'); best_avg_attack = float('inf'); best_avg_dist = float('inf')
    best_avg_overall_iter = best_avg_attack_iter = best_avg_dist_iter = 0

    print(f"HICZ start optimization: {agent_num} agent, objective : {MGR.parSet['target_labels']}")
    print(f"communication compression: {l_bits} bits")
    print("=" * 60)
    MGR.logHandler.write("HICZ Optimization Configuration (with Compression)\n")
    MGR.logHandler.write("=" * 50 + "\n")
    MGR.logHandler.write(f"Number of agents: {agent_num}\n")
    MGR.logHandler.write(f"Target labels: {MGR.parSet['target_labels']}\n")
    MGR.logHandler.write(f"Total iterations: {T}\n")
    MGR.logHandler.write(f"Batch size: {MGR.parSet['batch_size']}\n")
    MGR.logHandler.write(f"Quantization bits: {l_bits}\n")
    MGR.logHandler.write(f"Alpha: {MGR.parSet['alpha']}, Mu: {MGR.parSet['mu']}\n")
    MGR.logHandler.write("=" * 50 + "\n\n")

    for t in range(start_t, T):
        # Numerical Fallback: Immediately handle if non-finite numbers occur
        if (not _is_finite_array(x)) or (not _is_finite_array(v)) or (not _is_finite_array(y)) \
           or (not _is_finite_array(z)) or (not _is_finite_array(q)):
            print(f"[WARN] Non-finite detected at iter {t}, applying clipping/reset.")
            x = np.nan_to_num(x, copy=False); v = np.nan_to_num(v, copy=False)
            y = np.nan_to_num(y, copy=False); z = np.nan_to_num(z, copy=False); q = np.nan_to_num(q, copy=False)
            np.clip(x, -50, 50, out=x)

        # Slight Decay (Original Minimum Exponent Remains Unchanged)
        alpha = alpha0 / ((t + 1) ** (1e-5))
        beta  = beta0  * ((t + 1) ** (1e-5))
        gamma = gamma0 * ((t + 1) ** (1e-5))
        omega = omega0

        # —— Zero-Order Gradient Estimation (GPU parallelism implemented inside the objective function) ——
        G = np.zeros_like(x)
        for i in range(agent_num):
            cur = x[i].reshape(28, 28, 1)
            obj = objfunc_list[i]
            n_avail = obj.nFunc
            bsz = min(MGR.parSet['batch_size'], n_avail)
            idx = np.random.choice(np.arange(n_avail), bsz, replace=False)
            noise = np.random.normal(0, 0.01, cur.shape).astype(np.float32)  # noise
            est = obj.gradient_estimation(cur  , MGR.parSet['mu'], MGR.parSet['q'], idx)
            G[i] = est.reshape(-1).astype(np.float32)

        L_q = Lap.dot(q)

        # Auxiliary variable
        y = y + omega * q
        z = z + omega * L_q

        # Primal/Dual Update
        x = x - alpha * beta * (z + L_q) - alpha * (gamma * v + G)
        v = v + alpha * gamma * (z + L_q)

        # Compression
        for i in range(agent_num):
            e = x[i] - y[i]
            q[i] = Norm_sign(e)

        # Statistics and Records
        cur_overall, cur_attack, cur_dist, cur_best = [], [], [], []
        for i in range(agent_num):
            cur = x[i].reshape(28, 28, 1)
            obj = objfunc_list[i]
            obj.evaluate(cur, np.array([]), False)

            loss_overall_hist[i].append(obj.Loss_Overall)
            loss_dist_hist[i].append(obj.Loss_L2)
            loss_attack_hist[i].append(obj.Loss_Attack)

            cur_overall.append(obj.Loss_Overall)
            cur_attack.append(obj.Loss_Attack)
            cur_dist.append(obj.Loss_L2)
            cur_best.append(best_Loss[i])

            if (obj.Loss_Attack <= 4.0) and (obj.Loss_Overall < best_Loss[i]):
                best_Loss[i] = obj.Loss_Overall
                best_del[i] = cur.copy()
            best_loss_hist[i].append(best_Loss[i])

        avg_o = float(np.mean(cur_overall))
        avg_a = float(np.mean(cur_attack))
        avg_d = float(np.mean(cur_dist))
        avg_b = float(np.mean(cur_best))

        if avg_o < best_avg_overall: best_avg_overall, best_avg_overall_iter = avg_o, t
        if avg_a < best_avg_attack:  best_avg_attack,  best_avg_attack_iter  = avg_a, t
        if avg_d < best_avg_dist:    best_avg_dist,    best_avg_dist_iter    = avg_d, t

        avg_overall_hist.append(avg_o)
        avg_attack_hist.append(avg_a)
        avg_dist_hist.append(avg_d)
        avg_best_hist.append(avg_b)
        it_hist.append(t)

        if t % log_every == 0:
            print(f'Iter {t}/{T} ({t/T*100:.1f}%)  '
                  f'Avg Overall={avg_o:.6f}  Attack={avg_a:.6f}  Dist={avg_d:.6f}')
            comp_ratio = calculate_compression_ratio(q, l_bits)
            print(f'  Compression: {l_bits} bits, Ratio={comp_ratio:.1%}')
            print('-' * 60)

            csv_path = os.path.join(save_dir, "iter_metrics.csv")
            header = "iter,overall,attack,distortion\n"
            line = f"{t},{avg_o:.6f},{avg_a:.6f},{avg_d:.6f}\n"
            if t == 0 and not os.path.exists(csv_path):
                with open(csv_path, "w") as f:
                    f.write(header)
                    f.write(line)
            else:
                with open(csv_path, "a") as f:
                    f.write(line)
                    
        if plot_every and (t % (10*plot_every) == 0) and (t > 0):
            plot_progress_hicz(it_hist, loss_overall_hist, loss_dist_hist, loss_attack_hist,
                               best_loss_hist, avg_overall_hist, avg_attack_hist, avg_dist_hist,
                               avg_best_hist, MGR.parSet['save_path'], t, MGR.parSet['target_labels'],
                               l_bits, best_avg_overall, best_avg_overall_iter)

        if ckpt_every and (t % ckpt_every == 0) and (t > 0):
            _checkpoint_save(ckpt_path, t, x, v, y, z, q, np.random.get_state())

    # Final Plots/Report
    plot_final_results_hicz(it_hist, loss_overall_hist, loss_dist_hist, loss_attack_hist,
                            best_loss_hist, avg_overall_hist, avg_attack_hist, avg_dist_hist,
                            avg_best_hist, MGR.parSet['save_path'], MGR.parSet['target_labels'],
                            l_bits, best_avg_overall, best_avg_overall_iter,
                            best_avg_attack, best_avg_attack_iter,
                            best_avg_dist, best_avg_dist_iter)

    print_final_report_hicz(it_hist, loss_overall_hist, best_loss_hist, avg_overall_hist,
                            MGR.parSet['target_labels'], l_bits, best_avg_overall,
                            best_avg_overall_iter, best_avg_attack, best_avg_attack_iter,
                            best_avg_dist, best_avg_dist_iter)

    # Save the final checkpoint at the end of training
    _checkpoint_save(ckpt_path, T-1, x, v, y, z, q, np.random.get_state())

    return [x[i].reshape(28, 28, 1) for i in range(agent_num)]

def calculate_compression_ratio(q, l_bits):
    original_bits = q.size * 32
    compressed_bits = q.size * l_bits
    return compressed_bits / original_bits

# —— Plots/Report (Consistent with previous versions, with minor adjustments) ——
def plot_progress_hicz(iterations, loss_overall_list, loss_distortion_list, 
                      loss_attack_list, best_loss_list, avg_loss_overall,
                      avg_loss_attack, avg_loss_distortion, avg_best_loss,
                      save_path, current_iter, target_labels, l_bits,
                      best_avg_loss_overall=None, best_avg_loss_overall_iter=None):
    import matplotlib.pyplot as plt
    agent_num = len(loss_overall_list)
    plt.figure(figsize=(20, 5 * (agent_num + 1)))

    plt.subplot(agent_num + 1, 3, 1)
    plt.plot(iterations, avg_loss_overall, label='Avg Overall', linewidth=2)
    plt.plot(iterations, avg_best_loss, label='Avg Best', linewidth=2)
    if best_avg_loss_overall is not None and best_avg_loss_overall_iter is not None:
        if best_avg_loss_overall_iter < len(iterations):
            plt.axvline(x=best_avg_loss_overall_iter, color='green', linestyle='--',
                        alpha=0.7, label=f'Best Avg: {best_avg_loss_overall:.4f}')
            plt.plot(best_avg_loss_overall_iter, best_avg_loss_overall, 'go', markersize=8, markeredgewidth=2)
    plt.title(f'Average Loss History (Compression: {l_bits} bits)')
    plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plt.yscale('log')

    plt.subplot(agent_num + 1, 3, 2)
    plt.plot(iterations, avg_loss_overall, label='Overall', alpha=0.7)
    plt.plot(iterations, avg_loss_distortion, label='Distortion', alpha=0.7)
    plt.plot(iterations, avg_loss_attack, label='Attack', alpha=0.7)
    plt.title('Average Loss Components'); plt.xlabel('Iteration'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True); plt.yscale('log')

    plt.subplot(agent_num + 1, 3, 3)
    if len(iterations) > 1:
        loss_diff = np.diff(avg_loss_overall)
        plt.plot(iterations[1:], loss_diff)
        plt.title('Average Loss Change Rate'); plt.xlabel('Iteration'); plt.ylabel('Δ Loss'); plt.grid(True)

    for i in range(agent_num):
        row = i + 1
        plt.subplot(agent_num + 1, 3, row * 3 + 1)
        plt.plot(iterations, loss_overall_list[i], label='Overall', alpha=0.7)
        plt.plot(iterations, best_loss_list[i], label='Best', linewidth=2)
        plt.title(f'Agent {i} (Target {target_labels[i]})')
        plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plt.yscale('log')

        plt.subplot(agent_num + 1, 3, row * 3 + 2)
        plt.plot(iterations, loss_overall_list[i], label='Overall', alpha=0.7)
        plt.plot(iterations, loss_distortion_list[i], label='Distortion', alpha=0.7)
        plt.plot(iterations, loss_attack_list[i], label='Attack', alpha=0.7)
        plt.title(f'Agent {i} - Loss Components')
        plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plt.yscale('log')

        plt.subplot(agent_num + 1, 3, row * 3 + 3)
        if len(iterations) > 1:
            loss_diff = np.diff(loss_overall_list[i])
            plt.plot(iterations[1:], loss_diff)
            plt.title(f'Agent {i} - Loss Change Rate'); plt.xlabel('Iteration'); plt.ylabel('Δ Loss'); plt.grid(True)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/hicz_compressed_progress_iter_{current_iter}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_final_results_hicz(iterations, loss_overall_list, loss_distortion_list, 
                           loss_attack_list, best_loss_list, avg_loss_overall,
                           avg_loss_attack, avg_loss_distortion, avg_best_loss,
                           save_path, target_labels, l_bits,
                           best_avg_loss_overall, best_avg_loss_overall_iter,
                           best_avg_loss_attack, best_avg_loss_attack_iter,
                           best_avg_loss_distortion, best_avg_loss_distortion_iter):
    import matplotlib.pyplot as plt
    agent_num = len(loss_overall_list)
    plt.figure(figsize=(18, 6 * (agent_num + 1)))

    plt.subplot(agent_num + 1, 2, 1)
    plt.semilogy(iterations, avg_loss_overall, alpha=0.5, label='Current Avg')
    plt.semilogy(iterations, avg_best_loss, linewidth=2, label='Best Avg')
    if best_avg_loss_overall_iter < len(iterations):
        plt.axvline(x=best_avg_loss_overall_iter, color='green', linestyle='--', alpha=0.7,
                    label=f'Best Overall: {best_avg_loss_overall:.4f}')
        plt.plot(best_avg_loss_overall_iter, best_avg_loss_overall, 'go', markersize=10, markeredgewidth=2)
    plt.title(f'Average Loss - Final Results ({l_bits}-bit Compression)')
    plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

    plt.subplot(agent_num + 1, 2, 2); plt.axis('off')
    final_avg_text = f"""
    Average Performance Report (All Agents)
    ======================================
    Compression: {l_bits} bits
    Total Iterations: {len(iterations)}

    Overall Loss:
      Initial: {avg_loss_overall[0]:.6f}
      Final:   {avg_loss_overall[-1]:.6f}
      Best:    {best_avg_loss_overall:.6f} (iter {best_avg_loss_overall_iter})

    Attack Loss (Best Avg): {best_avg_loss_attack:.6f} (iter {best_avg_loss_attack_iter})
    Distortion Loss (Best Avg): {best_avg_loss_distortion:.6f} (iter {best_avg_loss_distortion_iter})
    """
    plt.text(0.1, 0.9, final_avg_text, fontsize=10, fontfamily='monospace',
             verticalalignment='top', transform=plt.gca().transAxes)

    for i in range(agent_num):
        plt.subplot(agent_num + 1, 2, (i + 1) * 2 + 1)
        plt.semilogy(iterations, loss_overall_list[i], alpha=0.5, label='Current')
        plt.semilogy(iterations, best_loss_list[i], linewidth=2, label='Best')
        best_idx = int(np.argmin(loss_overall_list[i]))
        if best_idx < len(iterations):
            plt.axvline(x=iterations[best_idx], color='purple', linestyle='--', alpha=0.5,
                        label=f'Best: {min(loss_overall_list[i]):.4f}')
            plt.plot(iterations[best_idx], min(loss_overall_list[i]), 'mo', markersize=8, markeredgewidth=1)
        plt.title(f'Agent {i} (Target {target_labels[i]})')
        plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

        plt.subplot(agent_num + 1, 2, (i + 1) * 2 + 2)
        plt.axis('off')
        best_agent_loss = float(np.min(loss_overall_list[i]))
        best_agent_iter = int(np.argmin(loss_overall_list[i]))
        improvement = (loss_overall_list[i][0] - best_agent_loss) / max(1e-20, loss_overall_list[i][0]) * 100.0
        final_text = f"""
        Agent {i} Report (Target {target_labels[i]})
        =============================
        Total Iterations: {len(iterations)}

        Overall Loss:
          Initial: {loss_overall_list[i][0]:.6f}
          Final:   {loss_overall_list[i][-1]:.6f}
          Best:    {best_agent_loss:.6f} (iter {best_agent_iter})
          Improvement: {improvement:.1f}%
        """
        plt.text(0.1, 0.9, final_text, fontsize=10, fontfamily='monospace',
                 verticalalignment='top', transform=plt.gca().transAxes)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/hicz_compressed_final_report.png", dpi=300, bbox_inches='tight')
    plt.close()

def print_final_report_hicz(iterations, loss_overall_list, best_loss_list, 
                           avg_loss_overall, target_labels, l_bits,
                           best_avg_loss_overall, best_avg_loss_overall_iter,
                           best_avg_loss_attack, best_avg_loss_attack_iter,
                           best_avg_loss_distortion, best_avg_loss_distortion_iter):
    print("\n" + "="*80)
    print(f" HICZ Optimization completion report ({l_bits}bit communication compression)")
    print("="*80)
    best_avg_loss = min(avg_loss_overall)
    avg_improve = (avg_loss_overall[0] - best_avg_loss) / max(1e-20, avg_loss_overall[0]) * 100.0
    print("Overall average performance:")
    print(f"  Initial average loss: {avg_loss_overall[0]:.6f}")
    print(f"  Final average loss: {avg_loss_overall[-1]:.6f}")
    print(f"  Best average loss: {best_avg_loss_overall:.6f} (Iteration{best_avg_loss_overall_iter})")
    print(f"  Best average attack loss: {best_avg_loss_attack:.6f} (Iteration{best_avg_loss_attack_iter})")
    print(f"  Best average distortion loss: {best_avg_loss_distortion:.6f} (Iteration{best_avg_loss_distortion_iter})")
    print(f"  Average improvement degree: {avg_improve:.1f}%\n")

    print("Performance details of each agent:")
    for i in range(len(loss_overall_list)):
        best_loss = float(np.min(loss_overall_list[i]))
        best_iter = int(np.argmin(loss_overall_list[i]))
        improve = (loss_overall_list[i][0] - best_loss) / max(1e-20, loss_overall_list[i][0]) * 100.0
        print(f"Agent {i} (Target {target_labels[i]}):")
        print(f"  Initial loss: {loss_overall_list[i][0]:.6f}")
        print(f"  Final loss: {loss_overall_list[i][-1]:.6f}")
        print(f"  Best loss: {best_loss:.6f} (Iteration{best_iter})")
        print(f"  Improvement degree: {improve:.1f}%\n")
    print("="*80)