# © 2025 Haonan Wang. All rights reserved.
# Contact: hnwang@tongji.edu.cn
# This code is provided for academic and research purposes only.
# Redistribution or modification without explicit permission is prohibited.

# ObjectiveFunc_HICZ.py 

import numpy as np
import tensorflow as tf

tf.random.set_seed(2025)
np.random.seed(2025)

class OBJFUNC_HICZ:
    def __init__(self, MGR, model, origImgs, origLabels, agent_id=0):
        self.const = float(MGR.parSet['const'])
        self.model = model
        self.agent_id = int(agent_id)
        self.target_label = int(MGR.parSet['target_label'])

        # Resident GPU original images/labels (NumPy versions retained as well)
        self.origImgs_np = origImgs.astype(np.float32)
        self.origLabels_np = origLabels.astype(np.float32)
        self.origImgs = tf.convert_to_tensor(self.origImgs_np, tf.float32)        # [N,28,28,1]
        self.origImgsAT = tf.atanh(self.origImgs * 1.9999999)                     # arctanh space
        self.origLabels = tf.convert_to_tensor(self.origLabels_np, tf.float32)    # one-hot

        self.nFunc = int(self.origImgs.shape[0])
        self.imageSize = float(np.size(self.origImgs_np) / self.nFunc)
        self.query_count = 0
        self.Loss_L2 = 1e10
        self.Loss_Attack = 1e10
        self.Loss_Overall = self.Loss_L2 + self.const * self.Loss_Attack

        rv_dist = MGR.parSet.get('rv_dist', 'UnitSphere')
        self.RV_Gen = self._draw_unit_sphere if rv_dist == 'UnitSphere' else self._draw_unit_ball

    # ===== Random direction (unit ball / unit sphere)   =====
    def _draw_unit_ball(self, shape):
        sample = tf.random.uniform(shape, minval=-1.0, maxval=1.0, dtype=tf.float32)
        v = tf.reshape(sample, (-1,))
        n = tf.norm(v) + 1e-20
        return tf.reshape(v / n, shape)

    def _draw_unit_sphere(self, shape):
        sample = tf.random.normal(shape, dtype=tf.float32)
        v = tf.reshape(sample, (-1,))
        n = tf.norm(v) + 1e-20
        return tf.reshape(v / n, shape)

    # ===== Batch Attack Loss (Tensor Forward; Take max within batch) =====
    @tf.function
    def _batch_attack_loss(self, advImgs_b, labels_b):
        scores = tf.cast(self.model.model(advImgs_b, training=False), tf.float32)
        eps = tf.constant(1e-20, tf.float32)
        score_t  = tf.maximum(eps, tf.reduce_sum(labels_b * scores, axis=1))
        score_nt = tf.maximum(eps, tf.reduce_max((1.0 - labels_b) * scores - (labels_b * 10000.0), axis=1))
        loss_each = tf.maximum(0.0, -tf.math.log(score_nt) + tf.math.log(score_t))    # [B]
        return tf.reduce_max(loss_each)  

    # ===== Evaluation (Interface/Numerics Unchanged: Overall = L2 + constAttack) =====
    def evaluate(self, delImgAT, randBatchIdx, addQueryCount=True):
        delImgAT_tf = tf.convert_to_tensor(delImgAT, tf.float32)
        idx = tf.range(self.nFunc, dtype=tf.int32) if randBatchIdx.size == 0 \
              else tf.convert_to_tensor(randBatchIdx, tf.int32)

        delImgsAT = tf.repeat(tf.expand_dims(delImgAT_tf, 0), repeats=self.nFunc, axis=0)
        advImgs   = tf.tanh(self.origImgsAT + delImgsAT) / 2.0
        advImgs_b = tf.gather(advImgs, idx)
        labels_b  = tf.gather(self.origLabels, idx)

        if addQueryCount:
            self.query_count += int(idx.shape[0])

        loss_attack = self._batch_attack_loss(advImgs_b, labels_b)
        loss_l2 = self.imageSize * tf.reduce_mean(tf.square(advImgs - self.origImgs) / 2.0)

        self.Loss_Attack  = float(loss_attack.numpy())
        self.Loss_L2      = float(loss_l2.numpy())
        self.Loss_Overall = float((loss_l2 + self.const * loss_attack).numpy())
        return self.Loss_Overall

    # ===== One-Point Estimation (q Batch Parallelization; Including L2) =====
    def gradient_estimation(self, delImgAT, mu, q, randBatchIdx=np.array([])):
        delImgAT_tf = tf.convert_to_tensor(delImgAT, tf.float32)
        idx = tf.range(self.nFunc, dtype=tf.int32) if randBatchIdx.size == 0 \
              else tf.convert_to_tensor(randBatchIdx, tf.int32)

        f0 = self.evaluate(delImgAT_tf, randBatchIdx)

        q_int = int(q)
        u_batch = tf.stack([self.RV_Gen(delImgAT_tf.shape) for _ in range(q_int)], axis=0)   # [q,H,W,C]
        delImgsAT_all = tf.repeat(tf.expand_dims(delImgAT_tf, 0), repeats=self.nFunc, axis=0) # [N,H,W,C]

        # q directions of full N-sample perturbed images: [q, N, H, W, C]
        adv_all = tf.stack(
            [tf.tanh(self.origImgsAT + (delImgsAT_all + mu * u_batch[k])) / 2.0 for k in range(q_int)],
            axis=0
        )  # [q,N,H,W,C]

        # L2 per-dir
        diff_all = adv_all - tf.repeat(tf.expand_dims(self.origImgs, 0), repeats=q_int, axis=0)
        loss_l2_per_dir = self.imageSize * tf.reduce_mean(tf.square(diff_all) / 2.0, axis=[1,2,3,4])  # [q]

        # Attack per-dir (Take max within batch)
        adv_big = tf.concat([tf.gather(adv_all[k], idx) for k in range(q_int)], axis=0)  # [q*B,H,W,C]
        lbl_big = tf.concat([tf.gather(self.origLabels, idx) for _ in range(q_int)], axis=0)
        scores_big = tf.cast(self.model.model(adv_big, training=False), tf.float32)
        eps = tf.constant(1e-20, tf.float32)
        score_t  = tf.maximum(eps, tf.reduce_sum(lbl_big * scores_big, axis=1))
        score_nt = tf.maximum(eps, tf.reduce_max((1.0 - lbl_big) * scores_big - (lbl_big * 10000.0), axis=1))
        loss_each = tf.maximum(0.0, -tf.math.log(score_nt) + tf.math.log(score_t))            # [q*B]
        loss_attack_per_dir = tf.reduce_max(tf.reshape(loss_each, (q_int, -1)), axis=1)       # [q]

        # Overall per-dir
        f_per_dir = loss_l2_per_dir + self.const * loss_attack_per_dir                        # [q]

        # Two-Point Estimation (q Batch Parallelization; Including L2)
        grad = tf.reduce_sum(((f_per_dir - f0)[:, None, None, None]) * u_batch, axis=0)       # [H,W,C]
        d = float(tf.size(delImgAT_tf).numpy())
        grad = (d / mu) * (grad / float(q_int))
        return grad.numpy()

    # ===== Aggregated Gradient (Consistent with Original Scaling)    =====
    def gradient_estimation_2_point(self, delImgAT, mu, q, randBatchIdx=np.array([])):
        delImgAT_tf = tf.convert_to_tensor(delImgAT, tf.float32)
        idx = tf.range(self.nFunc, dtype=tf.int32) if randBatchIdx.size == 0 \
              else tf.convert_to_tensor(randBatchIdx, tf.int32)

        q_int = int(q)
        u_batch = tf.stack([self.RV_Gen(delImgAT_tf.shape) for _ in range(q_int)], axis=0)
        delImgsAT_all = tf.repeat(tf.expand_dims(delImgAT_tf, 0), repeats=self.nFunc, axis=0) # [N,H,W,C]

        adv_plus_all = tf.stack(
            [tf.tanh(self.origImgsAT + (delImgsAT_all + mu * u_batch[k])) / 2.0 for k in range(q_int)],
            axis=0
        )  # [q,N,H,W,C]
        adv_minus_all = tf.stack(
            [tf.tanh(self.origImgsAT + (delImgsAT_all - mu * u_batch[k])) / 2.0 for k in range(q_int)],
            axis=0
        )  # [q,N,H,W,C]

        # L2 per-dir
        diff_plus  = adv_plus_all  - tf.repeat(tf.expand_dims(self.origImgs, 0), repeats=q_int, axis=0)
        diff_minus = adv_minus_all - tf.repeat(tf.expand_dims(self.origImgs, 0), repeats=q_int, axis=0)
        l2_plus  = self.imageSize * tf.reduce_mean(tf.square(diff_plus)  / 2.0, axis=[1,2,3,4])  # [q]
        l2_minus = self.imageSize * tf.reduce_mean(tf.square(diff_minus) / 2.0, axis=[1,2,3,4])  # [q]

        # Batch Attack Loss (Tensor Forward; Take max within batch)
        adv_plus_big  = tf.concat([tf.gather(adv_plus_all[k],  idx) for k in range(q_int)], axis=0)
        adv_minus_big = tf.concat([tf.gather(adv_minus_all[k], idx) for k in range(q_int)], axis=0)
        lbl_big = tf.concat([tf.gather(self.origLabels, idx) for _ in range(q_int)], axis=0)

        def _attack_per_dir(scores):
            eps = tf.constant(1e-20, tf.float32)
            score_t  = tf.maximum(eps, tf.reduce_sum(lbl_big * scores, axis=1))
            score_nt = tf.maximum(eps, tf.reduce_max((1.0 - lbl_big) * scores - (lbl_big * 10000.0), axis=1))
            loss_each = tf.maximum(0.0, -tf.math.log(score_nt) + tf.math.log(score_t))  # [q*B]
            return tf.reduce_max(tf.reshape(loss_each, (q_int, -1)), axis=1)            # [q]

        scores_plus  = tf.cast(self.model.model(adv_plus_big,  training=False), tf.float32)
        scores_minus = tf.cast(self.model.model(adv_minus_big, training=False), tf.float32)
        atk_plus  = _attack_per_dir(scores_plus)   # [q]
        atk_minus = _attack_per_dir(scores_minus)  # [q]

        f_plus  = l2_plus  + self.const * atk_plus
        f_minus = l2_minus + self.const * atk_minus

        grad = tf.reduce_sum(((f_plus - f_minus)[:, None, None, None]) * u_batch, axis=0)
        d = float(tf.size(delImgAT_tf).numpy())
        grad = (d / mu / 2.0) * (grad / float(q_int))
        return grad.numpy()

    def print_current_loss(self):
        print(f'Agent {self.agent_id} (Target {self.target_label}) - '
              f'Loss_Overall: {self.Loss_Overall:.6f}, '
              f'Loss_L2: {self.Loss_L2:.6f}, '
              f'Loss_Attack: {self.Loss_Attack:.6f}')