# © 2025 Haonan Wang. All rights reserved.
# Contact: hnwang@tongji.edu.cn
# This code is provided for academic and research purposes only.
# Redistribution or modification without explicit permission is prohibited.

#Rand200_HICZ.py

import os, sys, argparse, numpy as np, tensorflow as tf

# —— GPU 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("[GPU] Using GPUs:", gpus)
    except RuntimeError as e:
        print(e)
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# path
sys.path.append('models/')
sys.path.append('optimization_methods/')

from setup_mnist import MNIST, MNISTModel
import Utils as util
import ObjectiveFunc_HICZ as ObjectiveFunc
import Rand_200_HICZ as topk
from SysManager import SYS_MANAGER
from graph_generator import random_graph, metropolis_hastings, laplacian_matrix

tf.random.set_seed(2025)
np.random.seed(2025)
MGR = SYS_MANAGER()

def main():
    data, model = MNIST(), MNISTModel(restore="models/mnist", use_log=True)

    # Target tag: One for each agent
    # agent_num = int(MGR.parSet['agent_num'])
    # target_labels = list(range(agent_num))
    # MGR.Add_Parameter('target_labels', target_labels)
    agent_num = int(MGR.parSet['agent_num'])
    even_labels = [0, 2, 4, 6, 8]
    target_labels = [even_labels[i % len(even_labels)] for i in range(agent_num)]
    MGR.Add_Parameter('target_labels', target_labels)

    objfunc_list, delImgAT_Init_list = [], []
    for i in range(agent_num):
        target_label = target_labels[i]
        MGR_i = SYS_MANAGER()
        for k in MGR.parSet:
            MGR_i.Add_Parameter(k, MGR.parSet[k])
        MGR_i.Add_Parameter('target_label', target_label)
        MGR_i.Add_Parameter('agent_id', i)

        # Attack data of this agent (specified number)
        origImgs, origLabels, _ = util.generate_attack_data_set_specific(
            data, model, MGR_i, target_label, MGR.parSet['nFunc_per_agent']
        )

        
        objfunc = ObjectiveFunc.OBJFUNC_HICZ(MGR_i, model, origImgs, origLabels, agent_id=i)
        objfunc_list.append(objfunc)
        delImgAT_Init_list.append(np.zeros(origImgs[0].shape, dtype=np.float32))

    # log eta
    MGR.Add_Parameter('eta', MGR.parSet['alpha'] / delImgAT_Init_list[0].size)
    MGR.Log_MetaData()

    # communication graph
    adj = random_graph(agent_num, p=0.4, seed=None, link_type='undirected')
    w = metropolis_hastings(adj)
    lap = laplacian_matrix(w)

    # distributed zeroth-order optimization; save
    os.makedirs(MGR.parSet['save_path'], exist_ok=True)
    delImgAT_results = topk.Rand_K_HICZ(
        delImgAT_Init_list, MGR, objfunc_list, lap,
        plot_every=1000, log_every=10, ckpt_every=200,
        ckpt_path=os.path.join(MGR.parSet['save_path'], "hicz_ckpt")
    )

    # save
    for agent_id in range(agent_num):
        objfunc = objfunc_list[agent_id]
        target_label = target_labels[agent_id]
        for idx in range(MGR.parSet['nFunc_per_agent']):
            orig_img = objfunc.origImgs_np[idx]
            orig_prob = model.model(np.expand_dims(orig_img, axis=0), training=False).numpy()
            advImg = np.tanh(np.arctanh(orig_img * 1.9999999) + delImgAT_results[agent_id]) / 2.0
            adv_prob = model.model(np.expand_dims(advImg, axis=0), training=False).numpy()
            suffix = f"Adv{np.argmax(adv_prob)}_Orig{np.argmax(orig_prob)}_id{idx}"
            util.save_img(advImg, f"{MGR.parSet['save_path']}/Adv_{suffix}.png")
        util.save_img(np.tanh(delImgAT_results[agent_id]) / 2.0,
                      f"{MGR.parSet['save_path']}/Delta_Agent{agent_id}_Target{target_label}.png")

    consensus = np.mean(delImgAT_results, axis=0)
    util.save_img(np.tanh(consensus) / 2.0, f"{MGR.parSet['save_path']}/Delta_Consensus.png")
    MGR.logHandler.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-optimizer', default='Rand200')
    parser.add_argument('-agent_num', type=int, default=10)
    parser.add_argument('-nFunc_per_agent', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=3)
    parser.add_argument('-l_bits', type=int, default=4)
    parser.add_argument('-q', type=int, default=1)
    parser.add_argument('-alpha', type=float, default=1.0)
    parser.add_argument('-M', type=int, default=50)
    parser.add_argument('-nStage', type=int, default=25000)  # 25000
    parser.add_argument('-const', type=float, default=1.5)
    parser.add_argument('-mu', type=float, default=0.01)
    parser.add_argument('-rv_dist', default='UnitSphere')
    args = vars(parser.parse_args())

    for par in args:
        MGR.Add_Parameter(par, args[par])
    MGR.Add_Parameter('save_path', 'HICZ_Compressed/' + MGR.parSet['optimizer'] + '/')
    MGR.Add_Parameter('nFunc', MGR.parSet['agent_num'] * MGR.parSet['nFunc_per_agent'])
    os.makedirs(MGR.parSet['save_path'], exist_ok=True)
    main()