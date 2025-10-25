# Zeroth-Order Homogeneous Distributed Nonconvex Optimization with Communication Compression


## Description
This Python code generates universal adversarial attacks on neural networks for the MNIST classification task under the black-box setting. For an image **x**, the universal attack **d** is first applied to **x** in the *arctanh* space. The final adversarial image is then obtained by applying the *tanh* transform. Summarizing, **x**<sub>adv</sub> = *tanh*(*arctanh*(2**x**) + **d**)/2

We trains the algorithm on a variety of heterogeneous datasets to verify its applicability under data heterogeneity.
o evaluate the performance of the proposed algorithm under different levels of data heterogeneity, we create ten levels of data distributions among the 10 agents. The heterogeneity increases gradually from the homogeneous case, where all agents share the same digit data, to the fully heterogeneous case, where each agent has samples of a unique digit. Intermediate settings are obtained by progressively decreasing the class overlap among agents.

Below is a list of parameters that the present code takes:
1. **optimizer**: This parameter specifies the optimizer to use during attack generation. Currently the code supports ZOSGD and ZOSVRG.
2. **q**: The number of random vector to average over when estimating the gradient.
3. **alpha**: The optimizer's step size for updating solutions is alpha/(dimension of **x**)
4. **M**: (For ZOSVRG) The number of batches to apply during each stage.
5. **nStage**: (For ZOSVRG) The number of stages. Note that for ZOSGD, the number of iterations is equal to M × nStage.

## .py
The scripts **kdig0.py–kdig4.py** each select two groups of digit samples. Across all scripts, these groups span $k$ distinct digit classes in total.

The script **build_hetero_table.py** collects all experimental results and summarizes them into a single statistical table.



