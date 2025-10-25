# Zeroth-Order Homogeneous Distributed Nonconvex Optimization with Communication Compression


## Description
This Python code generates universal adversarial attacks on neural networks for the MNIST classification task under the black-box setting. For an image **x**, the universal attack **d** is first applied to **x** in the *arctanh* space. The final adversarial image is then obtained by applying the *tanh* transform. Summarizing, **x**<sub>adv</sub> = *tanh*(*arctanh*(2**x**) + **d**)/2

We compare the proposed HZCSD algorithm combined with four types of compressors ($k$-bit, Top-$k$, Rand-$k$, and Norm-sign) against a state-of-the-art zeroth-order (ZO) algorithm to demonstrate its convergence rate and communication efficiency.

Below is a list of parameters that the present code takes:
1. **optimizer**: This parameter specifies the optimizer to use during attack generation. Currently the code supports ZOSGD and ZOSVRG.
2. **q**: The number of random vector to average over when estimating the gradient.
3. **alpha**: The optimizer's step size for updating solutions is alpha/(dimension of **x**)
4. **M**: (For ZOSVRG) The number of batches to apply during each stage.
5. **nStage**: (For ZOSVRG) The number of stages. Note that for ZOSGD, the number of iterations is equal to M × nStage.

## .py
**$k$-bit Compressor**: **2bit_HIZC.py**, **4bit_HIZC.py**.
**Top-$k$ Compressor**: **Top50_HICZ.py**, **Top50_HICZ.py**.
**Rand-$k$ Compressor**: **Rand200_HICZ.py**, **Rand400_HICZ.py**.
**Norm-sign Compressor**: **NoSign_HICZ.py**.
**ZO Algorithm**: **# ZO_HICZ.py**.

**plot.py** plots the evolution of the average loss over iterations.
**plot_bit.py** plots the evolution of the average loss over inter-agent bits.




