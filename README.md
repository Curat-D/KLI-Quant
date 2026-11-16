# KLI-Quant

**Task-Sensitive W4A4 Quantization for Diffusion Models via Decoupled Activation and Weight Error Control**

KLI-Quant is a post-training quantization (PTQ) framework that runs diffusion transformers with 4-bit
weights and 4-bit activations (W4A4) in the NVFP4 format. As in SVDQuant, each linear layer keeps a small
high-precision low-rank branch next to the 4-bit residual. KLI-Quant calibrates both parts against the error
that reaches the network output.

> **Status:** code release in preparation. This repository will host the calibration pipeline, model
> configs, and the fused W4A4 kernels.

## Highlights

- **Sensitivity-weighted objective.** Layer output errors of the same size can cause very different step
  output errors, meaning the error in the denoiser output of one denoising call. KLI-Quant estimates the
  output sensitivity of each layer from the frozen full-precision model with vector–Jacobian probes, then
  calibrates against the sensitivity-weighted layer output error.
- **Low-rank branch for activation error.** The layer output error splits exactly into two terms. The
  activation term depends only on the low-rank branch, and the weight term depends only on the 4-bit
  residual. The low-rank branch is allocated in closed form to compensate activation error.
- **Joint code–scale calibration.** The 4-bit residual is calibrated by searching E2M1 codes and E4M3
  block scales together. This extends GPTQ with scale search under the Hessian of the quantized input.
- **Fused kernel.** One kernel computes the NVFP4 residual GEMM and the BF16 low-rank correction into a
  shared FP32 accumulator. Over 14 GEMM shapes it is 3.12× faster than BF16 on average, and with the
  rank-64 correction included its latency stays within 1.01–1.15× of a plain NVFP4 GEMM. End to end, it
  gives a 1.62× speedup on MiniMax-H3.

## Method

A quantized linear layer computes

$$\hat Y = X L^\top + X_4 R_4^\top, \qquad L = BA, \quad R = W - L.$$

The rank-$r$ branch $L = BA$ receives the unquantized input $X$. The residual $R$ is stored in NVFP4 as
$R_4$ and multiplied with the NVFP4 activation $X_4$. With $\Delta_X = X - X_4$ and $\Delta_R = R - R_4$,
the layer output error splits exactly into

$$E = XW^\top - \hat Y = \underbrace{\Delta_X R^\top}_{\text{activation}} + \underbrace{X_4 \Delta_R^\top}_{\text{weight}}.$$

The activation term depends only on $L$. The weight term is the residual error measured under the quantized
input. Existing pipelines use the other grouping, $E = X\Delta_R^\top + \Delta_X R_4^\top$, which ties both
errors to the residual.

Calibration runs in two stages:

1. **Bypass allocation.** Minimize the sensitivity-weighted activation term over rank-$r$ branches. This
   is a generalized Eckart–Young problem with a closed-form solution, computed with one truncated SVD per
   layer. The weighting is absorbed into $A$ and $B$, so it adds no inference cost.
2. **Residual calibration.** With $L$ fixed, quantize $R = W - L$ to NVFP4 by a joint search over block
   scales and codes under $H_{X_4}$. Rows are coupled through the output sensitivity.

## Evaluated models

- **Video:** HunyuanVideo-13B, Wan2.1-14B, Wan2.2-A14B, MiniMax-H3, FastH3 (4-step)
- **Image:** Qwen-Image (50-step, and 4-step with Lightning)

Baselines, all in the same NVFP4 W4A4 format: RTN, ViDiT-Q, S²Q-inspired, DeltaQuant, SVDQuant,
H-SVDQuant, GPTAQ.

## Release plan

- [ ] Quantization pipeline (sensitivity estimation, Stage 1, Stage 2)
- [ ] Configs for the evaluated models
- [ ] Fused NVFP4 + low-rank kernels for NVIDIA Blackwell GPUs
- [ ] Paper

## Citation

```bibtex
@article{kliquant2027,
  title  = {KLI-Quant: Task-Sensitive W4A4 Quantization for Diffusion Models via Decoupled Activation and Weight Error Control},
  author = {TBD},
  year   = {2027}
}
```

## License

To be announced.
