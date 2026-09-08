# 复现指南与核对记录

## 1. SEASON

论文：*SEASON: Mitigating Temporal Hallucination in Video Large Language Models via Self-Diagnostic Contrastive Decoding*。

核心复现：

1. 标准 vision pass 保存每层每帧特征的均值 `d_l`。
2. 第二次 vision pass 在每层执行 `h_l,t = (1-beta) h'_l,t + beta d_l`，得到 temporal negative。
3. 对原视频特征加入 Gaussian noise，得到 spatial negative。论文未给一个对所有 backbone 通用的噪声尺度，因此配置中保持显式 `null`，实际应沿用目标模型的 VCD 实现并报告该值。
4. 三个分支分别得到原始、空间负样本、时间负样本 logits 和 decoder attentions。
5. 在论文默认 `J=[20,21,22,23]` 上，将前一 token 对每帧 patch 的注意力求和并在 frame 维 softmax。
6. 计算两组 JSD，归一化为 `wS,wT`，再使用 Eq. 6 组合 logits。

论文设置：8 帧；`(alpha,beta)` 只搜索 `{(1.0,0.33),(0.5,0.25)}`；默认所有 vision layers 做 homogenization。论文报告在 H100 80GB 上，三分支会带来中等延迟开销。

集成注意：

- Temporal `d_l` 来自**标准 pass**，不能在已经 homogenized 的轨迹上重新求均值，否则会改变 Eq. 2。
- 生成第 `i` 个 token 时，诊断使用前一个 token `y_(i-1)` 的注意力。
- decoder KV cache 必须按原始/空间/时间三个条件分别维护，不能混用。
- 第一个生成 token 没有前一输出 token；建议在 prompt 最后 token 上计算，并在报告中明确。

公开性：截至检查日，项目页仍显示 `Code (Coming Soon)`，所以本目录是 clean-room implementation。

## 2. CMAC

论文：*Cross-Modal Attention Calibration for LVLM Hallucination Mitigation*。

核心复现：

1. 在 attention logits 中截取“后续文本 query → image key”的 cross-modal block。
2. 每个 batch/head 内用该 block 的均值作阈值，得到 `M = I(A_cross > mean(A_cross))`。
3. attention softmax 不变；对 mask 选中的 image-value contribution 使用均值 value 替代，得到 distorted branch。
4. 使用 `(1+alpha) logits_original - alpha logits_distorted`。
5. CMPC 将 image token 的 RoPE 位置跨度压缩为 `1/gamma`，并让后续文本从压缩后的 image span 继续计数。

默认：`alpha=3`、`gamma=2`、`top_p=1`。

实现判定：论文写“dim-wise mean”，公开 IMCCD 代码实际对 image-token 维和 hidden 维同时取 scalar mean（每 batch/head 一个值）。本复现采用公开代码行为，并在配置与 docstring 中明示。若 advanced CMAC 官方代码发布，应把这个选择重新做一次 differential test。

公开性：`third_party/IMCCD` 是作者仓库 commit `57bccdb7746e11b4295cc09b5c7f905052f12153` 的 shallow snapshot。仓库 README 表示 CVPR advanced version 尚待更新。

## 3. R²-TAR

论文：*Reallocating Attention Across Layers to Reduce Multimodal Hallucination*。

核心复现：

1. 从 post-softmax attention 计算 query tokens 分配给 visual tokens 的质量，跨 query 取均值，得到每个 head 的 `S_v`。
2. 浅层且 `S_v >= tau_perc` 标为 perception head；深层且 `S_v <= tau_reas` 标为 reasoning head。
3. 只放大命中的 head，其他 head 保持 1；在 per-head output concat 和 `W_O` 之前缩放。
4. perception/reasoning layer band 可重叠；在重叠层中仍由互斥 ratio 阈值区分两类头。

补充材料参数：

| 模型 | l_reas | tau_reas | g_reas | l_perc | tau_perc | g_perc |
|---|---:|---:|---:|---:|---:|---:|
| Kimi-VL-A3B-Thinking | 5 | 0.01 | 1.40 | 10 | 0.27 | 1.20 |
| Ocean-R1-7B-Instruct | 3 | 0.01 | 1.30 | 7 | 0.22 | 1.16 |
| R1-Onevision | 3 | 0.01 | 1.30 | 7 | 0.30 | 1.20 |

作者说明修改点位于 `modeling_qwen2_5_vl.py` 和 `modeling_kimi_vl.py` 的 `eager_attention_forward`，并缓存视觉 token 范围和超参数。

## 4. 公平实验协议

对每个 backbone × benchmark 至少保存：

- 精确 checkpoint revision、transformers/torch/CUDA 版本；
- prompt 模板和图片/视频预处理版本；
- seed、greedy/sampling、temperature、top-p、max-new-tokens；
- baseline 与 intervention 的逐样本输出；
- deterministic metric 原始结果和 evaluator 版本；
- 峰值显存、首 token 延迟、每 token 延迟、总耗时。

每个配置至少 3 个 seed。超参数应在 validation split 上选，测试集只执行一次，避免论文中“每个 benchmark grid search”造成 test-set selection bias。

## 5. 真实模型接入位置

`torch_ops.py` 提供可粘入模型实现的张量核：

- SEASON：在 generation loop 管理三套 logits/attention/KV cache，然后调用 `season_diagnostic_weights` 和 `season_contrastive_logits`。
- CMAC：在 RoPE 后替换 cross-modal logits，构造 mask；在 `attn_weights @ value_states` 处调用 `cmac_distorted_attention`。
- R²-TAR：在 attention softmax 后调用 `r2tar_gains`；把输出 reshape 为 `[B,Q,H,D]` 后调用 `r2tar_rescale`，再 concat 和 output projection。

FlashAttention/SDPA 通常不返回完整 attention matrix，因此三种方法都应先强制 eager attention 做一致性复现，再单独优化 kernel。

