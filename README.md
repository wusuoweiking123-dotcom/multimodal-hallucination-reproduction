# 三篇多模态幻觉论文复现

本目录复现三种互补的 inference-time 方法：

1. **SEASON**：视频帧级 temporal homogenization + token 级自诊断对比解码。
2. **CMAC**：attention 内的跨模态 value distortion（IMD）+ RoPE 位置压缩（CMPC）。
3. **R²-TAR**：按 visual-attention ratio 识别感知/推理头，再在输出投影前放大。

## 当前交付边界

这是“**公式级、模型无关、可测试**”复现，而不是宣称已在本机复跑论文表格：当前机器是 Apple M3，无 CUDA；原论文分别使用 H100 80GB、A40、A800 等 GPU 和 7B/16B 模型。此外，SEASON 代码仍未发布，R²-TAR 无公开代码，CMAC 官方仓库截至 2026-08-21 仍说明 CVPR advanced version 待更新。

因此目录明确区分：

- `src/hallucination_repro/`：根据论文公式独立实现的 NumPy 核心，可直接测试。
- `src/hallucination_repro/torch_ops.py`：嵌入真实 Hugging Face eager attention 的 PyTorch 运算。
- `third_party/IMCCD/`：CMAC 作者公开的早期 IMCCD 官方代码快照，仅用于交叉核对，**不冒充 CVPR 版 CMAC**。
- `configs/`：论文/补充材料给出的默认参数和搜索空间。
- `tests/`：公式不变量与张量行为测试。
- `docs/`：复现路线、逐篇核对和 research gap。

## 立即运行

无需 PyTorch，只需 NumPy：

```bash
git clone --recurse-submodules https://github.com/wusuoweiking123-dotcom/multimodal-hallucination-reproduction.git
cd multimodal-hallucination-reproduction
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/smoke_test.py
```

如果首次 clone 时没有拉取 submodule，可运行：

```bash
git submodule update --init --recursive
```

当前结果：14 个测试全部通过。

## 代码和公式对应

| 论文 | 代码 | 公式/操作 |
|---|---|---|
| SEASON | `season.py` | Eq. 2 temporal homogenization；Eq. 4 frame attention；Eq. 5 JSD 权重；Eq. 6 decoding |
| CMAC | `cmac.py` | Eq. 8 cross-modal mask；Eq. 9 value distortion；Eq. 12 decoding；Eq. 13 positions |
| R²-TAR | `r2tar.py` | Eq. 7 visual ratio；Eq. 8 head selection；Eq. 9 gain；Eq. 10 rescaling |

## GPU 上复跑论文表格

先阅读 [`docs/REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md)。完整复跑需要：

- SEASON：LLaVA-OV-7B / Qwen2.5-VL-7B / LLaVA-Video-7B，8 帧；VidHalluc、VideoHallucer、AutoEval-Video、TempCompass、VideoMME、MLVU、ActivityNet-QA。
- CMAC：LLaVA-1.5、InstructBLIP、Qwen-VL；POPE、CHAIR、MME。
- R²-TAR：Kimi-VL-A3B-Thinking、Ocean-R1-7B-Instruct、R1-Onevision；MathVista-mini、MathVision-mini、HallusionBench、MMStar、SEED-Bench。

先跑每篇的 baseline，再在**相同 checkpoint、prompt、seed、decode 策略**下启用 intervention。不能把不同模型版本或不同 evaluator 的结果直接与论文表格比较。

## 最值得继续做的方向

建议从 [`docs/RESEARCH_GAPS.md`](docs/RESEARCH_GAPS.md) 的 Gap 1 开始：构建一个带计算预算的 **token × head/layer × frame/time 联合自适应控制器**。三篇论文分别只处理其中一到两个轴，且依赖固定阈值或固定扰动；联合但稀疏的闭环控制是最自然、也最可验证的下一步。
