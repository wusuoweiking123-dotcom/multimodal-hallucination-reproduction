# Research gaps（基于三篇论文及 2026 相关工作）

## 结论先行

最有价值的 gap 不是再设计一种固定 attention 放大规则，而是：

> **如何在统一的因果/反事实框架下，对 token、head/layer、frame/time 三个轴进行输入自适应、预算受限的联合干预，并证明它改善的是视觉证据使用，而非 benchmark shortcut？**

这三篇工作刚好给出了三个互补但彼此割裂的信号：SEASON 的 token-level spatial/temporal JSD、CMAC 的 cross-modal/position distortion、R²-TAR 的 layer/head modality ratio。它们尚未回答这些信号是否指向同一个因果故障、何时互相冲突、以及如何只在必要位置启动干预。

## Gap 1：三轴联合控制缺失（首选课题）

现状：

- SEASON 动态选择 spatial vs. temporal penalty，但 layer 集合固定，且需要三路 forward。
- CMAC 操作 token-to-token cross-modal block，但阈值和位置压缩全局固定。
- R²-TAR 动态筛 head，但类别阈值和 layer boundary 是模型级固定参数，没有 token/frame 级风险。

研究问题：能否构造一个稀疏控制器 `pi(intervention | token, head, layer, frame)`，在固定额外 FLOPs/latency 预算下选择“对比、位置校准、head gain 或不干预”？

可检验假设：联合控制器在相同延迟预算下，优于三个模块简单叠加，并减少对 clean/easy samples 的性能伤害。

## Gap 2：attention heuristic 缺少因果识别

三篇都把 attention pattern 当作故障或功能代理：JSD 高表示某类依赖，cross-modal logit 高可能是 spurious correlation，visual ratio 高/低表示感知/推理功能。但相关性不等于该 head/token 对正确答案有正向因果贡献。

建议：

- 用 activation patching、head ablation、gradient gate 的 signed effect 校准 attention proxy；
- 以 counterfactual image/video pairs 测 `do(head gain)` 对正确 token log-odds 的平均处理效应；
- 区分“高注意但无用”“低注意但关键”和真正 causal heads。

R²-TAR 补充材料只把 gradient gate 用于 case analysis，没有把 signed causal usefulness 纳入在线选择，正好留下可延伸空间。

## Gap 3：生成过程中的时变漂移未被统一建模

SEASON 的诊断随 token 变化，但 negative construction 固定；R²-TAR 的 head 类型可随输入 attention 变化，却用固定 layer/ratio 规则；CMAC 对每一步使用固定 distortion。长 CoT 中错误会传播，且 2026 的相关工作已经分别观察到 associative reasoning/divergent thinking、KV semantic drift 和早期视觉错误向后续步骤传播。

空白在于：把“视觉证据衰减 → 中间推理偏移 → 最终 hallucination”建成可观测状态空间，并允许 controller 在不同生成阶段切换干预类型。

## Gap 4：单图、视频、多图、音视频之间缺少统一验证

三篇实验域彼此分离：CMAC 是单图 object hallucination，SEASON 是 8-frame video temporal hallucination，R²-TAR 是单图长推理。最新多图基准显示 integration-stage object representation 维护本身会产生新型错误；speech-vision 等模态也未覆盖。

建议新 benchmark 采用同一语义模板生成 single-image / shuffled-video / multi-image / audio-visual counterfactual variants，分别标注 perception、binding、ordering、reasoning、knowledge 五类错误，使跨域收益可比较。

## Gap 5：负样本和位置校准的 OOD 风险

- Gaussian corruption、temporal homogenization、value replacement 都可能把隐藏状态推离训练分布。
- CMPC 的连续压缩位置可能影响 OCR、计数、空间顺序和长上下文定位。
- aggressive contrastive alpha 会放大两个分支共有的噪声。

应研究 uncertainty-aware strength：以原始/干预分支的一致性、logit margin、attention entropy 和 calibration error 决定 alpha/beta/gamma/gain，并设置“不干预”安全门。

## Gap 6：评测与调参可能夸大泛化

SEASON 对每个 benchmark 在两组 `(alpha,beta)` 中搜索；CMAC/R²-TAR 也展示任务/模型敏感性。如果直接在 test benchmark 选参数，结果包含 selection bias。POPE/CHAIR/精确选择题也不能充分衡量开放式 factuality、答案完整性和拒答校准。

建议协议：跨 benchmark 留一验证（leave-one-benchmark-out）、固定预算、三个 seed、paired bootstrap CI，并同步报告 hallucination、task utility、ECE/Brier、latency 和“无需干预样本”的退化率。

## 推荐论文方案：CATS（暂定名）

**Causal Adaptive Tri-axial Steering for Multimodal Hallucination**

最小可发表版本：

1. 特征：每 token 的 SEASON JSD、CMAC cross-modal tail statistics、每层 R² visual ratio/entropy。
2. 标签：通过小规模 activation intervention 得到 signed causal benefit，而不是仅用 attention 大小。
3. 策略：轻量 gate 在 `{none, temporal CD, spatial CD, CMPC, head gain}` 中稀疏选择，并加入 latency Lagrangian。
4. 训练：只训练 gate，backbone 冻结；或先做无需训练的 contextual bandit/规则版本。
5. 评测：单图 + 多图 + 视频，包含短答案与长 CoT；与单模块、简单叠加和同延迟预算 baseline 比较。
6. 关键消融：去掉 causal label、去掉 no-op、固定 layer、固定 token、无预算约束、跨 backbone 零样本迁移。

主要风险：controller 训练数据泄漏、需要完整 attention 导致显存高、不同 backbone token layout 不统一。可先以 Qwen2.5-VL 系模型族降低工程变量，再跨 Kimi/LLaVA 验证。

## 相关公开证据

- [SEASON 项目页](https://chriswu018.github.io/season/)：方法与代码状态。
- [CMAC/IMCCD 官方仓库](https://github.com/lijm48/IMCCD)：现有公开实现范围。
- [R²-TAR CVPR 页面](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_Reallocating_Attention_Across_Layers_to_Reduce_Multimodal_Hallucination_CVPR_2026_paper.html) 与其 supplemental：layer band、adaptive selection 限制。
- [MIOH](https://openaccess.thecvf.com/content/CVPR2026/html/Min_Fine-Grained_Multi_Image_Object_Hallucination_Benchmark_CVPR_2026_paper.html)：多图 integration-stage failure。
- [Understanding and Mitigating Hallucinations in MCoT](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_Understanding_and_Mitigating_Hallucinations_in_Multimodal_Chain-of-Thought_Models_CVPR_2026_paper.html)：divergent-thinking 阶段的时变错误。
- [KVSmooth](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_KVSmooth_Mitigating_Hallucination_in_Multi-modal_Large_Language_Models_through_Key-Value_CVPR_2026_paper.html)：长生成中的 KV semantic drift。
- [Hallucination-as-Cue](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Understanding_the_Role_of_Hallucination_in_Reinforcement_Post-Training_of_Multimodal_CVPR_2026_paper.html)：benchmark shortcut 与“正确答案是否真的来自视觉”的识别问题。

