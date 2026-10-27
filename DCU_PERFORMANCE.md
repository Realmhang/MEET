# DCU (Hygon BW1000) 性能优化记录

本文档记录 MEET 在海光 BW1000 DCU 上的性能优化结果、有效优化项、以及**已验证无效的方向**（避免重复投入）。所有数字均为同机实测（150 步，2×BW1000，数据集 train_100K，随机 VAE 权重）。

## 1. 结果总览

| 档位 | samples/s | vs 客户报告基线 152.69 | vs PRO5000 + xFormers 491.79 |
|---|---|---|---|
| **b64** | **~530-541**（近期测点 517.8 / 524.8 / 531.8 / 535.0 / 537.6 / 541.4） | **约 +250%** | **约 108%** |
| **b128**（需客户同步调 LR） | **650.2** | 约 +326% | 约 132% |
| b16 | 184.5 - 193.4 | — | — |

开箱基线对齐：未优化路径（math SDPA 垫片）实测 **160.6**，与客户报告的 152.69 同带，确认比较口径一致。

复现命令：

```bash
# b64（默认已含全部有效优化）
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export MEET_FUSED_OPT=1
export MEET_VAE_FUSED_PREP=1      # 可选，-1.5%（见下）
GPU=0,1 PORT=9995 bash scripts/train.sh configs/LatentGenHX/train_100k_base.yaml \
  --dataloader.train.batch_size=64 \
  --dataset.train.mmap_dir=./dataset/train_100K \
  --model.autoencoder_ckpt=./ckpt/vae_random.ckpt \
  --trainer.config.logger=tensorboard --trainer.config.max_step=150
```

## 2. 有效优化项（已进 `dcu-dev` 分支）

| # | 优化 | 实测收益 | 开关 |
|---|---|---|---|
| 1 | **自研 Triton varlen 注意力**（替换 xFormers；fwd BM=64/BN=16 + split-KV；bwd 5-kernel tiled 拆分） | fwd **-35%**、bwd **-26.5%**（PMC 两轮引导） | 默认启用 |
| 2 | **lean wrapper**（注意力前后 launch 消减，-130 launch/步） | b16 **+4.2%** / b64 **+6-7%** | 默认启用 |
| 3 | **fused optimizer + gbv**（`MEET_FUSED_OPT=1`） | b16 **+2.9%** | env 开启 |
| 4 | **VAE 预处理融合**（1 个 Triton kernel 取代 ~25 个 aten 算子；no_grad 冻结区，逐位等价） | b64 **+1.5%**（同日双臂 511.4 → 519.0） | `MEET_VAE_FUSED_PREP=1` |
| 5 | **inductor GEMM autotune**（`max-autotune-no-cudagraphs`） | b64 **+2.2%**（GEMM 池 49.7 → 39.6ms，**-20%**） | 默认启用（编译期） |
| 6 | **健壮性修复**：tiled5 ragged 索引 int32 → int64；Σs² 超阈值回退 SDPA | 无性能代价；消除长序列/大复合物下的越界写与设备 fault | 默认启用（`MEET_N2_CAP` 可配） |

### 关于第 5 项的注意事项

- 必须用 `max-autotune-no-cudagraphs`：**开启 cudagraphs 会撞 DCU 图 bug**（`accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`）。
- 训练初期有形状专精导致的重编译抖动（实测 p10≈474 / p90≈547），形状缓存命中后消失。

## 3. 已验证无效的方向（排除清单）

以下方向均经过实测，**不要重复投入**：

### 3.1 算子 / kernel 层

| 方向 | 实测 |
|---|---|
| packed attention（cat/split 移入自定义 Function） | **-10%** |
| cross_feats 移出编译区（`dynamo.disable`） | **-3.8%** |
| 静态形状编译 | 0% |
| 主干加宽编译（`@torch.compile` 到 MEET.forward） | 0%（但确认**无数值损坏**） |
| inductor pointwise autotune | 0% |
| 自研 kernel 配置扫描（bm/bn/warps） | 0%（**当前配置已是最优**：fwd 64/16、bwd 32/32） |
| tile-blocked 连续布局（改善 P 缓冲合并） | 0%（证明瓶颈非访存） |
| `num_stages=2` 双缓冲 | **2.5× 恶化**（LDS 64KB 预算不足） |
| split-K GEMM | **0.07-0.39×** |
| 手写 Triton GEMM | 慢 2-2.5× |
| hipBLASLt | -18% |
| CK Tile | DTK 26.04 **不存在**该库 |
| 手写 MMAC HIP | 平台 bug（多 block 状态污染） |
| FA2 接管 LDM 注意力 | head dim ≤ 256 硬限（Dv=272 超限）；分片 V 方案 VMFault |
| VAE 原子→块池化融合 | kernel 快 2.8-3.0×，但 e2e **+0.5%（噪声内）** → 未采用 |

### 3.2 图捕获（CUDA graphs）

机械上**可行**（当前环境下能捕获并跑通 25 步、loss 正常），但所有配置均不可用：

| 配置 | 结果 |
|---|---|
| 多 bucket（256 / 512）+ 每图独立 pool | 每步 ~2 次 backward OOM（68.7GB 显存也不够；图池内存无法被 eager 反向复用） |
| 单图（N 桶 6656 + MAXLEN_BOUND 2560） | 仍 OOM / 数据集含 ~57K 块超大复合物 → 设备 fault |
| `expandable_segments` | 无效 |
| 桶化 + 静态形状 | **SIGABRT** |

### 3.3 其他

| 方向 | 实测 |
|---|---|
| dtype 统一（dist 分支 bf16） | b64 **0%**、b16 **-3.5%** → 已回退 |
| orjson 解析器 | +0.2%（噪声内）→ 按用户决定回退 |
| 优化器图捕获（kernel -30%） | e2e 仅 +1-1.5%，维护成本高 → **用户决定不采用**（保留 case 记录） |
| NUMA 绑核 | 双 GPU 同处 NUMA 0，无跨节点问题 |
| 数据管道（gzip+json → orjson） | 数据非 b64 瓶颈（DataLoader 单独 501 samples/s） |

## 4. MFU 分析（为什么这个负载天然低 MFU）

```
b64 每 rank：32 样本 × ~91 潜在块 = 2,918 token
有效 FLOPs/步 = LDM 训练 2.25 TF + 注意力 0.005 TF + VAE 前向 0.55 TF ≈ 2.81 TF
MFU = 2.81 / (480 TFLOPS × 0.123 s) ≈ 4.8%
```

| batch | MFU |
|---|---|
| b16 | 1.7% |
| b64 | 4.8% |
| b128 | 5.9% |

**结构性原因**（非"优化不足"）：

1. **每 rank token 数太少**（2,918 vs LLM 训练典型 4K-32K）→ 小 N 的 GEMM 只点亮 ~45% CU（80 个 CU 中约 36 tile）。
2. **模型小**（128M 参数）摊不动 480 TFLOPS 的大芯片。
3. **冻结 VAE 前置**（~20ms/步）产出的是训练输入，不产生被训练模型的 FLOPs。
4. **varlen 小序列**（~91 块/样本）+ 奇形头维（Dqk=144 / Dv=272）→ tile 利用率差、注意力 kernel 受 LDS bank conflict 限制（编译器内部布局，实测 0.8-3% 峰值）。

**结论**：即便平台四项全部修复，b64 上限约 800-900 samples/s（MFU 8-12%）。交付口径应以 **samples/s 对比基线**为准，而非 MFU。

## 5. 剩余空间与归属

| 归属 | 手段 | 空间 |
|---|---|---|
| **客户（协议）** | **b64 → b128**（需按线性缩放规则调 LR） | **+22.7%**（实测 530 → 650.2） |
| 客户（数据） | 数据集转二进制列式格式（消除 JSON 解析悬崖） | 0%（b64 下数据非瓶颈），长期保险 |
| 平台（已断供） | inductor 深度融合（36ms）、cat→index_put 回退（7.4ms）、图捕获/间隙（21ms）、LDS bank conflict（14ms） | +30-50% |
| 我们 | —— | **已穷尽**（见第 3 节排除清单） |

## 6. 验证状态

- `verify_all.py` 7 项精度校验全过（注意力前向/反向、VAE 路径、等价性）。
- 每轮优化均有同日双臂 A/B（对照 + 实验各 150 步）；b64 噪声带约 ±1-2%。
- 数值纪律：优先逐位等价；非逐位的改动（如 autotune 选不同 GEMM 算法）以 loss 同带为准。

## 7. 教训（方法论）

1. **理论修复必须 A/B**：本会话 20+ 次尝试，成功率约 30%。数值等价 ≠ 性能可行（packed、crossfix、tile-blocked 均属此类）。
2. **中性即回退**：无收益的改动一律还原，保持 diff 最小。
3. **跨口径对比会制造假瓶颈**：用 trace 的 kernel 时间对比墙钟时间，曾得出"带宽瓶颈"的错误结论（实测证伪）。
4. **先跑当日对照**：环境漂移下任何 A/B 必须先测当天基线。
5. **量化被消除算子的规模**："减少 N 个 launch"若算子本身太小，收益为零甚至为负。
