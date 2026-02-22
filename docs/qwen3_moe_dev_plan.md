# Qwen3 MoE 支持开发计划

## 一、背景

当前 nano-vllm 仅支持 dense 模型（`Qwen3ForCausalLM`）。Qwen3 MoE 系列模型（如 Qwen3-30B-A3B）在 Decoder Layer 中引入了 **Mixture of Experts (MoE)** 结构，用多个稀疏激活的 Expert MLP 替代单一 Dense MLP，显著提升了模型容量同时保持较低的计算开销。

### Qwen3 MoE 与 Dense 的关键差异

| 特性 | Dense (Qwen3) | MoE (Qwen3MoE) |
|---|---|---|
| MLP 层 | 单一 `Qwen3MLP` | Router + N 个 Expert MLP + Shared Expert |
| 每 token 计算量 | `2 × hidden × intermediate` | `2 × top_k × hidden × moe_intermediate + shared` |
| HF Config | `Qwen3Config` | `Qwen3MoeConfig` |
| `model_type` | `"qwen3"` | `"qwen3_moe"` |
| 额外 config 字段 | 无 | `num_experts`, `num_experts_per_tok`, `decoder_sparse_step`, `moe_intermediate_size`, `shared_expert_intermediate_size`, `norm_topk_prob` |

### Qwen3MoE 层结构示意

```
DecoderLayer(i):
  ├── input_layernorm (RMSNorm)
  ├── self_attn (Qwen3Attention)        ← 与 Dense 完全一致
  ├── post_attention_layernorm (RMSNorm)
  └── mlp:
      ├── if i % decoder_sparse_step == 0:  ← MoE 层
      │     ├── gate (Router Linear: hidden_size → num_experts)
      │     ├── experts[0..N-1] (Qwen3MLP × num_experts)
      │     └── shared_expert (Qwen3MLP × 1)
      └── else:                              ← Dense 层
            └── Qwen3MLP (与现有实现一致)
```

---

## 二、并行策略选择

### 方案 A：Expert Tensor Parallelism（推荐首选）

每个 Expert 内部沿用现有的 Column/Row Parallel 策略（`MergedColumnParallelLinear` + `RowParallelLinear`），所有 GPU 持有所有 Expert 的分片。

- **优点**：实现最简单，完全复用现有层；通信模式不变（每层 2 次 all-reduce）；CUDA Graph 友好
- **缺点**：所有 GPU 都存储全部 Expert 参数的切片，显存占用 = 总参数 / tp_size

### 方案 B：Expert Parallelism (EP)

将 N 个 Expert 按 `expert_id % tp_size` 分配到不同 GPU，每 GPU 仅持有 `N / tp_size` 个完整 Expert。

- **优点**：显存效率更高（每 GPU 只存部分 Expert）
- **缺点**：需要 all-to-all 通信（token dispatch/combine）；实现复杂度高；CUDA Graph 不兼容（动态 routing）

### 决策

**第一阶段采用方案 A（Expert TP）**，架构最简，与现有代码兼容性最好。后续可扩展 EP 支持。

---

## 三、开发任务分解

### Phase 1：核心模型实现

#### 1.1 新建 `nanovllm/models/qwen3_moe.py`

新增以下类（Attention 层直接复用 `qwen3.py` 中的 `Qwen3Attention`）：

| 类名 | 说明 |
|---|---|
| `Qwen3MoeSparseMoeBlock` | MoE 核心模块：Router + Expert MLP 组 + Shared Expert |
| `Qwen3MoeDecoderLayer` | 解码器层：根据 `layer_idx` 选择 MoE 或 Dense MLP |
| `Qwen3MoeModel` | Transformer 主干：Embedding → N × DecoderLayer → RMSNorm |
| `Qwen3MoeForCausalLM` | 顶层模型：Qwen3MoeModel + LM Head + packed_modules_mapping |

**关键实现细节：**

##### `Qwen3MoeSparseMoeBlock`

```python
class Qwen3MoeSparseMoeBlock(nn.Module):
    """MoE 稀疏专家模块：Router + Top-K Expert + Shared Expert。"""

    def __init__(self, config):
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # Router: hidden_size → num_experts
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, bias=False)

        # Expert MLP 组
        self.experts = nn.ModuleList([
            Qwen3MLP(config.hidden_size, config.moe_intermediate_size, config.hidden_act)
            for _ in range(config.num_experts)
        ])

        # Shared Expert（始终激活）
        self.shared_expert = Qwen3MLP(
            config.hidden_size, config.shared_expert_intermediate_size, config.hidden_act
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. Router → topk expert indices + weights
        # 2. 对每个 expert，选出对应 token 子集，前向计算，加权求和
        # 3. Shared expert 前向 + gate
        # 4. 合并输出
        ...
```

> **CUDA Graph 兼容性关键**：Router 的 token-to-expert dispatch 涉及动态索引。为保证 CUDA Graph 可捕获，应避免 `torch.where` 产生动态形状，改用固定形状的 permute + padding 方案（或在 decode 阶段直接对所有 expert 遍历，因 batch_size 较小开销可控）。

##### `Qwen3MoeDecoderLayer`

```python
class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        self.self_attn = Qwen3Attention(...)
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)

        # 根据 layer_idx 决定使用 MoE 还是 Dense MLP
        if layer_idx % config.decoder_sparse_step == 0:
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
```

##### `Qwen3MoeForCausalLM.packed_modules_mapping`

需要扩展以覆盖 Expert 内部的权重映射：

```python
packed_modules_mapping = {
    # Attention QKV
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    # Dense MLP（非 MoE 层）
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    # Expert MLP 内部同理——因使用子串匹配，
    # experts.{i}.gate_proj 和 experts.{i}.up_proj 也会被上面的规则匹配到
}
```

#### 1.2 修改 `nanovllm/engine/model_runner.py`

将模型类选择从硬编码改为动态分派：

```python
# 当前：
from nanovllm.models.qwen3 import Qwen3ForCausalLM
self.model = Qwen3ForCausalLM(hf_config)

# 改为：
MODEL_REGISTRY = {
    "qwen3": Qwen3ForCausalLM,
    "qwen3_moe": Qwen3MoeForCausalLM,
}
model_cls = MODEL_REGISTRY[hf_config.model_type]
self.model = model_cls(hf_config)
```

#### 1.3 验证权重加载 (`nanovllm/utils/loader.py`)

当前 `packed_modules_mapping` 使用子串匹配（`if k in weight_name`），需要确认：

- `"gate_proj"` 能正确匹配 `model.layers.X.mlp.experts.Y.gate_proj.weight`  ✅
- `"gate_proj"` 不会错误匹配 Router 的 `model.layers.X.mlp.gate.weight`  ✅（`"gate_proj"` 不是 `"gate"` 的子串）
- `"gate"` 如果被加入 mapping，需确保不会匹配到 `"gate_proj"` → **匹配顺序需要注意，先匹配长的 key**

可能需要的改动：调整匹配顺序或使用更精确的匹配（如 `.gate_proj.` vs `.gate.`）。

---

### Phase 2：Router 实现与 Token Dispatch

Router 是 MoE 的核心组件，负责将每个 token 分配到 top-k 个 Expert。

#### 2.1 Router Forward

```python
def route(self, hidden_states: torch.Tensor):
    """
    Args:
        hidden_states: [num_tokens, hidden_size]
    Returns:
        topk_weights: [num_tokens, top_k]   Top-K 路由权重
        topk_ids:     [num_tokens, top_k]   Top-K Expert 索引
    """
    # Router logits
    router_logits = self.gate(hidden_states)          # [num_tokens, num_experts]
    scores = F.softmax(router_logits, dim=-1)         # 概率化

    # Top-K 选择
    topk_weights, topk_ids = torch.topk(scores, self.top_k, dim=-1)

    # 可选：归一化 top-k 权重
    if self.norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids
```

#### 2.2 Token Dispatch 策略

**朴素实现（推荐首选）**：遍历每个 Expert，用 mask 筛选对应 token：

```python
final_hidden = torch.zeros_like(hidden_states)
for expert_idx, expert in enumerate(self.experts):
    mask = (topk_ids == expert_idx).any(dim=-1)     # [num_tokens]
    if mask.any():
        token_subset = hidden_states[mask]
        expert_out = expert(token_subset)
        # 加权写回
        weight = topk_weights[mask, (topk_ids[mask] == expert_idx).long().argmax(-1)]
        final_hidden[mask] += weight.unsqueeze(-1) * expert_out
```

**优化方向（后续）**：
- **Grouped GEMM**：将所有 Expert 的输入按 Expert 分组，使用 Triton grouped GEMM kernel 一次完成
- **Fused MoE Kernel**：参考 vLLM 的 `fused_moe` Triton kernel，避免逐 Expert 循环

---

### Phase 3：CUDA Graph 兼容性

MoE 的 token routing 涉及动态形状操作，对 CUDA Graph 不友好。需要处理：

#### 3.1 Decode 阶段（核心路径）

Decode 时 batch 中每个序列只有 1 个新 token，`num_tokens = batch_size`，规模较小。

**方案**：对所有 Expert 执行前向（跳过 mask 为空的 Expert），或 padding 到统一形状：

```python
# Decode 时简单遍历所有 expert，因为 token 数少开销可控
for expert_idx, expert in enumerate(self.experts):
    expert_out = expert(hidden_states)  # 全量计算
    mask = (topk_ids == expert_idx).any(dim=-1)
    weight = ...  # 从 topk_weights 中提取
    final_hidden += mask.unsqueeze(-1) * weight.unsqueeze(-1) * expert_out
```

此方案对所有 Expert 执行前向（计算冗余但形状固定），确保 CUDA Graph 可捕获。

#### 3.2 Prefill 阶段

Prefill 使用 eager 模式（不进 CUDA Graph），因此 **无 CUDA Graph 限制**，可自由使用动态 dispatch。

#### 3.3 实现建议

在 `Qwen3MoeSparseMoeBlock.forward` 中判断阶段：

```python
def forward(self, hidden_states):
    topk_weights, topk_ids = self.route(hidden_states)

    if get_context().is_prefill:
        # Prefill: 动态 dispatch，只计算激活的 expert
        output = self._forward_dynamic(hidden_states, topk_weights, topk_ids)
    else:
        # Decode: 固定形状，CUDA Graph 友好
        output = self._forward_static(hidden_states, topk_weights, topk_ids)

    # Shared Expert
    shared_out = self.shared_expert(hidden_states)
    shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
    output = output + shared_out * shared_gate

    return output
```

---

### Phase 4：显存管理优化

#### 4.1 模型参数显存估算

MoE 模型参数量显著大于同级别 Dense 模型：

| 模型 | 总参数 | 激活参数 | 显存(FP16) |
|---|---|---|---|
| Qwen3-30B-A3B | ~30B | ~3B/tok | ~60GB |
| Qwen3-235B-A22B | ~235B | ~22B/tok | ~470GB |

KV-Cache 大小不变（MoE 不影响 Attention 层），但**模型权重占用显著增加**，留给 KV-Cache 的显存减少。

#### 4.2 `allocate_kv_cache` 适配

当前逻辑：

```python
free = total_gpu_mem * utilization - peak_after_warmup
num_blocks = free // per_block_size
```

MoE 模型 `peak_after_warmup` 更大 → `num_blocks` 更少，这个逻辑**自动生效，无需修改**。

#### 4.3 后续优化方向

- **Expert Offloading**：将不活跃 Expert 参数卸载到 CPU/NVMe，按需加载
- **Expert Quantization**：对 Expert 参数做 INT8/INT4 量化，减少显存占用
- **Expert Parallelism (EP)**：跨 GPU 分配 Expert，Phase 1 未实现

---

### Phase 5：测试与验证

#### 5.1 正确性验证

| 测试项 | 方法 |
|---|---|
| 权重加载 | 对比 HF `transformers` 加载后的 `state_dict` 中每个参数的 shape 和值 |
| 单 token 输出 | 与 HF `model.generate()` 对比 prefill 阶段的 logits（`atol=1e-3`） |
| 多 token 生成 | 对比 greedy decoding 若干步的输出 token 序列 |
| Router 验证 | 打印 routing 分布，确认 top-k 选择和权重归一化正确 |
| Shared Expert | 单独验证 shared expert 的前向输出 |

#### 5.2 性能测试

| 测试项 | 指标 |
|---|---|
| Throughput | `bench.py` 对比 vLLM MoE 的吞吐量 |
| Latency | 单 token decode 延迟（CUDA Graph on/off） |
| 显存占用 | `torch.cuda.max_memory_allocated()` 峰值 |
| Expert 负载均衡 | 统计各 Expert 被 route 到的 token 数分布 |

#### 5.3 边界情况

- `decoder_sparse_step > 1` 时 Dense/MoE 层交替的正确性
- `tensor_parallel_size > 1` 时 Expert 并行的正确性
- `enforce_eager=True/False` 时 CUDA Graph 的兼容性
- 极长序列（prefill token 数 >> `max_model_len`）

---

## 四、文件改动清单

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `nanovllm/models/qwen3_moe.py` | **新增** | MoE 模型定义（4 个类） |
| `nanovllm/models/__init__.py` | 新增 | 可选：注册模型 |
| `nanovllm/engine/model_runner.py` | **修改** | 模型类动态选择（`MODEL_REGISTRY`） |
| `nanovllm/utils/loader.py` | **可能修改** | 权重名匹配精度提升（如需） |
| `nanovllm/config.py` | 无需修改 | HF AutoConfig 自动解析 MoE 字段 |
| `nanovllm/engine/block_manager.py` | 无需修改 | KV-cache 与 MoE 无关 |
| `nanovllm/engine/scheduler.py` | 无需修改 | 调度逻辑与 MoE 无关 |
| `nanovllm/layers/*` | 无需修改 | 现有层已满足需求 |
| `nanovllm/utils/context.py` | 无需修改 | Context 信息与 MoE 无关 |

---

## 五、开发里程碑

### M1：基础 MoE 推理（~3-5 天）

- [ ] 实现 `Qwen3MoeSparseMoeBlock`（朴素 Python 循环 dispatch）
- [ ] 实现 `Qwen3MoeDecoderLayer`（Dense/MoE 层交替）
- [ ] 实现 `Qwen3MoeModel` + `Qwen3MoeForCausalLM`
- [ ] 修改 `model_runner.py` 支持动态模型选择
- [ ] 验证权重加载正确性
- [ ] 对比 HF transformers 输出正确性（greedy decode）

### M2：CUDA Graph 兼容（~2-3 天）

- [ ] 实现 decode 阶段固定形状的 MoE forward
- [ ] 验证 CUDA Graph 捕获和回放正确性
- [ ] 对比 eager 和 CUDA Graph 输出一致性

### M3：性能优化（~3-5 天）

- [ ] Triton fused MoE kernel（grouped GEMM）替代 Python 循环
- [ ] 显存峰值优化（Expert 计算流水线化）
- [ ] 吞吐量基准测试 + 对标 vLLM

### M4：扩展功能（可选，~3-5 天）

- [ ] Expert Parallelism (EP) 支持
- [ ] Expert 量化（INT8/INT4）
- [ ] Expert Offloading（CPU/NVMe）
- [ ] 多 MoE 架构支持（DeepSeek MoE 等）

---

## 六、风险与注意事项

1. **CUDA Graph 兼容性**：MoE routing 的动态性是最大挑战，decode 阶段必须使用固定形状方案
2. **权重名冲突**：`gate`（Router）与 `gate_proj`（Expert MLP）的子串匹配需仔细验证
3. **显存压力**：MoE 模型参数量大，可能导致 KV-cache 可用 blocks 大幅减少
4. **数值精度**：Router softmax + topk 的数值稳定性需关注（建议 FP32 计算 routing）
5. **Shared Expert Gate**：部分 Qwen3 MoE 变体可能没有 `shared_expert_gate`，需兼容处理
