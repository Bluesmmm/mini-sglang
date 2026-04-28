# mini-sglang 精读学习计划（按逻辑流程·对标 nano-vllm）

> 目标：跟着「一个请求从发起到返回」的完整链路，逐文件精读，每天一个文件。
> 
> 假设：你已看完 nano-vllm，熟悉 PagedAttention、continuous batching、Llama forward 的基本概念。
> 
> 学习心法：
> 1. **先走通链路，再深挖细节**——先建立「请求怎么走完全程」的全局感。
> 2. **对标 nano-vllm**——nano-vllm 有的模块是「复习」，mini-sglang 独有的模块是「重点」。
> 3. **每读完一个文件，回答文件末尾的 3 个核心问题**——答不上来就再读一遍。
> 4. **跑代码验证**——不要只读，要动手 print 张量 shape、加断点。

---

## nano-vllm ↔ mini-sglang 模块对照表

| nano-vllm 模块 | 职责 | mini-sglang 对应文件/目录 | 差异说明 |
|---|---|---|---|
| `__init__.py` / CLI | 入口 | `__main__.py` → `server/launch.py` | server 模式 vs library 模式 |
| — | 参数解析 | `server/args.py` | nano-vllm 没有独立参数文件 |
| — | HTTP API | `server/api_server.py` | nano-vllm 没有 HTTP server |
| `tokenizer.py` | 分词/解码 | `tokenizer/tokenize.py` + `detokenize.py` + `server.py` | 多进程隔离 |
| — | 进程通信 | `message/*.py` | nano-vllm 没有 IPC |
| `scheduler.py` | 调度 | `scheduler/scheduler.py` + `prefill.py` + `decode.py` | 更复杂：chunked prefill + cache 调度 |
| `kvcache.py` | KV 缓存 | `kvcache/*.py` | 多了 Radix Tree 前缀缓存 |
| `attention.py` | Attention | `attention/*.py` | 多后端：FA / FlashInfer / TRTLLM |
| `model.py` | 模型定义 | `models/*.py` + `layers/*.py` | 多了 TP 切分、MoE |
| — | 权重加载 | `models/weight.py` + `utils.py` | nano-vllm 没有独立 weight loader |
| `sampling.py` | 采样 | `engine/sample.py` | 功能相似，但放在 engine 子系统 |
| — | CUDA Graph | `engine/graph.py` | nano-vllm 没有 graph 优化 |
| — | TP 分布式 | `distributed/*.py` | nano-vllm 没有分布式 |
| — | MoE | `moe/*.py` + `layers/moe.py` | nano-vllm 没有 MoE |

---

## 学习路线总览（按请求链路）

```
用户请求
   │
   ▼
Day 1–4    入口与启动      __main__ → args → launch → core
   │
   ▼
Day 5–9    请求接入        api_server → tokenizer → message IPC
   │
   ▼
Day 10–14  调度            scheduler → prefill → decode → cache → io
   │
   ▼
Day 15–19  KV 缓存         base → pool → naive → radix_cache
   │
   ▼
Day 20–22  推理引擎        engine → graph → sample
   │
   ▼
Day 23–27  Attention 后端  base → fa → fi → trtllm
   │
   ▼
Day 28–36  模型定义        models → layers（embedding → linear → norm → rotary → attention → llama）
   │
   ▼
Day 37–38  权重加载        weight → utils
   │
   ▼
Day 39–40  高级特性        distributed → moe
```

---

## 第一阶段：入口与启动（Day 1–4）

> nano-vllm 是 library，直接 `from nanovllm import LLM` 就能用；mini-sglang 是 server，启动流程复杂得多。

### Day 1 — `python/minisgl/__main__.py`（5 行）+ `shell.py`（4 行）

- **对标 nano-vllm**：`__init__.py` / CLI 入口
- **核心问题**：
  - `python -m minisgl` 是怎么走到 `launch_server()` 的？
  - 为什么用 `__main__.py` 而不是 `setup.py` 的 entry_points？
- **差异点**：nano-vllm 是 import 即用，mini-sglang 是 server 启动模式。
- **实操**：把 README 的 quickstart 跑通，看启动日志里 fork 了哪些进程。

### Day 2 — `python/minisgl/server/args.py`（268 行）

- **对标 nano-vllm**：nano-vllm 没有独立参数文件，配置散落在各模块
- **核心问题**：
  - `--attention-backend` 有哪几个选项？对应哪些文件？
  - `--tp-size`、`--page-size`、`--max-running-requests` 各自影响什么？
  - 哪些参数是 mini-sglang 独有的？（提示：radix、page、tp、moe）
- **学习方法**：这是全项目的「地图」。看一遍 args 就知道项目支持哪些功能。
- **实操**：`python -m minisgl --help`，把每个参数在脑子里建索引。

### Day 3 — `python/minisgl/server/launch.py`（117 行）

- **对标 nano-vllm**：没有对应文件（nano-vllm 没有 server 进程启动逻辑）
- **核心问题**：
  - launch 流程一共 fork/创建了哪些进程？分别叫什么？
  - 每个进程的职责是什么？它们之间怎么通信？
- **关键概念**：**进程隔离架构**——tokenizer / scheduler / engine 各自独立。这是与 nano-vllm 最大的工程差异。
- **实操**：画一张「启动后的进程树」图。

### Day 4 — `python/minisgl/core.py`（136 行）

- **对标 nano-vllm**：类似于 `LLM` 类里的 `Sequence`、`SamplingParams` 等数据结构
- **核心问题**：
  - `Req` 类有哪些字段？与 nano-vllm 的 `Sequence` 对比，多了什么？
  - `ForwardBatch` 包含哪些信息？这些信息分别由谁填充？
- **关键地位**：这是整个项目的数据中枢，所有模块都 import 这个文件。
- **实操**：`grep -rn "from minisgl.core import" python/` 看看谁依赖了它。

---

## 第二阶段：请求接入（Day 5–9）

> nano-vllm 没有 HTTP server 和 IPC，用户直接 Python API 调用；mini-sglang 有完整的 server + message 协议。

### Day 5 — `python/minisgl/server/api_server.py`（444 行）

- **对标 nano-vllm**：nano-vllm 没有对应文件
- **核心问题**：
  - `/v1/chat/completions` 的请求体经过哪几跳到达 scheduler？
  - 流式输出（SSE）是怎么实现的？
  - `async` / `await` 在整个链路中的作用？
- **实操**：curl 发送一个 chat completion，在 api_server.py 里加 print 看请求体。

### Day 6 — `python/minisgl/tokenizer/tokenize.py`（31 行）+ `detokenize.py`（111 行）

- **对标 nano-vllm**：`tokenizer.py`
- **核心问题**：
  - tokenize 为什么这么简单？因为 heavy lifting 在哪？
  - detokenize 为什么比 tokenize 复杂得多？增量解码的关键问题是什么？
- **差异点**：mini-sglang 把 tokenizer 抽成了独立子进程（后面 Day 7 看 server）。
- **实操**：写个 5 行脚本调用 `tokenize()` 和 `detokenize()`。

### Day 7 — `python/minisgl/tokenizer/server.py`（110 行）

- **对标 nano-vllm**：nano-vllm 没有 tokenizer 子进程
- **核心问题**：
  - tokenizer server 的主循环长什么样？
  - 它接收什么消息、发出什么消息？
- **关键模板**：这是你第一次看到「子进程主循环」的模板。后面 scheduler / engine 都是同样的模式。

### Day 8 — `python/minisgl/message/frontend.py`（29 行）+ `backend.py`（41 行）

- **对标 nano-vllm**：nano-vllm 没有 IPC 层
- **核心问题**：
  - frontend 和 backend 的命名分别指代什么？
  - 一个请求从前端到后端要经过哪些消息类型？
- **关键地位**：这个文件读懂后，你才有资格说「我看懂了 mini-sglang 的架构」。

### Day 9 — `python/minisgl/message/tokenizer.py`（43 行）+ `utils.py`（69 行）+ `__init__.py`

- **对标 nano-vllm**：nano-vllm 没有 IPC 层
- **核心问题**：
  - 消息是怎么序列化的？用 json？raw bytes？还是其他格式？
  - 为什么 tokenizer 有专门的消息类？
- **实操**：画一张「请求消息流转图」：client → api_server → tokenizer → scheduler → engine → 反向回去。

---

## 第三阶段：调度器（Day 10–14）

> nano-vllm 的 `scheduler.py` 是一个文件；mini-sglang 把调度拆成了 6 个文件，功能更复杂。

### Day 10 — `python/minisgl/scheduler/scheduler.py`（267 行）

- **对标 nano-vllm**：`scheduler.py`（整体对应）
- **核心问题**：
  - `run_one_iter()` 的主循环长什么样？每个 iter 做什么？
  - request 的状态机有哪些状态？（waiting / running / finished）
  - prefill 和 decode 是在同一个 iter 里做，还是分开做？
- **实操**：画出 scheduler 的状态机流转图。

### Day 11 — `python/minisgl/scheduler/prefill.py`（162 行）

- **对标 nano-vllm**：nano-vllm 的 prefill 逻辑在 `scheduler.py` 里，没有单独文件
- **核心问题**：
  - 一个 waiting 的请求是怎么被选进 prefill batch 的？
  - chunked prefill 是怎么决定 chunk size 的？
  - prefix cache hit 时怎么省下 prefill？

### Day 12 — `python/minisgl/scheduler/decode.py`（39 行）

- **对标 nano-vllm**：`scheduler.py` 里的 decode 逻辑
- **核心问题**：
  - 为什么 decode.py 这么短？（提示：decode 逻辑简单——把 running 队列攒成 batch）
  - 如果 prefill 和 decode 在同一个 iter 做，decode batch 会被怎么影响？

### Day 13 — `python/minisgl/scheduler/cache.py`（146 行）

- **对标 nano-vllm**：nano-vllm 的 cache 管理在 `kvcache.py` 里
- **核心问题**：
  - 这一层与 `kvcache/radix_cache.py` 的区别是什么？
  - 什么时候做 allocate？什么时候做 free？
- **关键概念**：scheduler 层管「策略」（给谁分配多少），kvcache 层管「实现」（具体怎么放）。

### Day 14 — `python/minisgl/scheduler/io.py`（133 行）+ `config.py` + `utils.py` + `table.py`

- **对标 nano-vllm**：`scheduler.py` 里的请求收发逻辑
- **核心问题**：
  - scheduler 是怎么从 tokenizer 接收新请求的？
  - 结果是怎么发回 tokenizer / api_server 的？
  - `table.py` 的 RequestTable 是什么作用？

---

## 第四阶段：KV 缓存（Day 15–19）

> nano-vllm 的 `kvcache.py` 实现 PagedAttention 的 block manager；mini-sglang 不仅实现了分页，还实现了 **Radix Tree 前缀复用**。

### Day 15 — `python/minisgl/kvcache/__init__.py`（74 行）+ `kvcache/base.py`（135 行）

- **对标 nano-vllm**：`kvcache.py` 的接口层
- **核心问题**：
  - `BaseCacheHandle` 是什么？为什么需要 handle 而不是直接传 indices？
  - allocate / free / share 三组操作的语义？
  - `BaseKVCachePool` 提供了哪些方法？

### Day 16 — `python/minisgl/kvcache/mha_pool.py`（68 行）

- **对标 nano-vllm**：`kvcache.py` 里的物理内存池
- **核心问题**：
  - 张量布局是什么形状？`[num_pages, page_size, num_kv_heads, head_dim]`？
  - page_size > 1 怎么影响内存效率和访问模式？
- **对比**：nano-vllm page_size 固定为 1，mini-sglang 支持变量 page_size。

### Day 17 — `python/minisgl/kvcache/naive_cache.py`（45 行）

- **对标 nano-vllm**：`kvcache.py` 的简单分配器
- **核心问题**：
  - 这个 naive 版本做什么？和 radix 版本的区别？
  - 什么时候会用 naive 而不是 radix？

### Day 18 — `python/minisgl/kvcache/radix_cache.py`（236 行）Part 1：树结构 + 插入

- **对标 nano-vllm**：nano-vllm 没有对应功能！
- **核心问题**：
  - 节点存储什么？（token_ids、cached_indices、children、parent、ref_count）
  - insert 时，树是怎么分裂（split）的？
  - 什么叫「前缀共享」？举 3 个具体例子。
- **关键概念**：**Radix Tree（压缩 trie）** 用于复用「相同 prompt 前缀」的 KV。3 个用户都发「请翻译：你好」，「请翻译：」的 KV 只算一次。

### Day 19 — `python/minisgl/kvcache/radix_cache.py`（236 行）Part 2：match + evict

- **对标 nano-vllm**：nano-vllm 没有对应功能！
- **核心问题**：
  - match_prefix 是怎么走树的？
  - evict 时哪些节点是「可被驱逐」的？
  - LRU 驱逐策略怎么实现？
- **实操**：跑 `tests/core/test_cache_allocate.py`，理解每个 test case。

---

## 第五阶段：推理引擎（Day 20–22）

> nano-vllm 的 forward + sampling 在 `model.py` + `sampling.py` 里；mini-sglang 把这些放在 `engine/` 子系统里。

### Day 20 — `python/minisgl/engine/engine.py`（233 行）

- **对标 nano-vllm**：`model.py` 的 forward 调用 + `scheduler.py` 的执行部分
- **核心问题**：
  - engine 的主循环长什么样？
  - 一个 ForwardBatch 从进来到出去，经过了哪些步骤？
  - engine 与 scheduler 的边界在哪？
- **实操**：在 engine 主循环加 print，看一个 iter 里 `ForwardBatch` 的字段值。

### Day 21 — `python/minisgl/engine/graph.py`（171 行）

- **对标 nano-vllm**：nano-vllm 没有 CUDA Graph 优化
- **核心问题**：
  - 什么是 CUDA Graph？为什么对 decode 有用？
  - 为什么只对 decode 做 graph，不对 prefill 做？
  - 怎么处理可变 batch size？（提示：padding 到固定 buckets）

### Day 22 — `python/minisgl/engine/sample.py`（75 行）+ `engine/config.py`

- **对标 nano-vllm**：`sampling.py`
- **核心问题**：
  - temperature、top-k、top-p 的实现与 nano-vllm 的差异？
  - greedy decoding 的条件是什么？
  - `sample()` 的输入输出分别是什么？

---

## 第六阶段：Attention 后端（Day 23–27）

> nano-vllm 只有一种 attention 实现（PagedAttention kernel）；mini-sglang 支持 **3 种后端切换**。

### Day 23 — `python/minisgl/attention/base.py`（63 行）+ `utils.py`（23 行）+ `__init__.py`

- **对标 nano-vllm**：`attention.py` 的接口层
- **核心问题**：
  - `BaseAttnBackend` 提供了哪些必须实现的方法？
  - `BaseAttnMetadata` 包含哪些字段？在哪一步被填充？
  - `__init__.py` 里是怎么根据 `args.attention_backend` 选择后端的？
- **关键概念**：**两阶段调用**——先 prepare metadata，再 forward 时复用。

### Day 24 — `python/minisgl/attention/fa.py`（182 行）

- **对标 nano-vllm**：`attention.py`（flash-attn 实现）
- **核心问题**：
  - 怎么调用 `flash_attn_with_kvcache` API？
  - prefill 和 decode 调用的 API 是同一个吗？
  - block_table 是怎么传给 kernel 的？

### Day 25 — `python/minisgl/attention/fi.py`（271 行）Part 1：prefill

- **对标 nano-vllm**：nano-vllm 没有 flashinfer 后端
- **核心问题**：
  - flashinfer 的 plan / run 两阶段调用模式？
  - `qo_indptr`、`kv_indptr`、`paged_kv_indices` 这些 indptr 数组什么含义？
  - variable-length attention 的 indptr 表示法？
- **关键概念**：**indptr 数组** 是 paged attention 的核心数据结构，表示「哪些 token 属于哪个 sequence」。

### Day 26 — `python/minisgl/attention/fi.py`（271 行）Part 2：decode + cuda graph

- **核心问题**：
  - decode kernel 与 prefill kernel 的参数差异？
  - 为什么 cuda graph 需要单独路径？（提示：动态 shape）
  - flashinfer 的 `BatchDecodeWithPagedKVCacheWrapper` 是怎么复用的？

### Day 27 — `python/minisgl/attention/trtllm.py`（162 行）

- **对标 nano-vllm**：nano-vllm 没有 TRTLLM 后端
- **核心问题**：
  - TRTLLM 后端与 FA/FI 的接口差异？
  - 三个后端的 metadata 分别长什么样？
  - 不同后端适合什么场景？（提示：FA 通用、FI 性能最强、TRTLLM NVIDIA 专用）
- **实操**：在启动参数里切换不同 backend，跑 benchmark 对比延迟。

---

## 第七阶段：模型定义（Day 28–36）

> nano-vllm 的 `model.py` 包含了 Llama 模型定义；mini-sglang 拆成了 `models/`（高层组装）+ `layers/`（底层算子），并且多了 **TP 切分**。

### Day 28 — `python/minisgl/models/base.py`（14 行）+ `register.py`（23 行）+ `utils/registry.py`

- **对标 nano-vllm**：`model.py` 里的注册逻辑
- **核心问题**：
  - 为什么用 registry 而不是 if-else 分发模型？
  - `@register_model("llama")` 装饰器做了什么？

### Day 29 — `python/minisgl/models/config.py`（86 行）

- **对标 nano-vllm**：`model.py` 里的 `ModelConfig`
- **核心问题**：
  - `hidden_size`、`num_attention_heads`、`num_kv_heads` 的关系？
  - GQA 在 config 里怎么体现？
  - rope_theta、rope_scaling 是干什么用的？

### Day 30 — `python/minisgl/layers/base.py`（99 行）

- **对标 nano-vllm**：`model.py` 里的 layer 基类
- **核心问题**：
  - 这个 base 与 PyTorch `nn.Module` 的关系？
  - 为什么需要自己包一层？（提示：权重加载 hook）
  - `load_weights()` 是怎么被调用的？

### Day 31 — `python/minisgl/layers/embedding.py`（110 行）

- **对标 nano-vllm**：`model.py` 里的 Embedding
- **核心问题**：
  - `VocabParallelEmbedding` 是怎么按 rank 切分词表的？
  - all-reduce 是在 embedding 阶段还是在 LM head 阶段做？
- **差异点**：nano-vllm 没有 TP，这是新东西。

### Day 32 — `python/minisgl/layers/linear.py`（127 行）

- **对标 nano-vllm**：`model.py` 里的 Linear
- **核心问题**：
  - `ColumnParallelLinear` vs `RowParallelLinear` 的区别？
  - QKV proj 为什么用 column parallel？O proj 为什么用 row parallel？
  - `MergedColumnParallel`（qkv 合并权重）在 TP 下怎么切？
- **关键概念**：TP 的 column / row 分别对应权重的 dim=0 / dim=1 切分。

### Day 33 — `python/minisgl/layers/norm.py`（38 行）+ `activation.py`（21 行）

- **对标 nano-vllm**：`model.py` 里的 Norm + Activation
- **核心问题**：
  - RMSNorm 与 LayerNorm 的差异？
  - SiLU（Swish）激活函数长什么样？
- **实操**：对比 PyTorch 的 `nn.LayerNorm`，看自定义 RMSNorm 多了什么（提示：fused kernel）。

### Day 34 — `python/minisgl/layers/rotary.py`（145 行）

- **对标 nano-vllm**：`model.py` 里的 RoPE
- **核心问题**：
  - cos/sin cache 是预计算的还是 on-the-fly 算的？
  - 为什么 rotary 要支持多种 scaling？（LLaMA 3 长上下文需要 yarn/linear）
  - apply_rotary_pos_emb 的输入输出 shape？

### Day 35 — `python/minisgl/layers/attention.py`（57 行）

- **对标 nano-vllm**：`model.py` 里的 Attention 层
- **核心问题**：
  - 这个文件把 QKV proj + RoPE + KV cache + attn backend + O proj 串起来了？
  - 它调用了 attention backend 的哪个接口？
- **关键地位**：这是连接「模型定义」和「attention 后端」的桥梁。

### Day 36 — `python/minisgl/models/llama.py`（85 行）+ `models/mistral.py`

- **对标 nano-vllm**：`model.py` 里的 LlamaModel
- **核心问题**：
  - `LlamaModel` 与 `LlamaForCausalLM` 的层次关系？
  - forward 函数里张量形状怎么变化？（建议画出来）
  - Mistral 和 Llama 的差异在哪？（hint：Sliding Window Attention）
- **实操**：把两边的 llama.py 并排打开，逐行对比。

---

## 第八阶段：权重加载（Day 37–38）

> nano-vllm 没有独立的 weight loader，权重加载是 inline 在模型初始化里的。

### Day 37 — `python/minisgl/models/weight.py`（124 行）

- **对标 nano-vllm**：nano-vllm 没有对应文件
- **核心问题**：
  - merged QKV 权重是怎么从 HF 的三个分立权重合并的？
  - TP 切分发生在加载时还是 forward 时？
  - weight_loader 是怎么被 Layer 的 `load_weights()` 调用的？
- **挑战**：跟着 `weight.py` 走一遍 llama 模型的所有权重，搞清楚每个 hf_name → minisgl_name 的映射。

### Day 38 — `python/minisgl/models/utils.py`（126 行）+ `utils/hf.py`（51 行）

- **对标 nano-vllm**：`utils.py` 里的下载逻辑
- **核心问题**：
  - `download_weights` 支持哪些来源？（HF Hub、ModelScope、本地路径）
  - `ModelScope` 支持是怎么做的？
- **实操**：让 `weight.py` 加载一个真实的小模型（如 Qwen2-0.5B），打印每个 layer 的权重 shape。

---

## 第九阶段：高级特性（Day 39–40）

> 这两块是「高级特性」，第一次过项目可以选读，但建议至少浏览一遍。

### Day 39 — `python/minisgl/distributed/info.py`（38 行）+ `distributed/impl.py`（97 行）+ `kernel/pynccl.py`（78 行）

- **对标 nano-vllm**：nano-vllm 没有分布式
- **核心问题**：
  - TP 通信原语有哪些？（all_reduce、all_gather）
  - 哪些 layer 需要 all_reduce？哪些需要 all_gather？
  - rank 0 与其他 rank 在做什么不一样的事？
- **关键概念**：TP 下每个 rank 只算一部分，最后 all-reduce / all-gather 把结果拼回来。

### Day 40 — `python/minisgl/moe/base.py`（18 行）+ `layers/moe.py`（59 行）+ `moe/fused.py`（256 行）

- **对标 nano-vllm**：nano-vllm 没有 MoE
- **核心问题**：
  - MoE 的 router 是什么？top-k gating 怎么工作？
  - `fused_moe` 做了哪些融合优化？
  - `moe_impl.py` 里的 Triton kernel 是做什么的？
- **可选**：如果时间紧，这三天可以合并到一天，只读 `layers/moe.py` 的接口，跳过 Triton kernel。

---

## 附：额外的模型文件（可选）

| 文件 | 行数 | 说明 |
|---|---|---|
| `models/qwen2.py` | 83 | Qwen2 dense model，和 llama 几乎一样 |
| `models/qwen3.py` | 83 | Qwen3 dense model |
| `models/qwen3_moe.py` | 83 | Qwen3 MoE model，学完 Day 40 再读 |
| `models/qwen2.py` | 83 | Qwen2 dense model |
| `benchmark/client.py` | 501 | benchmark 客户端，长但都是 HTTP 请求逻辑 |
| `benchmark/perf.py` | 74 | 性能统计工具 |
| `utils/mp.py` | 151 | 多进程工具 |
| `utils/logger.py` | 128 | 日志系统 |
| `env.py` | 87 | 环境变量配置 |

---

## 每天的标准学习流程（60–90 分钟）

1. **5 分钟 — 预热**：通读文件一遍，不求理解，获得整体感觉。
2. **30 分钟 — 精读**：逐函数/逐类阅读，每个都要能用一句话概括。
3. **15 分钟 — 追踪调用链**：`grep -rn "ClassName" python/` 找谁调用它、它调用谁。
4. **10 分钟 — 实验**：写最小调用脚本，或者改参数看结果。
5. **20 分钟 — 笔记**：用自己的话写 3 条要点，标记 1 个还没搞懂的疑问。

---

## 卡壳了怎么办？

- **5 分钟规则**：某段代码看 5 分钟还看不懂就跳过，标记下来继续读。等读完更多上下文回头看，常常豁然开朗。
- **画图大于读字**：复杂控制流（scheduler 主循环、radix tree 操作）一定要画图。
- **求助 LLM 的正确姿势**：不要问「这段代码什么意思」，要问「这段代码在解决什么问题？设计上还有哪些其他选择？」

---

## 完成里程碑

- ✅ **Day 4 后**：能给朋友讲清楚 mini-sglang 启动后的进程拓扑。
- ✅ **Day 9 后**：能画出「一个请求从 curl 到 scheduler」的完整链路图。
- ✅ **Day 14 后**：能讲清楚一个 iter 里 scheduler 怎么决定 prefill 和 decode。
- ✅ **Day 19 后**：能手写简化版 Radix Tree 的 insert + match（哪怕只支持字符串）。
- ✅ **Day 27 后**：能解释三种 attention backend 的差异、各自适合什么场景。
- ✅ **Day 36 后**：能脱稿画出 LlamaForCausalLM 的 forward 全流程（含张量 shape）。
- ✅ **Day 40 后**：能给一个新功能（比如 speculative decoding）规划落地方案——具体改哪些文件、哪些 layer。

祝学习愉快！
