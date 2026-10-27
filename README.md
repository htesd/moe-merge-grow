# moe-merge-grow

固定显存预算下训练 MoE：训练过程中合并冗余 expert 释放容量，再把容量分配给新 expert（分裂或新建），
让同一块显存里装下尽量有用的参数。目前处于第一阶段：先验证"显存是否真的是瓶颈"，再测冗余，最后做合并/分裂循环。

## 实现思路

- **物理槽位池**（`mmg/pool.py`）：所有层共享一个固定大小的 expert 池，每个槽位是一个独立 Parameter
  `[3, d_ff, d_model]`（gate / up / down^T）。参数列表从头到尾不变，优化器和 checkpoint 不需要重建。
- **逻辑 expert 表**（`mmg/moe.py`）：每层预分配 router 行；若干行组成一个逻辑 expert（logit 取 logsumexp），
  逻辑 expert 指向一个物理槽位。跨层 tie = 多层指向同一槽位，梯度由 autograd 自动求和。
- **结构操作**（`mmg/ops.py`），都在两次 optimizer step 之间执行：
  - `merge_slots`：先做 hidden unit 对齐（置换 + up/down 逐单元缩放，保函数），再加权平均；
    Adam 状态 `m = m_a + m_b`，`v = (√v_a + √v_b)²`；同层合并变成一个 router 组，softmax 质量精确守恒。
  - `split_expert`：复制到空闲槽位，router 行复制并各减 log 2（pre-top-k 混合不变）；对已合并的组则是
    copy-on-write 拆回。
  - `untie_expert`：跨层 copy-on-write。`add_fresh_expert`：新随机 expert（down 置零），从 donor 分走一部分
    路由质量避免冷启动，Adam 状态从 step 0 开始。`drop_expert`：剪枝 baseline。`soft_pull`：先软后硬的拉近项。
- 空闲槽位不参与前向，grad 为 None，AdamW 直接跳过，optimizer state 也被释放。

## 使用

```bash
pip install -r requirements.txt          # torch 由环境提供
python -m unittest discover -s tests -v  # CPU 即可

python scripts/prepare_fineweb.py --out /path/to/fineweb_edu --num-files 4
python scripts/train.py --data-dir /path/to/fineweb_edu --out-dir runs/E64 \
    --experts-per-layer 64 --mem-budget-gb 24 --total-tokens 2e9
python scripts/summarize.py runs
```

`--mem-budget-gb` 用 `torch.cuda.set_per_process_memory_fraction` 在大显存卡上模拟小卡的显存上限。
训练支持断点续训（`out_dir/ckpt.pt`）。
