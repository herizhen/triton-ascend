# CV Fusion Operator Development

CV fusion operators refer to operators that simultaneously use Cube Core and Vector Core within the same operator: Cube Core typically handles `tl.dot`, matrix multiplication, or convolution-like main computations, while Vector Core handles bias, activation, softmax, reduction, mask, layout rearrangement, or cross-block synchronization. The goal of CV fusion is to reduce kernel boundaries and GM round trips, but it requires simultaneous control over Cube tile, Vector tile, UB/L1 occupancy, and synchronization relationships.

## Simple CV Fusion Operator Development

Simple CV fusion can start from the matmul + activation example in `third_party/ascend/tutorials/03-matrix-multiplication.py`, or refer to the [fused attention example](../examples/04_fused_attention_example.md). The minimal path is as follows:

1. First implement stable Cube main computation, e.g., `acc = tl.dot(a, b, acc)`.
2. Fuse lightweight Vector post-processing before writing back the accumulator, e.g., bias, scale, activation, or dtype cast.
3. Use sub-block partitioning for large accumulators to avoid UB overflow during the Vector post-processing phase.
4. If a single Cube output block needs to be split across multiple Vector sub-blocks for processing, use `extension.parallel(..., bind_sub_block=True)` and `extension.extract_slice` from the Ascend extension.

Example structure:

```python
acc = tl.dot(a, b, acc)

SUB_M: tl.constexpr = BLOCK_M // 2
for s in extension.parallel(0, 2, bind_sub_block=True):
    acc_sub = extension.extract_slice(acc, (s * SUB_M, 0), (SUB_M, BLOCK_N), (1, 1))
    acc_sub = tl.where(acc_sub >= 0, acc_sub, 0.01 * acc_sub)
    c_sub = acc_sub.to(tl.float16)
    tl.store(c_ptrs_for_sub_block, c_sub, mask=c_mask_for_sub_block)
```

When developing simple CV fusion, maintain clear boundaries: Cube is responsible for producing large two-dimensional accumulators, while Vector handles element-wise operations or small-scale reductions within the same tile. If the Vector part requires sharing state across multiple Cube tiles, synchronization, workspace, or kernel splitting becomes necessary.

## Complex CV Fusion Operator Development

For complex CV fusion, refer to the best practices in [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops):

- [`tutorial/best_practice/002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py): In decode attention, QK/PV uses Cube, while softmax, mask, exponent, normalization, and discrete KV memory access rearrangement use Vector.
- [`tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md): Demonstrates how to use `insert_slice`, transpose, and kernel splitting optimization to reduce discrete memory access and padding overhead when fusing cat, slice, and conv1d update.

Complex CV fusion is recommended to be organized hierarchically according to the data flow:

1. **Main Computation Layer**: Identify which steps must use Cube, e.g., QK, PV, GEMM, batched matmul.
2. **Vector Post-processing Layer**: Identify whether softmax, activation, mask, scale, normalization, cat/slice, layout transform, etc., can be completed within the same tile.
3. **Memory Access Rearrangement Layer**: For discrete KV cache, MoE token rearrangement, and tail-axis tensors, prioritize using `insert_slice`, `extract_slice`, transpose, or axis borrowing transpose in UB to form hardware-friendly contiguous access.
4. **Pipeline and Synchronization Layer**: Explore overlapping execution of Cube and Vector through compilation options such as `multibuffer`, `set_workspace_multibuffer`, `tile_mix_vector_loop`, `tile_mix_cube_loop`.
5. **Kernel Splitting Layer**: CV fusion operators typically launch grids based on the number of Cube Cores; at runtime, Vector Cores cooperate at approximately a 1:2 ratio. Do not simply adopt the large grid approach used on GPUs.

For attention-like CV fusion, it is recommended to first get non-causal, short sequence, small head_dim cases working, then gradually add:

- Causal mask processing in stages.
- Long sequence K/V block loops.
- Numerically stable softmax updates for `m_i`/`l_i`.
- Accumulator workspace and sub-block partitioning for large HEAD_DIM.
- Load rearrangement under discrete KV cache indices.

When tuning complex CV fusion, prioritize observing the time proportions of Cube, Vector, and MTE2 in profiling. If Cube is waiting for Vector, consider reducing the granularity of Vector post-processing or enabling CV balance-related options. If Vector is waiting for data transfer, first check discrete memory access, tail-axis padding, and multibuffer configuration.