# Triton Operator Development Guide

Overview: This article focuses on notable issues in developing Triton operators on the NPU, covering three aspects: multi-core task parallelism, single-core data movement, and single-core data computation. First, it introduces the basis for setting the maximum number of hardware cores and the specific implementation in multi-core task parallelism. Then, it describes how to set an appropriate data block size within loops for single-core data movement, introduces common optimization techniques used in the process, and supplements the handling of potential UB OVERFLOW issues. Finally, it returns to individual operators, focusing on how to develop Triton operators at the single-core data computation level and emphasizing related key points.

## Document Organization

This guide separates general development principles from operator development paths categorized by hardware execution units:

- This page introduces common issues that all Triton-Ascend operators need to consider, including core division, on-chip memory, memory access, Tiling, and Autotune.
- [Vector Operator Development](./vector_operator.md) introduces operators primarily executed by the Vector Core, such as element-wise, reduction, Gather/Scatter, etc.
- [Cube Operator Development](./cube_operator.md) introduces operators centered around `tl.dot`, matrix multiplication, and batched matrix multiplication.
- [CV Fusion Operator Development](./cv_fusion_operator.md) introduces scenarios where Cube computation and Vector post-processing, reduction, Softmax, or cross-core collaboration coexist within the same operator.

For simple operators, refer to `docs/zh/examples/` and `third_party/ascend/tutorials/` in this repository first; for complex operators, refer to the complete optimization cases in `tutorial/best_practice/` on GitHub's [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops) repository.

## General Multi-Core Task Parallelism

### Setting the Maximum Number of Hardware Cores

In a Triton operator, a grid is typically used for core division. For GPUs, the number of compute cores (SMs) is usually in the tens to hundreds. However, for the Ascend NPU platform, the number of compute cores (AI Cores) is in the tens. \
Although the runtime interface allows a maximum of 65535 concurrent tasks, tasks exceeding the number of physical cores are executed through a new round of dispatching. If Triton operators from GPUs are directly run on the Ascend platform, these numerous tasks will introduce significant overhead from kernel launch and initialization, affecting operator performance. \
Therefore, the core division logic needs to be modified for the Ascend platform characteristics. The most recommended approach is to **directly fix the number of cores to the physical core count of the hardware** and perform more detailed data division within the core:

* For pure Vector operators, the number of cores equals the **number of Vector cores**
* For CV fusion operators, the number of cores equals the **number of Cube cores** (usually half the number of Vector cores), and the operator execution will call Vector cores in a 1:2 ratio

Generally, on an NPU card, one compute core (AI Core) contains one Cube core, and each Cube core is paired with two Vector cores. Therefore, the **number of Vector cores (vectorcore_num)** and **number of Cube cores (aicore_num)** can be obtained through the following interface:

```python
import torch
import triton.runtime.driver as driver
import torch_npu

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]

```

Refer to the example code below: first fix the number of cores, then process task blocks in batches through an inner loop:

```python
NUM_CORE = vectorcore_num
grid = (NUM_CORE ,)
_attn_fwd[grid](Q, K, V, M, Out, acc, scale, ...)

@triton.jit
def _attn_fwd(Q, K, V, M, Out, acc, scale,
              ...,
              stride_qz, stride_qh,
              Z: tl.constexpr, H: tl.constexpr,
              N_CTX: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr
              ):
    # Calculate total tasks, flatten 3D tasks (Z, H, M) into 1D total task count
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    # Each core selects tasks based on its own identifier
    pid = tl.program_id(0)  # Unique ID of the current core
    NUM_CORE = tl.num_programs(0)  # Get the fixed total number of launched cores
    # Loop rule: range(pid, NUM_BLOCKS, NUM_CORE) implements "strided task assignment"
    # - Start value pid: each core starts fetching tasks from its own ID to avoid task overlap
    # - Step NUM_CORE: stride by the total number of cores to ensure tasks are evenly distributed
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        # Calculate data offset for each task
        # 【Core: Reverse the 1D task index back to the original multi-dimensional index】
        # block_idx is the flattened 1D task index, split back into original dimensions using integer division/modulo
        # 1. Split the combined Z+H axis & M block axis:
        #   - Integer division by NUM_BLOCKS_M: extract the index of the combined Z+H axis (task_hz_idx)
        #   - Modulo by NUM_BLOCKS_M: extract the block index of the M dimension (task_m_idx)
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        # 2. Split the combined Z+H axis into original Z and H axes:
        #   - Integer division by H: restore the Z axis index (off_z)
        #   - Modulo by H: restore the H axis index (off_h)
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        # 3. Calculate data offset: based on the restored Z/H indices, locate the starting data position in Q/K/V tensors
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
```

## General Single-Core Data Movement

### Setting an Appropriate Data Block Size (BLOCK SIZE) within Loops

Taking `add_kernel` as an example, variables and operations together determine the on-chip memory space usage. Modifying the `BLOCK_SIZE` can adjust the size of data blocks and intermediate computation results within the loop. If the upper limit is exceeded, the compiler will prompt the expected size and report an error during compilation. To achieve the maximum compute-to-memory-access ratio, `BLOCK_SIZE` should be as large as possible without exceeding the on-chip space. This can be achieved by pre-setting different `BLOCK_SIZE` values using Triton-Ascend's [Autotune](../examples/06_autotune_example.md), and the runtime will automatically select the optimal setting.

```python
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               out_ptr,
               n,  # Total number of elements.
               BLOCK_SIZE: tl.constexpr,  # Number of elements per block.
               ):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n, BLOCK_SIZE)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        block_start = block_idx * BLOCK_SIZE
        # Block size is BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        # Load x, y data to on-chip memory
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        output = x + y

        tl.store(out_ptr + offsets, output, mask=mask)
```

### Ensure Data Alignment of the Tensor's Last Dimension as Much as Possible

[Description] For VV-type operators that require Vector core computation, the Ascend hardware's UB requires the size of the tensor's last dimension to be divisible by 32 Bytes. For CV-type operators that require both Vector and Cube core computation, the size of the tensor's last dimension must be divisible by 512 Bytes. If the last dimension length is insufficient, it will be automatically padded. In this context, various operations on tensors with shapes like (2048, 3) and (2048, 1) in a model will suffer significant performance degradation due to automatic padding. In such cases, consider using a transpose operation to move the alignment axis to a lower dimension, and transpose back to the original state only when storing, thereby avoiding automatic padding and optimizing computation speed. Since the transpose operation itself is also affected by automatic padding rules, special techniques are also needed to avoid padding. Here is a tip called "borrowing an axis for transpose," applicable to scenarios where **tensor.numel() % 256Byte == 0**. The specific operation is as follows:

- Note: VV-type operators refer to operators that only use the Vector Core during computation; CV-type operators refer to operators that use both the AI Core and the Vector Core during computation.
- Example

```python
# conv_state = tensor([2048, 3], bfloat16)
conv_state = tl.load(conv_state_ptr + conv_batch_offs * conv_batch_stride + doffs * 3 + tl.arange(0, 2048 * 3)) # Load as 1D tensor, no automatic padding since numel is aligned.
conv_state_T = conv_state.reshape(128, 16 * 3).trans().reshape(16, 3 * 128).trans().reshape(3 * 2048,) # Split the long axis (2048) to borrow an alignment axis (16) for the short axis (3), making both axes aligned.
```

### First Move Data to UB, Then Select Target Values from UB

[Description] In NPU discrete scenarios, data can be first moved to UB, and then target values can be selected from the shared memory.

- Example

```diff
@triton.jit
def pick_kernel(
        x_ptr,
        idx_ptr,
        y_ptr,
        stride_x,
        stride_idx,
        stride_y,
        M: tl.constexpr,
        N: tl.constexpr
):
    pid = tl.program_id(0)
    rn = tl.arange(0, N)

    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    # Original approach
    # val = tl.load(x_ptr + idx * stride_x, mask=mask)
    # Modified approach
    rm = tl.arange(0, M)
    x_shared = tl.load(x_ptr + rm * stride_x)  # [M]
    val = tl.gather(x_shared, idx, 0)

    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

- Performance Analysis and Comparison Before and After Optimization

Using the msprof tool to execute the test case yields a PROF_* folder containing the `op_summary_*.csv` file, which helps analyze the pipeline status. Note: "*" represents a timestamp, [Reference method for performance data collection](../debug_guide/profiling.md).

||Op Name|aiv_mte2_time(us)|aiv_mte2_ratio|
|:---- |:--------|:--------|:--------|
|Unoptimized|pick_kernel|0.686|0.008|
|Optimized|pick_kernel|1.041|0.066|

Analyzing the data in the table reveals a significant difference in `aiv_mte2_time(us)` and `aiv_mte2_ratio` before and after optimization. The optimized approach first moves most of the data to UB, reducing the number of times small batches of data are moved from L2 to UB, thus decreasing the total time for L2 to UB data movement.

### Compute and Data Movement Parallelism

Triton-Ascend supports two data processing modes: serial compute and data movement, and parallel compute and data movement.

Serial compute and data movement: Data is first moved from global memory to on-chip memory, computation is completed, and then the next batch of data is moved. This approach has significant idle waiting time and low efficiency.

Parallel compute and data movement: While the first batch of data is being moved to on-chip memory, computation on it begins; subsequently, the second batch of data is moved, forming a pipelined operation where "movement + computation" overlap, significantly improving overall throughput.

The key to achieving parallel compute and data movement lies in designing a reasonable data tiling strategy so that while the current batch of data is being computed, the data required for the next stage can be prepared in advance, thereby parallelizing data movement and computation. Currently, the compiler defaults to `multiBuffer=True`, which enables parallel compute and data movement by default.

### Tiling Optimization

When the AI Core performs computation, data must first be moved to on-chip memory. The on-chip memory space is usually much smaller than the total amount of data the AI Core needs to process. Taking the Atlas 800T/I A2 product as an example, the on-chip memory capacity is 192KB, and with the double buffer feature enabled by default, the capacity is halved. Therefore, operators need to tile the data, loading and processing only a small portion at a time.

- Example

```diff
@libentry()
@triton.autotune(configs=runtime.get_tuned_config("masked_fill"), key=["N"])
@triton.jit
- def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
+ def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
+   base_offset = pid * BLOCK_SIZE

+   # Calculate the total number of blocks to process
+   num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

+   # Loop through each sub-block
+   for sub_block_idx in range(num_sub_blocks):
+       # Calculate the offset for the current sub-block
+       sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
+       offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
-       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        # Load input and mask
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)

        # Write the original input first
        tl.store(out + offsets, input_vals, mask=mask)

        # Overlay and write value at the position that needs to be filled
-       value_to_write = tl.full([BLOCK_SIZE], value, dtype=input_vals.dtype)
+       value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        overwrite_vals = tl.where(fill_mask_vals, value_to_write, tl.load(out + offsets, mask=mask, other=0))
        tl.store(out + offsets, overwrite_vals, mask=mask)
```

### Triton Autotune

In Tiling optimization, the values of tiling parameters like `BLOCK_SIZE`, `BLOCK_SIZE_SUB` directly impact operator performance. Manually debugging parameter combinations is inefficient and finding the optimal values is difficult. `triton.autotune` is an automatic tuning tool provided by the Triton framework. It can iterate through pre-defined parameter configurations, compare performance through actual execution, and automatically select the optimal parameter combination. It is a core supporting tool for Tiling optimization.

If you are interested in the recommended usage of `configs=[]` on Triton-Ascend and the applicable boundaries of automatic Tiling, you can further refer to the [Triton-Ascend autotune usage guide](./autotune_guide.md).

- Core Function
Automatically traverse the parameter space: Batch test the performance of different values for `constexpr` type tiling parameters like `BLOCK_SIZE`, `BLOCK_SIZE_SUB`.
Performance baseline comparison: Use the operator's execution time as the metric to filter the optimal parameters suitable for the current hardware.
Cache tuning results: The optimal configuration after tuning is cached, and subsequent calls to the operator directly reuse it, avoiding repeated tuning.

- Simple Example

    ```diff
    import triton.language as tl

    @triton.autotune(
    configs=[ # List of parameter configurations to test, candidate values should be powers of 2
            triton.Config({'BLOCK_SIZE': 128}),
            triton.Config({'BLOCK_SIZE': 256}),
            triton.Config({'BLOCK_SIZE': 512}),
        ],
        key=['n_elements'], # Tuning dimension: the input dimension on which the parameter value depends
    )
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    ```

- Note: Setting the following environment variable will print the optimal parameter information.

    ```diff
    export TRITON_PRINT_AUTOTUNING=1
    ```

### Advanced: Using max_autotune for Automatic Tuning

For Ascend NPU operators, achieving optimal performance requires tuning not only `BLOCK_SIZE` but also multiple hardware-related