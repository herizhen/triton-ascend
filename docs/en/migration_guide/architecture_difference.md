# Differences Between Ascend and GPU Development

## Multi-Core Task Parallelism Strategy

NPU uses a physical core binding mode in Triton multi-core parallelism, which fundamentally differs from GPU's logical dimension parallelism with automatic hardware physical mapping.

- Core Comparison

    |Dimension       |GPU (NVIDIA) |Ascend (NPU)|
    |-----------|--------------|-----------|
    |Grid Essence| Logical task dimension (decoupled from physical cores)| Physical core group mapping (bound to AI Core topology)|
    |Core Count / Dimension Limit| No hard limit on grid dimensions/size| Grid size ≤ total AI Core count, 2D must match topology|

GPU: Can bind multiple dimension axes (3D grid=[n,m,l] equals n×m×l parallel threads), each thread corresponds to one kernel execution, executed only once.\
NPU: Vector cores and Cube cores belong to multiple physical cores, with different core counts across hardware generations. Each core executes one Block only once and supports repeated scheduling of that Block.

### Fully Utilizing Core Count

Ascend NPU has multiple compute cores. Properly allocating and fully utilizing all available cores is a key factor in improving operator performance.
When calling Triton kernel functions, control the number of cores used by setting launch parameters. Using the GELU operator as an example:

```Python
triton_gelu[n, 1, 1](...)  # First parameter indicates number of cores used, n means using n cores
```

By tuning the core count, you can fully schedule and utilize all compute resources, maximizing parallelism and throughput. Note: In the current version, the core count must be ≤ 65535.

## Single-Core Data Movement Strategy

### Data Tiling

When writing Triton kernel functions, a proper data tiling strategy is crucial for performance optimization. By adjusting different tiling granularity parameters, you can balance compute load and memory access efficiency across different dimensions.

Common tiling parameters include:

```text
ncore: Number of cores used (cross-core tiling)
xblock: Inter-core data block size (inter-core tiling)
xblock_sub: Intra-core tiling granularity (fine-grained intra-core partitioning)
```

Developers can manually select the optimal tiling configuration based on the actual scenario, ensuring each computation fully utilizes on-chip memory and avoids performance bottlenecks caused by frequent global memory access.

Using the GELU operator as an example, adjusting tiling parameters can effectively adapt to on-chip cache capacity limits, thereby improving execution efficiency.

Note: The on-chip memory capacity of the Atlas 800T/I A2 product is 192KB. Therefore, this limit must be considered when designing the tiling strategy to ensure that the data volume per computation round does not exceed the on-chip memory capacity.

#### GELU Operator Example

GELU operator development example, computing results using 3 methods.

standard_unary: Standard Torch computation.

triton_easy_kernel: Simple Triton implementation.

triton_better_kernel: More efficient Triton implementation.

#### Standard Torch Implementation

Input tensor x0, compute the GELU operator using Torch, return the result.

```Python
def standard_unary(x0):
    res = x0 * 0.5 * (1.0 + torch.erf(x0 / torch.sqrt(torch.tensor(2.0))))
    return res
```

#### Simple Triton Implementation

The following is a simple kernel example written in Triton, demonstrating how to define and call a basic Triton kernel function. This example implements a simple mathematical operation (GELU activation function).

```Python
# Define triton_kernel function
@triton.jit
def triton_easy_kernel(in_ptr0, out_ptr0, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block)
    ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    tl.store(out_ptr0 + idx_block, ret)
```

Notes

1. Memory Limitation: In the above implementation, all input data is loaded into memory at once for computation. If the input tensor is too large, it may exceed the on-chip memory capacity of a single core, causing memory overflow errors. Therefore, this simple approach is more suitable for small tensor computations or for understanding the basic syntax and calling methods of Triton kernels.

2. Applicable Scenarios: Although this method helps quickly understand and get started with Triton programming, for large-scale datasets or high-performance applications, it is recommended to adopt more complex data tiling strategies (such as Tiling) to fully utilize hardware resources and avoid memory overflow issues. Through this approach, developers can quickly get started with Triton programming while learning how to define, call, and optimize Triton kernel functions.

#### More Efficient Triton Implementation

When writing high-performance operators using Triton on Ascend NPU, to fully utilize hardware resources, avoid memory overflow, and improve execution efficiency, a data tiling strategy is typically required. Below is an optimized Triton kernel implementation example suitable for large tensor computations.

```Python
# Define triton_kernel function
@triton.jit
def triton_better_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        x = tl.load(in_ptr0 + x_index, xmask)
        ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        tl.store(out_ptr0 + x_index, ret, xmask)

# Call triton_kernel function
ncore = 32
xblock = 32768
xblock_sub = 8192
triton_better_kernel[ncore, 1, 1](x0, out1, x0.numel(), xblock, xblock_sub)
```

Key Code Explanation

```Python
# Calculate the starting offset address of the data block processed by the current core, implementing inter-core tiling. Each core is responsible for a data range of XBLOCK size.
xoffset = tl.program_id(0) * XBLOCK

# Further subdivide the data block within a single core, processing XBLOCK_SUB size data each time, implementing intra-core tiling.
for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):

# Construct the data index array for the current iteration, used to access input and output tensors.
x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]

# Set a mask to prevent out-of-bounds access, ensuring only valid data ranges are processed.
xmask = x_index < xnumel

# Used to load data from global memory to on-chip memory and write computation results back to global memory, respectively.
tl.load() and tl.store()
```

## Compilation Optimization Capabilities

### AscendNPU IR Optimization

Compilation options adapted for AscendNPU IR optimization based on Ascend hardware and software characteristics are shown in the table below.
**Usage**: Pass the compilation option values during the autotune configuration phase.
Taking enabling the `multibuffer` option as an example, pass `'multibuffer': True` in the autotune configuration phase, i.e., `triton.Config`. See [autotune example](../examples/06_autotune_example.md) for details:

```python
    def get_autotune_config():
        return [
            triton.Config({'XS': 1 * 128, 'multibuffer': True}),]
```

| Option                                      | Capability                                                                 | Enabled |
| ----------------- | ------------ | ----------------- |
| multibuffer                                   | Enable pipeline parallel data movement                                    | Default true; true, false. Configurable in autotune                     |
| unit_flag                                     | Optimization item for cube data movement                                  | Default None; true, false. Configurable in autotune                     |
| limit_auto_multi_buffer_only_for_local_buffer | Optimization item for CV operators, cube data movement optimization       | Default None; true, false. Configurable in autotune |
| limit_auto_multi_buffer_of_local_buffer       | Specific scope for enabling double buffer in cube operators               | Default None; ["no-limit", "no-l0c"], configurable in autotune           |
| set_workspace_multibuffer                     | Only effective when limit_auto_multi_buffer_only_for_local_buffer=false   | Default None; e.g., [2,4], configurable in autotune                            |
| enable_hivm_auto_cv_balance                   | set_workspace_multibuffer only effective when limit_auto_multi_buffer_only_for_local_buffer=false | Default None; true, false. Configurable in autotune |
| tile_mix_vector_loop                          | Optimization item for CV operators, current vector split count            | Default None; e.g., [2,4,8], configurable in autotune                       |
| tile_mix_cube_loop                            | Optimization item for CV operators, current cube split count              | Default None; e.g., [2,4,8], configurable in autotune                      |
| auto_blockify_size                            | TRITON_ALL_BLOCKS_PARALLEL optimization item, specifies the size of the expanded leftmost first dimension | Default 1; e.g., [2,4,8], configurable in autotune                       |

- Note: Optimization compilation options are in the ascend/backend/compiler.py code.
- Note: CV operators refer to operators that use both AI Core and Vector Core during computation.