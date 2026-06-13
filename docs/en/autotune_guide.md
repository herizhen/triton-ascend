# Triton-Ascend Autotune Usage Guide

## Document Purpose

This document is intended for users who already know how to write Triton kernels and understand the basic concepts of the community version `triton.autotune`. It focuses on the recommended usage of Triton-Ascend:

- Recommended autotune writing style on Triton-Ascend;
- The meaning of `configs=[]` in the Ascend backend;
- The applicable boundaries of the automatic Tiling mode, and when to fall back to handwritten `triton.Config`.

## Quick Start

On Triton-Ascend, it is recommended to keep the basic writing style of the community version `@triton.autotune`; when you want the system to automatically generate and evaluate candidate configurations, set `configs` to `[]`:

```python
import triton
import triton.language as tl
import triton.backends.ascend.runtime


@triton.autotune(
    configs=[],
    key=["M", "N"],
)
@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ...
```

This means:

- The semantics of `key` remain consistent with the community version, used to determine which input changes will trigger re-selection of the configuration;
- `configs=[]` in Triton-Ascend means "the Ascend backend automatically generates candidate configurations and completes optimization," rather than "no available configurations."

### 1. Enable Ascend's Extension for Autotune First

Only after importing the following line will the Triton-Ascend autotune extension path be activated:

```python
import triton.backends.ascend.runtime
```

Without this step, the community version `triton.autotune` is still used, and `configs=[]` will not trigger Ascend's automatic Tiling generation.

### 2. `@triton.autotune` Must Directly Wrap `@triton.jit`

The order must be as follows:

```python
@triton.autotune(configs=[], key=["M", "N"])
@triton.jit
def kernel(...):
    ...
```

`@triton.autotune` must directly wrap `@triton.jit`, with no other decorators inserted between them. Otherwise, the kernel DSL cannot be parsed, preventing entry into the Triton-Ascend automatic Tiling generation and optimization pipeline.

### 3. The Meaning of `key` is Consistent with the Community Version

The essence of `key` is the cache key for autotune. Any parameter included in `key` will trigger re-autotuning whenever its value changes.

In most cases, `key` contains shape parameters like `M/N/K`, `seq_len`, `hidden_size`, as they often significantly affect the optimal Tiling; however, `key` is not limited to shape parameters. Any parameter whose change affects configuration selection can also be placed in `key`.

### 4. Parameters Intended for Automatic Tuning Should Not Be Fixed in Advance

If you want a `tl.constexpr` to participate in automatic Tiling generation, the following three conditions must be met simultaneously:

- It must be a Tiling parameter, i.e., a parameter that affects the data size or tile size processed by each block (logical core);
- Do not explicitly pass its value at launch time;
- Do not set a default value for it in the kernel definition.

For example, in the following writing style, `BLOCK_M` will participate in automatic tuning:

```python
kernel[grid](
    x,
    y,
    out,
    M,
    N,
)
```

If you explicitly pass it at launch time:

```python
kernel[grid](
    x,
    y,
    out,
    M,
    N,
    BLOCK_M=128,
)
```

Then this parameter is fixed and no longer belongs to the automatic generation scope.

Similarly, if a default value is provided for a tuning parameter in the kernel definition, for example:

```python
@triton.jit
def kernel(
    ...,
    BLOCK_M: tl.constexpr = 128,
):
    ...
```

Then this parameter will not participate in automatic tuning. For parameters intended to be automatically generated and optimized by the framework, they should be kept as `tl.constexpr` that are "not explicitly passed at launch time and have no default value in the kernel definition."

### 5. If Tiling Parameters Affect the Grid, the Grid Must Be Written in Lambda Form

If a Tiling parameter affects the grid size, the grid cannot be written as a fixed value or a static expression that only depends on runtime parameters. It must be written as a `lambda` form that depends on the `meta` parameter. This is consistent with the community autotune requirements.

```python
grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
```

The reason is that when autotune evaluates different candidate configurations, the values of parameters like `BLOCK_M` will change; if the grid does not change with the candidate configuration, it cannot be guaranteed that each candidate configuration is executed with the correct launch method.

## Usage Notes

The benchmark semantics of autotune are consistent with the community version, executing the kernel multiple times. If the kernel has side effects, such as containing atomic operations, inplace writes, or modifying the cumulative state of input/output buffers, it still needs to be handled through the existing hook mechanisms of the community.

## Extensions of Triton-Ascend Compared to Community Autotune

The typical pattern of the community version `triton.autotune` is: users manually provide a set of `triton.Config`, the framework performs benchmarking, and then caches the optimal result.

Triton-Ascend, while keeping this interface habit unchanged, mainly extends the following aspects.

### 1. Supports `configs=[]` for Automatic Candidate Configuration Generation

This is the most core extension. Users do not need to manually write a set of `triton.Config` first; instead, they can leave `configs` empty and let the Ascend backend automatically generate candidate configurations based on the kernel DSL semantics and runtime shape.

### 2. Supports Parallel Compilation of Multiple Configurations

When autotune needs to evaluate multiple candidate configurations, Triton-Ascend will compile these candidate configurations in parallel by default to reduce the first-time tuning latency.

This capability is enabled by default and can be disabled via the environment variable `TRITON_AUTOTUNE_PARALLEL_COMPILE=0`.

### 3. Supports Using Profiler to Collect Kernel Performance

Triton-Ascend supports switching the performance collection method during autotune benchmarking: in addition to the default benchmark mode, a profiler can be used to collect kernel performance data for each candidate config. It only focuses on the on-chip computation time of the kernel, which is more accurate than the default performance collection method for kernels with short execution times, but it adds some overhead. This capability can be enabled via the environment variable `TRITON_BENCH_METHOD='npu'`.

## Scope and Behavior of Automatic Tiling Generation Capability

The first point above explains that Triton-Ascend supports `configs=[]` for automatic candidate configuration generation. The following points elaborate on this capability.

### 1. Automatic Generation Scope Focuses on Tiling Parameters

The focus of Ascend's automatic generation is on the `tl.constexpr` parameters related to Tiling in the kernel, i.e., parameters that affect the data size or tile size processed by each block (logical core).

This capability is not equivalent to "automatically tuning all parameters for you." Compilation parameters like `num_warps`, `num_stages`, and non-Tiling parameters of the kernel are not within the current automatic generation scope.

### 2. Candidate Configurations Will Carry Ascend Hardware Constraints

When generating candidates, the Ascend backend will filter based on the NPU's on-chip storage capacity, alignment constraints, core utilization, etc., rather than simply enumerating a batch of configurations and blindly benchmarking them.

### 3. The Goal of the Automatic Tiling Mode is to Conveniently Provide "Good Enough" Configurations

To balance optimization time and optimization effectiveness, the current automatic Tiling will perform extensive pruning on the number of generated configurations. Therefore, it does not guarantee that the automatically generated results will necessarily reach the performance upper limit of manual extreme tuning.

The goal of this capability is to conveniently provide users with a Tiling configuration with decent performance, while minimizing the user's entry barrier and optimization cost.

### 4. When Automatic Tiling Generation Fails, Users Need to Write `triton.Config` Manually

If the automatic Tiling mode cannot generate any usable candidate configurations, users need to switch to manually writing `triton.Config`. It is also recommended to file an issue for such scenarios to help Triton-Ascend subsequently complete the parsing and automatic generation capabilities.

## Manual `triton.Config` Mode

If the automatic Tiling mode fails to generate configurations, or the generated Tiling performance does not meet expectations, simply revert to the standard community writing style. Triton-Ascend maintains compatibility with this part of the semantics:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}),
    ],
    key=["M", "N"],
)
@triton.jit
def kernel(...):
    ...
```

In this mode:

- Configurations are manually provided by the user;
- The framework is responsible for benchmarking, selecting the optimal configuration, and cache reuse;
- Usage habits are consistent with the community autotune.

## Advanced Usage: Joint Tuning of Automatic Tiling with Other Parameters

The following content is for advanced usage, only to be considered when users want to continue jointly tuning non-Tiling parameters or compilation parameters of the kernel in the automatic Tiling mode.

### 1. Community Autotune: Manual Enumeration of All Tuning Parameters

The standard community writing style involves users manually enumerating all candidate configurations:

```python
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "BLOCK_K": BK,
                "GROUP_SIZE_M": GS,
            },
            num_warps=num_warps,
        )
        for BM in [16, 32, 64]
        for BN in [16, 32, 64]
        for BK in [16, 32, 64]
        for GS in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a, b, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M):
    ...
```

### 2. Triton-Ascend: Manual Enumeration of Tiling Parameters and Ascend Compilation Parameters

In Triton-Ascend, if you want to continue using the manual enumeration mode, you can also include Ascend-side parameters in the manual configuration space:

```python
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "BLOCK_K": BK,
                "GROUP_SIZE_M": GS,
                "multibuffer": MS,
            }
        )
        for BM in [16, 32, 64]
        for BN in [16, 32, 64]
        for BK in [16, 32, 64]
        for GS in [1, 2, 4, 8]
        for MS in [False, True]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a, b, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M):
    ...
```

### 3. Triton-Ascend: Automatic Tiling Generation with Joint Tuning of Other Parameters

If you want Tiling parameters to continue being automatically generated by `configs=[]`, but also want to tune other non-Tiling parameters or compilation parameters simultaneously, you can pass these additional search dimensions via `hints`:

```python
@triton.autotune(
    configs=[],
    key=["M", "N", "K"],
    hints={
        "GROUP_SIZE_M": [1, 2, 4, 8],
        "multibuffer": [False, True],
    },
)
@triton.jit
def matmul_kernel(a, b, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M):
    ...


matmul_kernel[grid](a, b, M, N, K)
```

The meaning of this approach is:

- Tiling-related parameters are still automatically generated by Triton-Ascend;
- Non-Tiling parameters or compilation parameters are explicitly provided by the user via `hints` as candidate sets;
- Autotune evaluates the combined configuration space of both parts.

## Summary

The key extension of Triton-Ascend compared to the community autotune is not changing the user interface, but adding the capability to "automatically generate Tiling candidates and complete optimization" on top of the community interface. For most users, the most recommended usage is:

- Keep the writing style of the community version `@triton.autotune`;
- Set `configs` to `[]`;
- Let the Ascend backend automatically complete candidate generation, filtering, benchmarking, and cache reuse based on the kernel DSL and runtime shape.

If the scenario is not suitable for the automatic Tiling mode, simply revert to manually writing `triton.Config`.