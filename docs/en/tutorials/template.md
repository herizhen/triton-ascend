# Technical Documentation Template Based on XXX Model Deployment Tutorial

<p align="center">
  <a href="Model-Deployment-Tutorial-Template.md"><b>English</b></a> | <a href="Model-Deployment-Tutorial-Template.zh.md"><b>中文</b></a>
</p>

This template is based on deployment tutorials for models such as DeepSeek-V3.2 and Qwen-VL-Dense, aiming to provide a reference for technical documentation writing. Users can follow the template guidelines to systematically complete the construction of relevant technical documents.

## 1 Introduction

**Documentation Writing Requirements:**

- Describe the model's basic architecture, core features, and main application scenarios in one sentence.
- Clearly state what the document intends to do and the goal to be achieved in one sentence.
- Specify the vLLM-Ascend version used in the document and the supported model versions.

**Example 1: Model Introduction**

DeepSeek-V3.2 is a sparse attention model. Its main architecture is similar to DeepSeek-V3.1, but it adopts a sparse attention mechanism, aiming to explore and validate optimization solutions for training and inference efficiency in long-context scenarios.

**Example 2: Document Purpose**

This document will demonstrate the main verification steps for the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy, and performance evaluation.

**Example 3: Version Information**

This document is verified and written based on **vLLM-Ascend v0.13.0**. The current model (XXX) is fully supported in this version, and **v0.13.0 and later versions** can run stably. To use the latest features (such as PD separation, MTP, etc.), it is recommended to use v0.13.0 or above.

## 2 Supported Features

Introduce the features supported by the model, including supported hardware, quantization methods, data parallelism, long sequence features, etc.

**Documentation Writing Requirements:**

- Use a table format to present the support status of models and features.
- Or provide cross-references that can be navigated (recommended).

**Example 1: Feature Support List**

| Model Name | Support Status | Notes | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Caching | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallel | Data Parallel | Prefill-Decode Separation | Segmented ACL Graph Execution | Full Graph ACL Graph Execution | Max Model Length | MLP Weight Prefetch | Documentation |
| ------ | ---------- | ------ | ------ | ---------- | ------ | ------------ | -------------- | ------ | ---------- | ---------- | ---------- | ------------ | ---------- | ---------- | ------------------- |----------- | ----------- | ------------- | ------------- | ---------- |
| DeepSeek V3/3.1 | ✅ | | ✅ | Atlas 800I A2:<br>Minimum card requirement is xx | ✅ | ✅ | ✅ | | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 240k | | [DeepSeek-V3.1](../../tutorials/models/DeepSeek-V3.1.md) |
| DeepSeek V3.2 | ✅ | | ✅ | Atlas 800I A2:<br>Minimum card requirement is xx | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 160k | ✅ | [DeepSeek-V3.2](../../tutorials/models/DeepSeek-V3.2.md)|
| Qwen3 | ✅ | | ✅ | Atlas 800I A2:<br>Minimum card requirement is xx | ✅ | ✅ | ✅ | | | ✅ | ✅ | | | ✅ | | ✅ | ✅ | 128k | ✅ | [Qwen3-Dense](../../tutorials/models/Qwen3-Dense.md) |

>**Note**: This is a simplified example. Please refer to the complete feature matrix for the full table.

**Example 2: References**

Please refer to the [Supported Features List](../user_guide/support_matrix/supported_models.md) for the model's feature matrix.

Please refer to the [Feature Guide](../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weights

**Documentation Writing Requirements:** Describe the hardware resources, software environment, and model files required for deployment.

**Example:**

- `DeepSeek-V3.2-Exp-W8A8` (Quantized version): Requires 1 Atlas 800 A3 (64G × 16) node or 2 Atlas 800 A2 (64G × 8) nodes. [Model Weights](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V3.2-Exp-W8A8)
- `DeepSeek-V3.2-w8a8` (Quantized version): Requires 1 Atlas 800 A3 (64G × 16) node or 2 Atlas 800 A2 (64G × 8) nodes. [Model Weights](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V3.2-W8A8/)

It is recommended to download the model weights to a shared directory across multiple nodes.

### 3.2 Verify Multi-Node Communication (Optional)

**Example:**

If a multi-node environment needs to be deployed, please follow the [Verify Multi-Node Communication Environment](../installation.md#verify-multi-node-communication) guide for communication verification.

## 4 Installation

**Documentation Writing Requirements:**

- Provide specific installation steps and commands (parameters should explain their meaning, value range, units, etc.).
- Version number writing convention: Prioritize using placeholders (values configured uniformly); if a fixed value is used and it differs from the document's verified version, add a comment "Please replace according to the actual version".
- Provide verification commands and expected status: Guide users to check the installation result by executing commands (e.g., docker ps), and describe the success status code or output characteristics.

### 4.1 Docker Image Installation

**Example:** Omitted

### 4.2 Source Code Installation

**Example:** Omitted

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

**Documentation Writing Requirements:**

- Describe the architectural characteristics and applicable scenarios of single-node deployment.
- Provide startup command templates and descriptions of key parameters.
- Provide service verification methods (e.g., curl commands) and expected results, describing success characteristics (e.g., 200 OK).
- Provide common issue guidance below the startup command; if already described in the public FAQ, a direct link can be provided.

**Example:**

Single-node deployment completes Prefill and Decode within the same node, suitable for XXX scenarios.

Startup command:

```bash
# Omitted
```

Common issue tip: If encountering xxx issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

Service verification:

```bash
# Omitted
```

Expected result: Omitted (write according to actual output).

### 5.2 Multi-Node PD Separation Deployment

**Documentation Writing Requirements:**

- Describe the principle and applicable scenarios of the PD separation architecture.
- Provide the startup process, key configurations, and **deployment verification instructions**.
- Specify performance metrics.
- Provide common issue guidance below the startup command; if already described in the public FAQ, a direct link can be provided.

**Example:** Omitted

### 5.3 Special Deployment Forms (Optional)

**Documentation Writing Requirements:**

- If the model has non-standard deployment forms (e.g., offline batch processing for embedding models, low-latency online services for reranker models), the corresponding deployment plan must be clearly reflected in the document.
- Can be extended by referring to sections 5.1 and 5.2 of this chapter.

## 6 Functional Verification

**Documentation Writing Requirements:**

- Guide users on how to test whether the basic functions of the model are normal through simple interface tests after the service starts.
- Provide expected results, describing success characteristics (e.g., HTTP 200, JSON response containing a choices field).

**Example:**

After the service starts, the model can be invoked by sending a prompt:

```shell
       curl http://<node0_ip>:<port>/v1/completions \
           -H "Content-Type: application/json" \
           -d '{
               "model": "deepseek_v3.2",
               "prompt": "The future of AI is",
               "max_tokens": 50,
               "temperature": 0
           }'
```

Expected result: Omitted (write according to actual output).

## 7 Accuracy Evaluation

**Documentation Writing Requirements:** Introduce standardized methods and tools for evaluating model output quality (accuracy). The following provides two accuracy evaluation methods as examples; alternatively, directly link to existing documentation.

### Using AISBench

For details, please refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md).

### Using Language Model Evaluation Harness

Taking the `gsm8k` dataset as the test dataset, run the accuracy evaluation of `DeepSeek-V3.2-W8A8` in online mode.

1. For `lm_eval` installation, please refer to [Using lm_eval](../developer_guide/evaluation/using_lm_eval.md).
2. Run `lm_eval` to perform accuracy evaluation.

```shell
lm_eval \
  --model local-completions \
  --model_args model=/root/.cache/Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

## 8 Performance

Omitted, requirements are the same as for accuracy evaluation.

## 9 Performance Tuning

### 9.1 Recommended Configurations

**Documentation Writing Requirements:**

Provide recommended configurations for the model in three typical scenarios (long sequence, low latency, high throughput). Clearly state that the configurations are not globally optimal and guide users to tune based on actual conditions.

**Example:**

> **Note**: The following configurations are verified based on a specific test environment and are for reference only. The actual optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, accuracy requirements, and deployment machine ratio. It is recommended to tune according to the actual situation by referring to section 9.2.

#### Table 1: Scenario Overview

| Scenario | Deployment Form | *Total Cards | Weight Version | Scenario Key Points |
|------|------|---------|----------|----------|
| High Throughput<br>(32K Push 1K) | 1P1D Deployment | 16 (A3) | glm5.1w4a8 | In short-sequence high-throughput scenarios, try adjusting the xxx parameter |
| Long Sequence |  |  |  |  |
| Low Latency |  |  |  |  |

> `*Total Cards` indicates the total number of NPUs used across all nodes.

#### Table 2: Node Detailed Configuration

| Scenario | Configuration | Cards | TP | DP | BS | Concurrency | Max Context | MTP Speculation Count | FUSED_MC2 | EP Switch | FC+CP Switch | Async Scheduling |
|------|------|------|----|----|----|------|----------|---------|---------------|--------|-------|------|
| High Throughput (32K Push 1K) | Server-P Node/Single Node | 8 | 8 | 2 | 32 | 64 | 30k | 3 | Off | On | On | On |
| High Throughput (32K Push 1K) | Server-D Node | 8 | 2 | 8 | 8 | 64 | 30k | 12 | Off | On | Off | On |
| Long Sequence | Server-P Node/Single Node |  |  |  |  |  |  |  |  |  |  |  |
| Long Sequence | Server-D Node |  |  |  |  |  |  |  |  |  |  |  |
| Low Latency | Server-P Node/Single Node |  |  |  |  |  |  |  |  |  |  |  |
| Low Latency | Server-D Node |  |  |  |  |  |  |  |  |  |  |  |

> For the complete startup command and parameter descriptions, please refer to the deployment examples in Chapter 5.

### 9.2 Tuning Approach

#### 9.2.1 General Tuning Reference

**Documentation Writing Requirements:**

If no special tuning is involved, directly provide links to the feature overlay table and public performance tuning documentation for reference.

**Example:**

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

#### 9.2.2 Model-Specific Optimizations (Optional)

**Documentation Writing Requirements:**

If the model has specific optimizations, summarize the key optimization techniques and parameter tuning experience for that model.

**Example:**

#### Optimizations Enabled by Default

The following optimizations are enabled by default and require no additional configuration:

| Optimization Technique | Technical Principle | Performance Benefit |
| --------- | --------- | --------- |
| Rope Optimization | The cos_sin_cache and index operations of positional encoding are only performed in the first layer; subsequent layers directly reuse them | Reduces repetitive computation during the decode phase, accelerating inference |
| AddRMSNormQuant Fusion | Merges element-wise multi-scale normalization and quantization operations into a single operator | Optimizes memory access patterns, improving computational efficiency |
| Zero-like Elimination | Removes unnecessary zero tensor operations in the Attention forward pass | Reduces memory footprint, improves matrix operation efficiency |
| FullGraph Optimization | Captures and replays the entire decode graph in one go via `compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"}` | Significantly reduces scheduling latency, stabilizes multi-device performance |

#### Optimizations Requiring Explicit Enablement

| Optimization Technique | Applicable Scenario | Enablement Method | Technical Principle | Notes |
| --------- | --------- | --------- | --------- | --------- |
| FlashComm_v1 | High concurrency, Tensor Parallel (TP) scenarios | `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` | Decomposes traditional Allreduce into Reduce-Scatter and All-Gather, reducing the RMSNorm computation dimension | Threshold protection: Only takes effect when the actual token count exceeds the threshold, preventing performance regression in low-concurrency scenarios|
| Matmul-ReduceScatter Fusion | Large-scale distributed environments | Automatically enabled after enabling FlashComm_v1 | Fuses matrix multiplication with Reduce-Scatter operations, enabling pipeline parallel processing | Same as FlashComm_v1, has threshold protection |
| Weight Prefetch | MLP-intensive scenarios (Dense models)| `export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1` | Uses vector computation time to preload MLP weights into L2 Cache | Requires adjustment of the prefetch buffer size |
| Async Scheduling | Large-scale models, high-concurrency scenarios | `--async-scheduling` | Non-blocking task scheduling, improving concurrent processing capability | Used in conjunction with FullGraph optimization |

## 10 FAQ

**Documentation Writing Requirements:**

- Add a note at the beginning of the chapter: For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html); this chapter only includes specific issues unique to this model.
- For **specific issues unique to this model**, provide the following elements: Problem description, root cause analysis, and solution.