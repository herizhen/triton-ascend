# Quick Start

## Project Introduction

Triton-Ascend is an optimized version of Triton adapted for Huawei Ascend chips, providing efficient kernel function auto-tuning, operator compilation, and deployment capabilities. It supports Ascend Atlas A2/A3 series products, is compatible with Triton core syntax, and has been deeply optimized for Ascend NPU features, including automatic parsing of kernel function parameters, optimized memory access logic, and improved secure deployment mechanisms.

## Online Documentation

We provide comprehensive online documentation and resources covering environment setup, operator development, tuning practices, and frequently asked questions to facilitate quick onboarding and in-depth usage. For details, please refer to the [Online Documentation](https://triton-ascend.readthedocs.io/zh-cn/latest/index.html)

## Environment Requirements

### Hardware Requirements

Supported Operating Systems: linux(aarch64/x86_64)

Supported Ascend Products: Atlas A2/A3 series

Minimum Hardware Configuration: Single card 32GB memory (recommended)

### Software Dependencies

Python (**3.9 - 3.13**, note: py3.9 does not support aarch64 architecture), CANN_TOOLKIT, CANN_OPS, and [requirements.txt](../../requirements.txt) and [requirements_dev.txt](../../requirements_dev.txt), etc.

You can visit the Ascend community official website and follow the [Community Software Installation Guide](https://www.hiascend.com/cann/download) to complete the installation and configuration of CANN. Developers can find the corresponding installation commands by selecting the CANN version, product series, CPU architecture, operating system, and installation method.

During installation, please select one of the following versions for CANN version "**{version}**". It is recommended to download and install version 9.0.0:

- Note: If the user does not specify an installation path, the software will be installed to the default path. The default installation paths are as follows. For root user: `/usr/local/Ascend`, for non-root user: `${HOME}/Ascend`, where `${HOME}` is the current user's home directory.

The above environment variable configuration only takes effect in the current window. Users can write the ```source ${HOME}/Ascend/ascend-toolkit/set_env.sh``` command into the environment variable configuration file (e.g., .bashrc file) as needed.

You need to select the corresponding compatible version (version 9.0.0 is recommended) based on the actual Ascend card model you are using. The installation of CANN takes approximately 5-10 minutes, please wait patiently for the installation to complete.

The installation of requirements can be done as follows:

```shell
pip install -r requirements.txt -r requirements_dev.txt
```

## Environment Setup

Users can follow the steps in the environment preparation section of the [Installation Guide](installation_guide.md) to set up the Triton-Ascend environment.

### Obtaining the Triton-Ascend Software Package

Users can directly install the latest stable version package via the command line.

```shell
pip install triton-ascend
```

- Note: Starting from version 3.2.1, Triton-Ascend alleviates the installation override issue by declaring Triton as an installation dependency. When installing Triton-Ascend, the community Triton is installed first, and then Triton-Ascend overrides the directory with the same name, thereby preventing subsequent installation of other software packages that depend on Triton from reinstalling Triton and overriding Triton-Ascend. The reason x86 and arm use different versions of the community Triton installation package is that the community only started providing arm version installation packages from version 3.5 onwards: x86 depends on `triton==3.2.0`, arm depends on `triton==3.5.0`.
- Note 1: This solution is used to alleviate the installation override issue and cannot completely eliminate the conflict caused by the community Triton and Triton-Ascend sharing the same `triton` package directory; if the subsequent installation process explicitly reinstalls or upgrades the community Triton, it may still affect the installed Triton-Ascend. In this case, please uninstall both the community Triton and Triton-Ascend first, and then reinstall Triton-Ascend.

You can also choose to download nightly packages from the [Download Page](https://test.pypi.org/project/triton-ascend/#history) and install them locally.

- Note 2: If you choose to download and install the nightly package yourself, please select the Triton-Ascend package that corresponds to your server's Python version and architecture (aarch64/x86_64).
- Note 3: Nightly packages are built daily. Developers submit MRs frequently, and these packages have not undergone stable testing. There may be functional bugs. Please be aware.

## Quick Setup Using Docker

We provide a Dockerfile to help you build a Docker environment image. The build process uses the `quay.io/ascend/cann` pre-built image as the base image, skipping the CANN installation step, significantly speeding up the build process.

You need to specify the `CANN_BASE_IMAGE` parameter via `--build-arg` to select the CANN base image suitable for your machine. Available CANN base image tags can be viewed at [quay.io/ascend/cann](https://quay.io/repository/ascend/cann?tab=tags).

| CANN Version | Chip Type | Python Version | Image Tag |
|---|---|---|---|
| 8.5.0 | `A2` | 3.10 | `8.5.0-910b-ubuntu22.04-py3.10` |
| 8.5.0 | `A3` | 3.10 | `8.5.0-a3-ubuntu22.04-py3.10` |
| 8.5.0 | `A2` | 3.11 | `8.5.0-910b-ubuntu22.04-py3.11` |
| 8.5.0 | `A3` | 3.11 | `8.5.0-a3-ubuntu22.04-py3.11` |
| 9.0.0-beta.2 | `A2` | 3.10 | `9.0.0-beta.2-910b-ubuntu22.04-py3.10` |
| 9.0.0-beta.2 | `A3` | 3.10 | `9.0.0-beta.2-a3-ubuntu22.04-py3.10` |
| 9.0.0-beta.2 | `A2` | 3.11 | `9.0.0-beta.2-910b-ubuntu22.04-py3.11` |
| 9.0.0-beta.2 | `A3` | 3.11 | `9.0.0-beta.2-a3-ubuntu22.04-py3.11` |

You can use the `npu-smi` command to view the NPU model on your system.

Machines corresponding to different chip types can be referenced as follows:

| Option No. | **Chip Type** | Corresponding Machine/Product Series | Typical Complete Machine |
| :---: |:-----------------:| :---: |:-----------------------------------:|
| 1 | `A3` | Atlas A3 Training Series Products | Atlas 900 A3 SuperPoD |
| 2 | `A2` | Atlas A2 Training Series Products | Atlas800T A2 |

```bash
git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
docker build \
--build-arg CANN_BASE_IMAGE=quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.10 \
-t triton-ascend-image:latest -f ./docker/Dockerfile .
```

To start a container from this image, you can refer to the following command:

```bash
docker run -u 0 -dit --shm-size=512g --name=triton-ascend_container --net=host --privileged \
--security-opt seccomp=unconfined \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /home:/home \
-v /etc/ascend_install.info:/etc/ascend_install.info \
triton-ascend-image:latest \
/bin/bash

# Enter the container
docker exec -u root -it triton-ascend_container /bin/bash
```

## Running Triton Examples

Run example: [01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py)

```bash
# Set CANN environment variables (taking the root user's default installation path `/usr/local/Ascend` as an example)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Clone the triton-ascend source repository and examples (optional, required when running examples without source compilation installation)
git clone https://github.com/triton-lang/triton-ascend.git
# Run the tutorials example:
python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
```

Observing output similar to the following indicates that the environment is correctly configured.

```shell
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

## Migrating Triton Examples from GPU to NPU

Triton-Ascend maintains full compatibility with the community Triton syntax. By simply replacing the **tensor device declaration** and a few `torch.cuda.*` interfaces, existing GPU examples can run on the Ascend NPU. The following demonstrates the complete migration process using a typical vector addition test.

The GPU version example file `test_add.py` is as follows:

```python
import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


@pytest.mark.parametrize('SIZE,BLOCK_SIZE', [(98432, 1024)])
def test_add(SIZE, BLOCK_SIZE):
    device_id = torch.cuda.current_device()
    device = torch.device('cuda', device_id)

    x = torch.randn(SIZE, device='cuda', dtype=torch.float32)
    y = torch.randn(SIZE, device='cuda', dtype=torch.float32)

    output_cpu = torch.empty(SIZE, dtype=torch.float32)
    output = output_cpu.cuda()

    def grid(meta):
        return (triton.cdiv(SIZE, meta['BLOCK_SIZE']),)

    add_kernel[grid](x, y, output, SIZE, BLOCK_SIZE=BLOCK_SIZE)

    torch.cuda.synchronize()

    output_torch = x + y
    assert_close(output, output_torch, rtol=1e-3, atol=1e-3)
```

Migration only requires replacing GPU-related APIs with their corresponding NPU versions. The mapping is as follows:

| GPU Usage | NPU Usage |
| ------------------------------- | ------------------------------- |
| `device='cuda'` | `device='npu'` |
| `tensor.cuda()` | `tensor.npu()` |
| `torch.cuda.current_device()` | `torch.npu.current_device()` |
| `torch.cuda.synchronize()` | `torch.npu.synchronize()` |

Kernel functions annotated with `@triton.jit` use the Triton general language and generally do not require special modifications. The Launch grid calling method is also identical to the GPU version.

The core changes are shown in diff format:

```diff
import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

# ...（kernel code remains unchanged）...

@pytest.mark.parametrize('SIZE,BLOCK_SIZE', [(98432, 1024)])
def test_add(SIZE, BLOCK_SIZE):
-   device_id = torch.cuda.current_device()
+   device_id = torch.npu.current_device()

-   x = torch.randn(SIZE, device='cuda', dtype=torch.float32)
-   y = torch.randn(SIZE, device='cuda', dtype=torch.float32)
+   x = torch.randn(SIZE, device='npu', dtype=torch.float32)
+   y = torch.randn(SIZE, device='npu', dtype=torch.float32)

    output_cpu = torch.empty(SIZE, dtype=torch.float32)
-   output = output_cpu.cuda()
+   output = output_cpu.npu()

    add_kernel[grid](x, y, output, SIZE, BLOCK_SIZE=BLOCK_SIZE)

-   torch.cuda.synchronize()
+   torch.npu.synchronize()

    output_torch = x + y
    assert_close(output, output_torch, rtol=1e-3, atol=1e-3)
```
After modification, you can run the test case using `pytest`. Successful execution indicates a successful migration.
```
pytest test_add.py
```
If the `pytest` component is not installed, you can install it using `pip install pytest`.