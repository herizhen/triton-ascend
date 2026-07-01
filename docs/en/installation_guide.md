# Installation Guide

Choose the appropriate installation method based on your needs and jump to the corresponding steps:
- **Quick Setup with Docker Image**: Use the ready-to-use image published by Triton-Ascend to quickly build a development environment. Follow the instructions in [OVERVIEW.zh.md](../../docker/OVERVIEW.zh.md).
- **Install via pip**: Choose this option to directly try the TA pip package. Proceed to the next step <a href="#env-prepare">Environment Preparation</a> to complete the prerequisites, then perform the pip installation.
- **Install from Source**: Choose this option if you are a TA developer. Proceed to the next step <a href="#env-prepare">Environment Preparation</a> to complete the prerequisites, then choose either <a href="#auto-code-base">Quick Installation</a> or <a href="#hand-code-base">Manual Installation</a>.
- **Install via Dockerfile**: No environment preparation is needed; you can jump directly to <a href="#docker-build">Install via Dockerfile</a>.

## Quick Installation with Docker Image

### Verify Device Model

| Chip Series | Product Example                  | Corresponding Tag                |
|-------------|----------------------------------|----------------------------------|
| Ascend 910b | Atlas 800T A2, Atlas 900 A2 PoD  | 3.2.1-910b-ubuntu22.04-py3.11    |
| Ascend A3   | Atlas 800T A3                    | 3.2.1-a3-ubuntu22.04-py3.11      |
| Ascend 950  | 950PR Series                     | 3.2.1-950-ubuntu22.04-py3.11     |

Note: For more images, see [OVERVIEW.zh.md](../../docker/OVERVIEW.zh.md).

### Pull the Image

```bash
docker pull quay.io/ascend/{image_tag}
```

### Create a Container

```bash
# Assuming your NPU device model is A3, the device is installed on /dev/davinci1, and your NPU driver is installed at /usr/local/Ascend:
container_name=triton-ascend_container
image_tag=quay.io/ascend/triton:3.2.1-a3-ubuntu22.04-py3.11
docker run -u 0 -dit --shm-size=512g --name=${container_name} --net=host --privileged \
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
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /home:/home \
${image_tag} \
/bin/bash
```

### Enter the Container

```bash
docker exec -it triton-ascend_container bash
```

Run example: [01-vector-add.py](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/01-vector-add.py)

Observe output similar to the following to confirm the environment is set up.

```
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    The maximum difference between torch and triton is 0.0
```

## Other Three Setup Methods

<a id="env-prepare"></a>

### Environment Preparation

#### Python Version Requirements

| Triton-Ascend Version | Supported Python Versions | Notes             |
|-----------------------|---------------------------|-------------------|
| 3.2.1                 | py3.9 - py3.13            | py3.9 does not support aarch64 |
| 3.2.0                 | py3.9 - py3.11            |                   |
| 3.2.0rc4              | py3.9 - py3.11            |                   |

#### Install CANN

The heterogeneous computing architecture CANN (Compute Architecture for Neural Networks) is a heterogeneous computing architecture launched by Ascend for AI scenarios.
It supports multiple AI frameworks upward, including MindSpore, PyTorch, TensorFlow, etc., and serves AI processors and programming downward, playing a key role in improving the computing efficiency of Ascend AI processors.

You can visit the Ascend community official website and follow the [community software installation guide](https://www.hiascend.com/cann/download) to complete the installation and configuration of CANN. Developers can find the corresponding installation command by selecting the CANN version, product series, CPU architecture, operating system, and installation method.

During installation, please select one of the following versions for CANN version "**{version}**". It is recommended to download and install version 8.5.0:

- Note: If the user does not specify an installation path, the software will be installed to the default path. The default installation paths are as follows. Root user: `/usr/local/Ascend`, non-root user: `${HOME}/Ascend`, where `${HOME}` is the current user's home directory.
The above environment variable configuration only takes effect in the current window. Users can write the ```source ${HOME}/Ascend/ascend-toolkit/set_env.sh``` command into the environment variable configuration file (e.g., .bashrc file) as needed.

**CANN Versions:**

- Commercial Version

| Triton-Ascend Version | CANN Commercial Version | CANN Release Date |
|-----------------------|-------------------------|-------------------|
| 3.2.1                 | CANN 9.0.0              | 2026/04/30        |
| 3.2.0                 | CANN 8.5.0              | 2026/01/16        |
| 3.2.0rc4              | CANN 8.3.RC2<br>CANN 8.3.RC1 | 2025/11/20<br>2025/10/30 |

- Community Version

| Triton-Ascend Version | CANN Community Version | CANN Release Date |
|-----------------------|------------------------|-------------------|
| 3.2.1                 | CANN 9.0.0             | 2026/04/30        |
| 3.2.0                 | CANN 8.5.0             | 2026/01/16        |
| 3.2.0rc4              | CANN 8.3.RC2<br>CANN 8.5.0.alpha001<br>CANN 8.3.RC1 | 2025/11/20<br>2025/11/12<br>2025/10/30 |

#### Install torch_npu

The currently supported torch_npu version is 2.7.1.post4.

```bash
pip install torch_npu==2.7.1.post4
```

Note: If you encounter the error `ERROR: No matching distribution found for torch==2.7.1+cpu`, you can try manually installing torch before installing torch_npu.

```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

<a id="code-require"></a>

### Install Triton-Ascend via pip

#### Latest Stable Version

You can install the latest stable version of Triton-Ascend via pip.

```shell
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

- Note: For triton-ascend 3.2.0 and below, Triton-Ascend and Triton cannot coexist. You need to uninstall the community Triton first, then install Triton-Ascend.<br>
For triton-ascend 3.2.1 and above, Triton-Ascend mitigates the installation override issue by declaring Triton as an installation dependency.
When installing Triton-Ascend, the community Triton will be installed first, and then Triton-Ascend will overwrite the same-named directory, thus preventing other packages that depend on Triton from reinstalling Triton and overwriting Triton-Ascend later.
The reason x86 and arm use different versions of the community Triton installation package is that the community only started providing arm version installation packages from version 3.5 onwards: x86 depends on triton==3.2.0, arm depends on triton==3.5.0.

```shell
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

#### Historical Stable Versions

```shell
pip install triton-ascend==3.2.0
```

### Install Triton-Ascend from Source

If you need to develop or customize Triton-Ascend, you should use the source code compilation and installation method. This approach allows you to modify the source code according to project requirements and compile and install a customized version of Triton-Ascend.

Before building, you need to complete the <a href="#code-require">dependency installation</a> for the relevant build components.

We recommend using the <a href="#auto-code-base">Quick Installation</a> method to install Triton-Ascend from source; if you have special requirements, such as the target machine not being able to connect to the internet, you can perform a <a href="#hand-code-base">Manual Installation</a>.

#### System Recommendations

| PyTorch Version | Recommended GCC Version | Recommended GLIBC Version |
|-----------------|-------------------------|----------------------------|
| PyTorch 2.7.1   | 11.2.1                  | 2.28                       |
| PyTorch 2.8.0   | 13.3.1                  | 2.28                       |
| PyTorch 2.9.1   | 13.3.1                  | 2.28                       |
| PyTorch 2.10    | 13.3.1                  | 2.28                       |

<a id="code-require"></a>

#### Dependencies

##### Install System Library Dependencies

Install zlib1g-dev/lld/clang, optionally install the ccache package to speed up the build.

- Recommended version clang >= 15
- Recommended version lld >= 15

```bash
Using Ubuntu as an example:
sudo apt update
sudo apt install zlib1g-dev clang-15 lld-15
sudo apt install ccache # optional
```

Triton-Ascend's build strongly depends on zlib1g-dev. If you use a yum source, please refer to the following command to install:

```bash
sudo yum install -y zlib-devel
```

##### Install Python Dependencies

```bash
pip install ninja cmake wheel pybind11 # build-time dependencies
```

<a id="auto-code-base"></a>

#### Quick Installation

```bash
git clone https://github.com/triton-lang/triton-ascend.git
cd triton-ascend
git checkout main

# Optional: If you have a pre-compiled LLVM locally, you can directly specify the local LLVM path to avoid downloading the LLVM pre-compiled package. If not, ignore this and directly execute the installation command below.
export LLVM_SYSPATH=/path/to/LLVM

# Execute the installation command
pip install -e python
```

<a id="hand-code-base"></a>

#### Manual Installation - Build Based on LLVM

Triton uses LLVM 22 to generate code for GPU and CPU. Similarly, Ascend's Bisheng Compiler also relies on LLVM to generate NPU code, so you need to compile the LLVM source code to use it. Please pay attention to the specific version of LLVM that is depended upon. LLVM build supports two build methods. **Choose one of the following two methods; no need to execute both.**

##### Code Preparation: `git checkout` to Checkout the Specified LLVM Version

   ```bash
   git clone --no-checkout https://github.com/llvm/llvm-project.git
   cd llvm-project
   git checkout fad3272286528b8a491085183434c5ad4b59ab92
   wget https://raw.githubusercontent.com/triton-lang/triton-ascend/6765b03c81c4e9ecb277e4ef1dde61dea0d044f0/third_party/ascend/llvm_patch/fad3272.patch
   git apply fad3272.patch
   ```

##### Build and Install LLVM with clang

- Step 1: Install LLVM using clang. Please install clang and lld on the environment and specify the version (recommended version clang>=15, lld>=15).
  If not installed, install clang, lld, and ccache using the following commands:

  ```bash
  apt-get install -y clang-15 lld-15 ccache
  ```

- Step 2: Set the environment variable LLVM_INSTALL_PREFIX to your target installation path:

   ```bash
   export LLVM_INSTALL_PREFIX=/path/to/llvm-install
   ```

- Step 3: Execute the following commands to build and install LLVM:

  ```bash
  cd {PATH_TO}/llvm_project # Path to the directory where LLVM code was pulled; adjust as needed
  mkdir build
  cd build
  cmake ../llvm \
    -G Ninja \
    -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15 \
    -DCMAKE_LINKER=/usr/bin/lld-15 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
    -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
  ninja install
  ```

- Step 4: Copy FILECHECK to the target installation path:

   ```bash
   cp  {PATH_TO}/llvm_project/build/bin/FileCheck ${LLVM_INSTALL_PREFIX}/bin/FileCheck
   ```

##### Clone Triton-Ascend

```bash
git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
```

##### Build Triton-Ascend

- Step 1: Ensure that the target installation path ${LLVM_INSTALL_PREFIX} for LLVM from the [Build Based on LLVM] section is set.
- Step 2: Ensure that clang>=15, lld>=15, and ccache are installed.

   ```bash
   LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
   TRITON_BUILD_WITH_CCACHE=true \
   TRITON_BUILD_WITH_CLANG_LLD=true \
   TRITON_BUILD_PROTON=OFF \
   TRITON_WHEEL_NAME="triton-ascend" \
   TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
   python3 setup.py install
   ```

Note 1: The recommended GCC version is in the "System Recommendations" section above. If GCC < 9.4.0, you may encounter the error "ld.lld: error: unable to find library -lstdc++fs", indicating that the linker cannot find the stdc++fs library.
This library is used to support filesystem features in GCC versions prior to 9. In this case, you need to manually uncomment the relevant code snippet in the CMake file:

triton-ascend/CMakeLists.txt

   ```bash
   if (NOT WIN32 AND NOT APPLE)
   link_libraries(stdc++fs)
   endif()
   ```

   Uncommenting this and rebuilding the project should resolve the issue.

<a id="docker-build"></a>

### Install via Dockerfile

We provide a Dockerfile to help you install the Docker environment image. The build process uses the `quay.io/ascend/cann` pre-built image as the base image, skipping the CANN installation step and significantly speeding up the build.

You need to specify the `CANN_BASE_IMAGE` parameter via `--build-arg` to select the CANN base image suitable for your machine. Available CANN base image tags can be viewed at [quay.io/ascend/cann](https://quay.io/repository/ascend/cann?tab=tags).

| CANN Version | Chip Type | Python Version | Image Tag |
|---|---|---|---|
| 8.5.0 | `A2` | 3.10 | `8.5.0-910b-ubuntu22.04-py3.10` |
| 8.5.0 | `A3` | 3.10 | `8.5.0-a3-ubuntu22.04-py3.10` |
| 8.5.0 | `A2` | 3.11 | `8.5.0-910b-ubuntu22.04-py3.11` |
| 8.5.0