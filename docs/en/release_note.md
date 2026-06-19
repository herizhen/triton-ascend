# Triton-Ascend Release

The Triton-Ascend release provides a stable code base snapshot, packaged as binary packages that can be easily installed via PyPI. Additionally, the release marks the development team's formal announcement to the community regarding the availability of new features, completed improvements, and changes that may affect users (e.g., breaking changes).

## Release Compatibility Matrix

Below is the release compatibility matrix for Triton-Ascend versions:

| Triton-Ascend Version | Python Version | Manylinux Version | Hardware Platform | Hardware Product |
| --- | --- | --- | --- | --- |
| 3.2.0 | >=3.9, <=3.11 | glibc 2.27+, x86-64, aarch64 | Ascend NPU | Atlas A2/A3 |

## Release Schedule

Below is the release schedule for Triton-Ascend. Note: Patch versions are optional.

| Major Version | Release Branch Cut Date | Release Date | Patch Release Date |
| --- | --- | --- | --- |
| 3.2.0 | December 08, 2025 | January 2026 | --- |

## Release Highlights

### Triton-Ascend 3.2.0

**First Release: Ascend NPU Support**

Triton-Ascend 3.2.0 is the first Triton version to officially support Huawei Ascend NPU. This release is based on the Triton 3.2.0 community version and is specifically adapted for the Ascend NPU hardware architecture.

#### Key Features

1. **Full-Stack Ascend NPU Support**
   - Complete compilation pipeline from Triton IR to NPU instruction set
   - Support for all Triton Ops

2. **Performance Optimization**
   - NPU-specific kernel optimizations
   - CV computation optimizations

3. **Developer Tools**
   - Comprehensive debug output support
   - Compilation intermediate product dump

#### Known Limitations

1. **Data Types**: Support for some data types is still under development
2. **Operator Coverage**: The supported operator set is being continuously expanded

#### Migration Guide

For existing Triton GPU users migrating to Ascend NPU, see [GPU Triton Operator Migration](./migration_guide/migrate_from_gpu.md)