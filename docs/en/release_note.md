# Triton-Ascend Release

The Triton-Ascend release provides a stable codebase snapshot, packaged as binary wheels that can be easily installed via PyPI. Additionally, releases allow the development team to formally announce to the community the availability of new features, completed improvements, and changes that may affect users (e.g., breaking changes).

## Release Compatibility Matrix

The following is the release compatibility matrix for Triton-Ascend:

| Triton-Ascend Version | Python Version | Manylinux Version | Hardware Platform | Hardware Product |
| --- | --- | --- | --- | --- |
| 3.2.0 | >=3.9, <=3.11 | glibc 2.27+, x86-64, aarch64 | Ascend NPU | Atlas A2/A3 |

## Release Schedule

The following is the release schedule for Triton-Ascend. Note: Patch versions are optional.

| Major Version | Release Branch Cut Date | Release Date | Patch Release Date |
| --- | --- | --- | --- |
| 3.2.0 | December 08, 2025 | January 2026 | --- |

## Release Highlights

### Triton-Ascend 3.2.0

**First Release: Ascend NPU Support**

Triton-Ascend 3.2.0 is the first Triton release to officially support Huawei Ascend NPUs. This version is based on the Triton 3.2.0 community release and is specifically adapted for the Ascend NPU hardware architecture.

#### Key Features

1. **Full-Stack Ascend NPU Support**
   - Complete compilation pipeline from Triton IR to NPU instruction set
   - Support for all Triton Ops

2. **Performance Optimizations**
   - NPU-specific kernel optimizations
   - CV computation optimizations

3. **Developer Tools**
   - Comprehensive debug output support
   - Compilation intermediate product dump

#### Known Limitations

1. **Data Types**: Support for some data types is still under development
2. **Operator Coverage**: The supported operator set is continuously being expanded

#### Migration Guide

For existing Triton GPU users migrating to Ascend NPU, see [GPU Triton Operator Migration](./migration_guide/migrate_from_gpu.md)