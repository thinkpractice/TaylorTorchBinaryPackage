# TaylorTorch Binary Artifact Bundle Builder

This repo builds a SwiftPM-ready binary package that wraps libtorch with a tiny ATen shim suitable for Swift import. It stages an `.artifactbundle` containing:
- `libATenCXX` (static shim)
- ATen shim headers/modulemap
- The libtorch runtime libraries for the selected platform/compute backend

## Layout
- `ATenCXX/` headers and shim source
- `CMakeLists.txt` build and bundling logic (creates `ATenCXX.artifactbundle` + zip)
- `cmake/info.json.in` template for the artifact manifest
- `get-pre-build-torch.sh` helper to download a matching libtorch
- `.github/workflows/build-artifactbundles.yml` CI to build zips and checksums per platform

## Local build steps
1. Fetch libtorch (pick platform/back-end; default CPU):  
   `LIB_TORCH_COMPUTE_PLATFORM=cpu OUTPUT_PATH=. ./get-pre-build-torch.sh`
2. Configure and build:
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLIBTORCH_ROOT="$PWD/libtorch"
   cmake --build build --target package-swift-zip
   ```
3. Outputs:
   - `build/artifactbundle/ATenCXX.artifactbundle` (folder)
   - `build/artifactbundle/ATenCXX.artifactbundle.zip` (ready for binaryTarget URL)
   - Optional checksum: `swift package compute-checksum build/artifactbundle/ATenCXX.artifactbundle.zip`

## CI / GitHub Actions
- Workflow builds matrix (Linux x86_64 CPU, macOS arm64 CPU) with libtorch 2.9.1.
- Produces zipped bundles and `.checksum` files as artifacts and attaches them to tagged releases.

## Using in Package.swift
- Local dev (path):  
  `binaryTarget(name: "ATenCXX", path: "build/artifactbundle/ATenCXX.artifactbundle")`
- Remote (URL + checksum): upload the zip, run `swift package compute-checksum`, then:
  ```swift
  .binaryTarget(
    name: "ATenCXX",
    url: "<https-url-to-ATenCXX.artifactbundle.zip>",
    checksum: "<sha256>"
  )
  ```

## Notes
- Bundles are platform/compute-specific; build per target combo.
- Consumer machine still needs compatible system libs/driver stack for the chosen libtorch.
