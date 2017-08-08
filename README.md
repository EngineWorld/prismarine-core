# OpenGL Ray Tracer (aka. "Paper")

## Updates

<details>
<summary>Highway Update (beta)</summary>

### "Highway" Update (formelly OpenGL 4.6) Mainly Done (06.08.2017)

<img src="logo/highway.png" alt="HighwayUpdate" width="960"/>

- [x] Use compatible library with OpenGL 4.6
- [x] Change shading language version to *460*
- [x] OpenGL 4.6 group vote in traverse code
- [x] Extended SPIR-V support
- [x] Indirect mesh loading
- [ ] Memory and semaphores extensions support
- [ ] Float 16 bit support
- [ ] Float 64 bit support
</details>

<details>
<summary>Master Update (on development)</summary>

### "Master" Update (low level API, tweaks) WIP (??.08.2017)

<img src="logo/paper.png" alt="MasterUpdate" width="1024"/>

- [x] Support of accessor buffers
- [x] Lower level accessor set
- [x] Fully device memory based storing and copying, when loading mesh
- [x] Divide uploading and current buffer storing (you can load meshes, while you using in traverse stable buffers)
- [x] Divide BVH traverse and primitive intersection stages
- [x] Surface and environment shaders
- [ ] Improved support of storing constants
- [ ] Advanced optimization
</details>

<details>
<summary>Clockwork Update (on development)</summary>

### "Clockwork" Update (improved BVH system) planned (??.??.2017)

<img src="logo/clockwork.png" alt="ClockworkUpdate" width="960"/>

- [x] Initial updates and factoring
- [ ] Grouping geometry nodes by 32 primitives (expect that it will after morton code sorting stage)
- [ ] SIMD optimized intersection stage
- [ ] Sorting by ranging (for better exclusion of triangle hits)
- [ ] Consideration of trBVH support
</details>

## Details

- Here is headers-only (at this moment) ray tracer. May used for other projects. Main framework in shaders.

## Features: 

- CMake support (Windows)
- Fixed most bugs
- GPU optimized BVH (HLBVH)
- Direct light for sun
- Modern OpenGL based
- Support NVidia GTX 1070 and newer
- Optimized for performance
- Open source (at now)

## Requirement

- OpenGL 4.6 with extensions :)
- Latest CMake

## Building 

- Run CMAKE and configure project (probably, available only for Windows)
- Also, you need [shaderc](https://github.com/google/shaderc) for preprocess shaders (also can be found with Vulkan SDK)

## Running and testing

Basic render application: 

```
${ApplicationName}.exe -m sponza.obj -s 1.0
-m model_name.obj   - loading 3D model to view (planned multiply models and animation support)
-s 1.0              - scaling of 3D model
```

## Contributors

- ???

## Leaders

- Alexey S (capitalknew@gmail.com)

## Inspired by

- [RadeonRays SDK](https://github.com/GPUOpen-LibrariesAndSDKs/RadeonRays_SDK)
- [WebGL Path Tracing by evanw (and forks)](https://github.com/evanw/webgl-path-tracing)
- [GPU Path Tracer by peterkutz](https://github.com/peterkutz/GPUPathTracer)
- [Something from Shadertoy](https://www.shadertoy.com/)
- [Radix Sort by CiNoNim](https://github.com/cNoNim/radix-sort)
- Other functions and modifications from few resources
