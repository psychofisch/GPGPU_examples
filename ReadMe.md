# GPGPU_examples

## Fluid Particles
This is a proof of concept implementation of fluid particles on CPU and GPU.

### Build
This project uses [OpenFrameworks v0.9.8](http://openframeworks.cc/versions/v0.9.8/of_v0.9.8_vs_release.zip) and it is configured that the folder structure looks this:
```
<some folder>
└───GPGPU_examples
│   │   ReadMe.md (this file)
│   │
|   └───NvidiaOpenCL
|   |   |   inc
|   |   |   lib
|   |   |   src
│   └───FluidParticles
│       │   FluidParticles.sln
│       │   bin
│       │   ...
│   
└───of_v0.9.8_vs_release
    │   addons
    │   apps
    |   ...
```

It also requires the [Nvidia CUDA Toolkit 9.1](https://developer.nvidia.com/cuda-downloads) to be installed.

### Controls
| Key        | Function         
| ---------- |-------------|
| left mouse | rotate around y axis |
| right mouse | move camera up and down |
| D | add particles |
| M | switch between CPU and GPU (currently OpenGL compute shaders) |
| R | set number of particles back to 0 |
| V (hold) | add single particles |
| Esc | close application |

# Author
Thomas Fischer (psychofisch)
