# GPGPU_examples

## Fluid Particles
This is a proof of concept implementation of fluid particles on CPU and GPU.
Currently implemented modes are CPU, OpenGL Compute Shader, CUDA and OpenCL.

### Build
This project uses [OpenFrameworks v0.9.8](http://openframeworks.cc/versions/v0.9.8/of_v0.9.8_vs_release.zip) and it is configured that the folder structure looks this:
```
<some folder>
└───GPGPU_examples
│   │   ReadMe.md (this file)
│   │
│   └───FluidParticles
│   |   │   FluidParticles.sln
│   |   │   bin
│   |   └   ...
|   |
|   └───NvidiaCUDA
|   |   |   inc
|   |   └   src
|   |
|   └───NvidiaOpenCL
|   |   |   inc
|   |   |   lib
|   |   └   src
│   
└───of_v0.9.8_vs_release
    │   addons
    │   apps
    └   ...
```

It also requires the [Nvidia CUDA Toolkit 9.1](https://developer.nvidia.com/cuda-downloads) to be installed.

### Controls
| Key        | Function         
| ---------- |-------------|
| left mouse | rotate around y axis |
| right mouse | move camera up and down |
| D | add particles |
| M | switch between CPU and the different GPU modes (if more than 1000 particles are present in the scene "CPU" mode becomes unavailable -> performance) |
| R | set number of particles back to 0 |
| V (hold) | add single particles |
| Esc | close application |

### Settings
The file ```GPGPU_examples/FluidParticles/bin/data/settings.xml``` holds all available settings for the program that can not be changed via the HUD.  

| Command   | Function
| --------- | ------------ |
| MAXPARTICLES | sets the maximum amount of particles in the scene |
| CONTROLS/MOUSESENS | sets the mouse sensitivity |

All settings in ```GPGPU_examples/FluidParticles/bin/data/hud.xml``` are saved automatically and can be modified at runtime.

# Author
Thomas Fischer (psychofisch)
