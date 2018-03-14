# GPGPU_examples

## Intro
This is a proof of concept implementation of fluid particles on CPU and GPU.
Currently implemented modes are  
&ensp;Fluid: CPU, OpenGL Compute Shader, CUDA and OpenCL.  
&ensp;CollisionDetection: CPU.

### Build
This project uses [OpenFrameworks v0.9.8](http://openframeworks.cc/versions/v0.9.8/of_v0.9.8_vs_release.zip) and it is configured that the folder structure looks this:
```
<some folder>
└───GPGPU_examples
│   │   ReadMe.md (this file)
│   │
│   └───CollisionDetection
│   │   └   ...
│   │
│   └───Executables
│   │   └   ...
│   │
│   └───FluidParticles
│   │   └   ...
│   │
│   └───NvidiaCUDA
│   │   │   inc
│   │   └   src
│   │
│   └───NvidiaOpenCL
│   │   │   inc
│   │   │   lib
│   │   └   src
│   
└───of_v0.9.8_vs_release
    │   addons
    │   apps
    └   ...
```

It also requires the [Nvidia CUDA Toolkit 9.1](https://developer.nvidia.com/cuda-downloads) to be installed.

## CollisionDetection
#### Controls
| Key        | Function         
| ---------- |-------------|
| mouse | look around |
| WASD  | move  |
| M | switch between CPU and the different GPU modes (if more than 1000 particles are present in the scene "CPU" mode becomes unavailable -> performance) |
| Esc | close application |

#### Settings
The file ```GPGPU_examples/CollisionDetection/bin/data/settings.xml``` holds all available settings for the program that can not be changed via the HUD.  
```
┌───settings.xml
```

All settings in ```GPGPU_examples/CollisionDetection/bin/data/hud.xml``` are saved automatically and can be modified at runtime.

## Fluid
#### Controls
| Key        | Function         
| ---------- |-------------|
| left mouse | rotate around y axis |
| right mouse | move camera up and down |
| E | add particles |
| M | switch between CPU and the different GPU modes (if more than 1000 particles are present in the scene "CPU" mode becomes unavailable -> performance) |
| R | set number of particles back to 0 |
| V (hold) | add single particles |
| Esc | close application |

#### Settings
The file ```GPGPU_examples/FluidParticles/bin/data/settings.xml``` holds all available settings for the program that can not be changed via the HUD.  
```
┌───settings.xml
│   └───GENERAL
│   │   │   MAXPARTICLES → int, maximum number of particles
│   │   │   DROPSIZE → int, number of particles dropped
│   │
│   └───CONTROLS
│   │   │   MOUSESENS → float, mouse sensitivity
│   │
│   └───CPU
│   │   │   ENABLED → int, enables/disables mode
│   │   │   THRESHOLD → int, skip CPU mode if #particles > threshold
│   │
│   └───COMPUTE
│   │   │   ENABLED → int, enables/disables mode
│   │   │   SOURCE → string, path to source file
│   │
│   └───CUDA
│   │   │   ENABLED → int, enables/disables mode
│   │   │   ARGC → int, number of cmd line arguments
│   │   │   ARGV → string, cmd line arguments
│   │
│   └───OCL
│   │   │   ENABLED → int, enables/disables mode
│   │   │   SOURCE → string, path to source file
│   │   │   PLATFORMID → int, specify platformID
│   │   │   DEVICEID → int, specify deviceID
│   │
│   └───THRUST
│   │   │   ENABLED → int, enables/disables mode
```

All settings in ```GPGPU_examples/FluidParticles/bin/data/hud.xml``` are saved automatically and can be modified at runtime.


# Author
Thomas Fischer (psychofisch)
