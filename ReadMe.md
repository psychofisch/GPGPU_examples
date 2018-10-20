# GPGPU_examples

## Intro
This repository contains multiple technical demos that show the power of GPU computing and also the difference between the used APIs.
- Fluid: This is a proof of concept implementation of a fluid particles system. The following modes are available: CPU, OpenGL Compute Shader, CUDA, OpenCL and Thrust. (Thrust is disabled by default, because only part of the implementation works)
- Collision: This demo shows a simple AABB collision detection system. The following modes are available: CPU, OpenGL Compute Shader, CUDA, OpenCL and Thrust.
- MiniGame: This is a *proof of concept* mini game that is a combination of the 2 demos above. Thrust is not available.
- VectorAdd: A simple project that only shows the basics of GPU computing. Available modes: CPU, OpenGL Compute Shader, CUDA, OpenCL and Thrust.

Disclaimer: none of these implementations use optimization algorithms! (like a k-d tree for example) They are not optimized in any way, because this was not the goal of the implementations. The goal was to compare the different APIs on a code and performance basis.

### Build
All of the included projects require [OpenFrameworks v0.9.8](http://openframeworks.cc/versions/v0.9.8/of_v0.9.8_vs_release.zip) and are configured to support the following folder structure below.  
They also requires the [Nvidia CUDA Toolkit 9.1](https://developer.nvidia.com/cuda-downloads).
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
│   └───MiniGame
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

## CollisionDetection
#### Controls
| Key        | Function         
| ---------- |-------------|
| mouse | look around |
| WASD  | move  |
| E  | add boxes  |
| M | switch between CPU and the different GPU modes |
| C | lock camera |
| Esc | close application |

#### Settings
The file ```GPGPU_examples/CollisionDetection/bin/data/settings.xml``` holds all available settings for the program that can not be changed via the HUD.  
```
┌───settings.xml
│   └───GENERAL
│   │   │   BOXES → int, number of boxes
│   │   │   ADD → int, number of boxes that get added with each keypress
│   │   │   VERT → string, filename of the vertex shader for the boxes
│   │   │   FRAG → string, filename of the fragment shader for the boxes
│   │
│   └───CONTROLS
│   │   │   MOUSESENS → float, mouse sensitivity
│   │
│   └───CPU
│   │   │   ENABLED → int, enables/disables mode
│   │   │   THRESHOLD → int, skip CPU mode if #boxes > threshold
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
│   │   │   VERT → string, filename of the vertex shader for the particles
│   │   │   FRAG → string, filename of the fragment shader for the particles
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
│   │   │   WORKGROUPCOUNT → int, size of the workgroup defined in the  compute shader
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

## MiniGame
#### Controls
| Key        | Function         
| ---------- |-------------|
| left mouse | rotate around y axis |
| right mouse | move camera up and down |
| E | start game |
| M | switch between CPU and the different GPU modes (if more than 1000 particles are present in the scene "CPU" mode becomes unavailable -> performance) |
| R | reset game |
| Esc | close application |

#### Settings
The file ```GPGPU_examples/MiniGame/bin/data/settings.xml``` holds all available settings for the program that can not be changed via the HUD.  
```
┌───settings.xml
│   └───GENERAL
│   │   │   TITLE → string, set the game title
│   │   │   MAXPARTICLES → int, maximum number of particles
│   │   │   DROPSIZE → int, number of particles dropped
│   │   │   VERT → string, path to the vertex shader of the particles
│   │   │   FRAG → string, path to the fragment shader of the particles
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

All settings in ```GPGPU_examples/MiniGame/bin/data/hud.xml``` are saved automatically and can be modified at runtime.

# Known Issues
* Collision: While developing my whole system crashed sometimes when switching into CUDA mode. I'm still not sure what is causing this, because I can't reproduce it. BEWARE!
* Collision: After a lot of adding and removing of boxes the rendering failed. Boxes that are not colliding get rendered flickering in a cyan and red.

# Author
[psychofisch](https://twitter.com/psychofish_)
