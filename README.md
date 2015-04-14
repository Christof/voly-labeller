# voly-labeller
Multiple labeller implementation for volume rendered medical data

# Controls
## File menu
The file menu provides items to:
- open a scene (`Ctrl+O`)
- import assets (`Ctrl+I`)
- save a scene (`Ctrl+S`)
- reset the scene (`Ctrl+R`)
- hide the ui (`F1`)
- exit the application (`Esc`)

## 3D control and states
The 3D controls and state transitions are defined in `config/states.xml`.
Camera movements are defined to move forward and backward (`W` and `S` respectively),
to strafe left or right (`A` and `D`), to increase and decrease declination (`R` and `F`)
as well as to change the azimuth (`Q` and `E`).

# Building
```
mkdir build
cd build
cmake ..
make
./voly-labeller
```

# Dependencies
- Qt5
- OpenGL 4.3
- Eigen3
- Assimp
- Boost 1.57
- GTest
