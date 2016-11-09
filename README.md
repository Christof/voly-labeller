# voly-labeller
Multiple labeller implementation for volume rendered medical data

# Contributing
Contribution guidelines are given in [CONTRIBUTING.md](CONTRIBUTING.md).

# Controls
## Command-line interface

```
Usage: ./voly-labeller [options] scene
Multiple labelling implementations for volume rendered medical data

Options:
  -h, --help                          Displays this help.
  -v, --version                       Displays version information.
  --offline                           Enables offline rendering
  --layers <Layer Count>              Number of layers. Default is 4
  --hard-constraints                  Simulate hard constraints
  --apollonius                        Use apollonius graph to determine label
                                      insertion order
  --disable-labelling                 Disable drawing lables
  --internal-labelling                Enable internal labelling
  --optimize-on-idle                  Optimize costs when the camera is not
                                      moving
  -s, --screenshot <Camera Position>  Takes a screenshot of the given camera
                                      position. Characters after a '_' are
                                      ignored but added to the filenameCamera
                                      Position. Multiple camera positions for
                                      multiple screenshots can be supplied with
                                      ',' as separator. Camera Positions
                                      containing a space character are not
                                      supported!

Arguments:
  scene                               Scene file to load.
```

## File menu
The file menu provides items to:
- open a scene (`Ctrl+O`)
- import assets (`Ctrl+I`)
- save a scene (`Ctrl+S`)
- reset the scene (`Ctrl+R`)
- exit the application (`Esc`)

The view menu enables you to:
- hide the ui (`F1`)
- toggle bounding volumes (`F2`)
- toggle forces info (`F3`)
- toggle fullscreen mode (`F11`)

The simulation menu has items to:
- toggle the label update (`Space`). If the label update is disabled the best new position is
  still calculated but not applied.

## 3D control and states
The 3D controls and state transitions are defined in `config/states.xml`.
Camera movements are defined to move forward and backward (`W` and `S` respectively),
to strafe left or right (`A` and `D`), to move up and down (`R` and `F`),
to increase and decrease declination (`R` and `F`)
as well as to change the azimuth (`T` and `G`). The camera can also be rotated around the
origin using mouse movement while holding the left mouse button. While holding the control key
and the left mouse button, the zoom mode is enabled. Moving the mouse up zooms into the scene;
moving the mouse down zooms out of the scene. Holding the shift key and the left mouse button,
enables the camera move mode. The camera is moved inverse to the mouse movements.

# Assets
Some assets - which are necessary to run the application - can be downloaded from
a shared [folder](https://drive.google.com/folderview?id=0ByTbZ7z8JSt-fnRNM09UcVNRQ3BBVnA2ZUx1bjFidXRnSDgtN0dqaEdya2d6MjJDcmJ6Wms&usp=sharing). They must be located in the `assets` in the projects root directory.

# Building
```
mkdir build
cd build
cmake ..
make
./voly-labeller
```

# Dependencies
- Qt5.7
- OpenGL 4.5
- Eigen3
- Assimp
- Boost 1.59
- ITK 4.7
- CUDA 7.0
- Thrust 1.8
- Magick++ 8:6.7.7.10-6

Some of these can be installed by running `scripts/install_dependencies.sh`.
CUDA (including THRUST), Qt5.7 and OpenGL 4.5 must be installed separately.

## Building
- gcc 4.9.3

## Testing
- GTest
- lcov (from [source](https://github.com/linux-test-project/lcov) or at least 1.11)

### GTest
GTest can be installed by installing the library with `apt-get`:

```
sudo apt-get install libgtest-dev
```

Afterwards the sources needs to be compiled and library file needs to be moved:

```
cd /usr/src/gtest
sudo cmake .
sudo make
sudo mv libg* /usr/lib/
```

