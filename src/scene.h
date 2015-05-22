#ifndef SRC_SCENE_H_

#define SRC_SCENE_H_

#include <memory>
#include "./abstract_scene.h"
#include "./camera.h"
#include "./forces/labeller.h"

class Nodes;
class InvokeManager;
class CameraController;
class CameraRotationController;
class CameraZoomController;
class CameraMoveController;
class Quad;
class FrameBufferObject;

/**
 * \brief Default implementation of AbstractScene
 *
 * It uses Nodes to manage which nodes are visible.
 * In initialize a default scene is loaded for now
 * (and also persisted as `config/scene.xml`).
 */
class Scene : public AbstractScene
{
 public:
  Scene(std::shared_ptr<InvokeManager> invokeManager,
        std::shared_ptr<Nodes> nodes,
        std::shared_ptr<Forces::Labeller> labeller);
  ~Scene();

  virtual void initialize();
  virtual void update(double frameTime, QSet<Qt::Key> keysPressed);
  virtual void render();
  virtual void resize(int width, int height);

  void pick(Eigen::Vector2f position);

 private:
  Camera camera;
  std::shared_ptr<CameraController> cameraController;
  std::shared_ptr<CameraRotationController> cameraRotationController;
  std::shared_ptr<CameraZoomController> cameraZoomController;
  std::shared_ptr<CameraMoveController> cameraMoveController;
  double frameTime;

  std::shared_ptr<Nodes> nodes;
  std::shared_ptr<Forces::Labeller> labeller;
  std::shared_ptr<Quad> quad;
  std::unique_ptr<FrameBufferObject> fbo;

  int width;
  int height;
  bool shouldResize = false;

  void renderScreenQuad();

  bool performPicking;
  Eigen::Vector2f pickingPosition;
  void doPick();
};

#endif  // SRC_SCENE_H_
