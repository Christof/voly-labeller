#ifndef SRC_DEMO_SCENE_H_

#define SRC_DEMO_SCENE_H_

#include <memory>
#include "./abstract_scene.h"
#include "./camera.h"

class Nodes;
class InvokeManager;
class CameraController;

/**
 * \brief Default implementation of AbstractScene
 *
 * It uses Nodes to manage which nodes are visible.
 * In initialize a default scene is loaded for now
 * (and also persisted as `config/scene.xml`).
 */
class DemoScene : public AbstractScene
{
 public:
  DemoScene(std::shared_ptr<InvokeManager> invokeManager,
            std::shared_ptr<Nodes> nodes);
  ~DemoScene();

  virtual void initialize();
  virtual void update(double frameTime, QSet<Qt::Key> keysPressed);
  virtual void render();
  virtual void resize(int width, int height);

 private:
  Camera camera;
  std::shared_ptr<CameraController> cameraController;
  double frameTime;

  std::shared_ptr<Nodes> nodes;
};

#endif  // SRC_DEMO_SCENE_H_
