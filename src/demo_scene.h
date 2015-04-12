#ifndef SRC_DEMO_SCENE_H_

#define SRC_DEMO_SCENE_H_

#include <map>
#include <vector>
#include <memory>

#include "./abstract_scene.h"
#include "./camera.h"
#include "./camera_controller.h"
#include "./mesh.h"

class Node;
class InvokeManager;

class DemoScene : public AbstractScene
{
 public:
  DemoScene(std::shared_ptr<InvokeManager> invokeManager);
  ~DemoScene();

  virtual void initialize();
  virtual void update(double frameTime, QSet<Qt::Key> keysPressed);
  virtual void render();
  virtual void resize(int width, int height);

  virtual void loadScene(std::string filename);


 private:
  Camera camera;
  std::shared_ptr<CameraController> cameraController;
  double frameTime;
  double cameraSpeed = 10.0f;


  void prepareShaderProgram();
  void prepareVertexBuffers();

  std::vector<std::shared_ptr<Node>> nodes;
};

#endif  // SRC_DEMO_SCENE_H_
