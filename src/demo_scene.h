#ifndef SRC_DEMO_SCENE_H_

#define SRC_DEMO_SCENE_H_

#include "./abstract_scene.h"
#include "./camera.h"
#include "./mesh.h"

#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <map>
#include <memory>

class DemoScene : public AbstractScene
{
 public:
  DemoScene();
  ~DemoScene();

  virtual void initialize();
  virtual void update(double frameTime, QSet<Qt::Key> keysPressed);
  virtual void render();
  virtual void resize(int width, int height);

 private:
  Camera camera;
  double frameTime;
  double cameraSpeed = 10.0f;

  void prepareShaderProgram();
  void prepareVertexBuffers();
  std::map<Qt::Key, std::function<void ()>> keyPressedActions;

  std::unique_ptr<Mesh> mesh;
};

#endif  // SRC_DEMO_SCENE_H_
