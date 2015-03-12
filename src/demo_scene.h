#ifndef SRC_DEMO_SCENE_H_

#define SRC_DEMO_SCENE_H_

#include "./abstract_scene.h"

#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>

class DemoScene : public AbstractScene
{
 public:
  DemoScene();
  ~DemoScene();

  virtual void initialize();
  virtual void update(float t);
  virtual void render();
  virtual void resize(int width, int height);

 private:
  QOpenGLShaderProgram shaderProgram;
  QOpenGLVertexArrayObject vertexArrayObject;
  QOpenGLBuffer positionBuffer;
  QOpenGLBuffer colorBuffer;

  void prepareShaderProgram();
  void prepareVertexBuffers();
};

#endif  // SRC_DEMO_SCENE_H_
