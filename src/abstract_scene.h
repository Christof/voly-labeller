#ifndef SRC_ABSTRACT_SCENE_H_

#define SRC_ABSTRACT_SCENE_H_

#include <QOpenGLContext>
#include <QtDebug>
#include <QOpenGLFunctions_4_3_Core>

class AbstractScene
{
 public:
  AbstractScene() : context(0)
  {
  }
  virtual ~AbstractScene()
  {
  }

  void setContext(QOpenGLContext *context, QOpenGLFunctions_4_3_Core *gl)
  {
    this->context = context;
    this->gl = gl;
  }

  virtual void initialize() = 0;

  virtual void update(float t) = 0;

  virtual void render() = 0;

  virtual void resize(int width, int height) = 0;

 protected:
  QOpenGLContext *context;
  QOpenGLFunctions_4_3_Core *gl;
};

#endif  // SRC_ABSTRACT_SCENE_H_
