#ifndef SRC_ABSTRACT_SCENE_H_

#define SRC_ABSTRACT_SCENE_H_

#include <QOpenGLContext>
#include <QtDebug>
#include <QKeyEvent>
#include "./gl.h"

class AbstractScene
{
 public:
  AbstractScene() : context(0)
  {
  }
  virtual ~AbstractScene()
  {
  }

  void setContext(QOpenGLContext *context, Gl *gl)
  {
    this->context = context;
    this->gl = gl;
  }

  virtual void initialize() = 0;

  virtual void update(double frameTime, QSet<Qt::Key> keysPressed) = 0;

  virtual void render() = 0;

  virtual void resize(int width, int height) = 0;

 protected:
  QOpenGLContext *context;
  Gl *gl;
};

#endif  // SRC_ABSTRACT_SCENE_H_
