#ifndef SRC_ABSTRACT_SCENE_H_

#define SRC_ABSTRACT_SCENE_H_

#include <QOpenGLContext>
#include <QOpenGLFunctions_4_3_Core>
#include <QtDebug>

#include "./gl_assert.h"

class AbstractScene
{
 public:
  AbstractScene() : mContext(0)
  {
  }
  virtual ~AbstractScene()
  {
  }

  void setContext(QOpenGLContext *context)
  {
    mContext = context;
    gl = context->versionFunctions<QOpenGLFunctions_4_3_Core>();
    if (!gl) {
        qWarning() << "Could not obtain required OpenGL context version";
        exit(1);
    }
    gl->initializeOpenGLFunctions();
    glCheckError();
  }

  QOpenGLContext *context() const
  {
    return mContext;
  }

  virtual void initialize() = 0;

  virtual void update(float t) = 0;

  virtual void render() = 0;

  virtual void resize(int width, int height) = 0;

 protected:
  QOpenGLContext *mContext;
  QOpenGLFunctions_4_3_Core *gl;
};

#endif  // SRC_ABSTRACT_SCENE_H_
