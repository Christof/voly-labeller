#ifndef SRC_ABSTRACT_SCENE_H_

#define SRC_ABSTRACT_SCENE_H_

class QOpenGLContext;

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
};

#endif  // SRC_ABSTRACT_SCENE_H_
