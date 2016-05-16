#ifndef SRC_ABSTRACT_SCENE_H_

#define SRC_ABSTRACT_SCENE_H_

#include "./graphics/gl.h"

class QOpenGLContext;

/**
 * \brief Base class for all scenes.
 *
 * It consists of the following virtual methods:
 * - AbstractScene::initialize is called before the scene is rendered the first
 *   time
 * - AbstractScene::update is called every frame to update the scene
 * - AbstractScene::render is called every frame to render the scene
 * - AbstractScene::resize is called when the window size is changed
 */
class AbstractScene
{
 public:
  AbstractScene() : context(0)
  {
  }

  virtual ~AbstractScene()
  {
  }

  void setContext(QOpenGLContext *context, Graphics::Gl *gl)
  {
    this->context = context;
    this->gl = gl;
  }

  virtual void initialize() = 0;
  virtual void cleanup() = 0;

  virtual void update(double frameTime) = 0;

  virtual void render() = 0;

  virtual void resize(int width, int height) = 0;

 protected:
  QOpenGLContext *context;
  Graphics::Gl *gl;
};

#endif  // SRC_ABSTRACT_SCENE_H_
