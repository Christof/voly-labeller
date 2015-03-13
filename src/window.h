#ifndef SRC_WINDOW_H_

#define SRC_WINDOW_H_

#include <QWindow>
#include <memory>

class AbstractScene;
class QOpenGLFunctions_4_3_Core;

/**
 * \brief
 *
 *
 */
class Window : public QWindow
{
  Q_OBJECT
 public:
  Window(std::shared_ptr<AbstractScene> scene, QScreen *screen = 0);
  ~Window();
 protected slots:
  void resizeOpenGL();
  void render();
  void update();

 private:
  void initializeOpenGL();

  QOpenGLContext *context;
  QOpenGLFunctions_4_3_Core *gl;
  std::shared_ptr<AbstractScene> scene;
};

#endif  // SRC_WINDOW_H_
