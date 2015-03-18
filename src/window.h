#ifndef SRC_WINDOW_H_

#define SRC_WINDOW_H_

#include <QWindow>
#include <QElapsedTimer>
#include <QSet>
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
  explicit Window(std::shared_ptr<AbstractScene> scene, QScreen *screen = 0);
  ~Window();
 protected slots:
  void resizeOpenGL();
  void render();
  void update();

 protected:
  bool event(QEvent *event) Q_DECL_OVERRIDE;
  void exposeEvent(QExposeEvent *event) Q_DECL_OVERRIDE;
  void keyReleaseEvent(QKeyEvent *ev) Q_DECL_OVERRIDE;
  void keyPressEvent(QKeyEvent *ev) Q_DECL_OVERRIDE;

 private:
  void initializeOpenGL();
  void renderLater();

  QElapsedTimer timer;
  QOpenGLContext *context;
  QOpenGLFunctions_4_3_Core *gl;
  std::shared_ptr<AbstractScene> scene;
  bool updatePending;
  long frameCount;
  QSet<Qt::Key> keysPressed;
};

#endif  // SRC_WINDOW_H_
