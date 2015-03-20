#ifndef SRC_WINDOW_H_

#define SRC_WINDOW_H_

#include <QWindow>
#include <QtQuick/QQuickView>
#include <QElapsedTimer>
#include <QSet>
#include <memory>
#include "./gl.h"

class AbstractScene;

/**
 * \brief
 *
 *
 */
class Window : public QQuickView
{
  Q_OBJECT
 public:
  explicit Window(std::shared_ptr<AbstractScene> scene, QWindow *parent = 0);
  ~Window();
 protected slots:
  void resizeOpenGL();
  void render();
  void update();

 protected:
  bool event(QEvent *event) Q_DECL_OVERRIDE;
  void keyReleaseEvent(QKeyEvent *ev) Q_DECL_OVERRIDE;
  void keyPressEvent(QKeyEvent *ev) Q_DECL_OVERRIDE;

 private:
  QSurfaceFormat createSurfaceFormat();
  void initializeContext(QSurfaceFormat format);
  void initializeOpenGL();
  void renderLater();

  QElapsedTimer timer;
  QOpenGLContext *context;
  Gl *gl;
  std::shared_ptr<AbstractScene> scene;
  bool updatePending;
  long frameCount;
  QSet<Qt::Key> keysPressed;
};

#endif  // SRC_WINDOW_H_
