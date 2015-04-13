#ifndef SRC_WINDOW_H_

#define SRC_WINDOW_H_

#include <QWindow>
#include <QtQuick/QQuickView>
#include <QElapsedTimer>
#include <QSet>
#include <memory>
#include "./gl.h"

class AbstractScene;
class QStateMachine;

/**
 * \brief Main window which draws the 3D scene before Qt Gui is drawn
 *
 *
 */
class Window : public QQuickView
{
  Q_OBJECT
 public:
  explicit Window(std::shared_ptr<AbstractScene> scene, QWindow *parent = 0);
  ~Window();

  std::shared_ptr<QStateMachine> stateMachine;
 protected slots:
  void resizeOpenGL();
  void render();
  void update();

  void printCurrentState();

 protected:
  void keyReleaseEvent(QKeyEvent *ev) Q_DECL_OVERRIDE;
  void keyPressEvent(QKeyEvent *ev) Q_DECL_OVERRIDE;

 private:
  QSurfaceFormat createSurfaceFormat();
  void handleLazyInitialization();
  void initializeContext(QSurfaceFormat format);
  void initializeOpenGL();

  QElapsedTimer timer;
  QOpenGLContext *context;
  Gl *gl = nullptr;
  std::shared_ptr<AbstractScene> scene;
  bool updatePending;
  long frameCount;
  QSet<Qt::Key> keysPressed;
};

#endif  // SRC_WINDOW_H_
