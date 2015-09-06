#ifndef SRC_WINDOW_H_

#define SRC_WINDOW_H_

#include <QWindow>
#include <QtQuick/QQuickView>
#include <QElapsedTimer>
#include <QSet>
#include <memory>
#include "./graphics/gl.h"

class AbstractScene;
class QStateMachine;
class QOpenGLDebugLogger;

/**
 * \brief Main window which draws the 3D scene before Qt Gui is drawn
 *
 *
 */
class Window : public QQuickView
{
  Q_OBJECT
  Q_PROPERTY(double averageFrameTime MEMBER avgFrameTime NOTIFY
                 averageFrameTimeUpdated)
 public:
  explicit Window(std::shared_ptr<AbstractScene> scene, QWindow *parent = 0);
  ~Window();

  std::shared_ptr<QStateMachine> stateMachine;

signals:
  void averageFrameTimeUpdated();

 protected slots:
  void resizeOpenGL();
  void render();
  void update();

  void toggleFullscreen();

 protected:
  void keyReleaseEvent(QKeyEvent *ev) Q_DECL_OVERRIDE;
  void keyPressEvent(QKeyEvent *ev) Q_DECL_OVERRIDE;

 private:
  QSurfaceFormat createSurfaceFormat();
  void onMessageLogged(QOpenGLDebugMessage message);
  void handleLazyInitialization();
  void initializeOpenGL();
  void updateAverageFrameTime(double frameTime);

  QElapsedTimer timer;
  QOpenGLContext *context;
  std::shared_ptr<Graphics::Gl> gl;
  std::shared_ptr<AbstractScene> scene;
  QOpenGLDebugLogger *logger;
  bool updatePending;
  QSet<Qt::Key> keysPressed;

  int framesInSecond = 0;
  double runningTime = 0;
  double avgFrameTime = 0;
};

#endif  // SRC_WINDOW_H_
