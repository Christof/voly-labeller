#ifndef SRC_WINDOW_H_

#define SRC_WINDOW_H_

#include <QOpenGLFunctions_4_5_Core>
#include <QtOpenGLExtensions>

#include <QWindow>
#include <QtQuick/QQuickView>
#include <QElapsedTimer>
#include <QSet>
#include <memory>
#include "./graphics/gl.h"

class AbstractScene;
class QStateMachine;
class QOpenGLDebugLogger;
class VideoRecorder;

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
  /**
   * Constructor of Window
   *
   * @param[in] scene The scene which will be displayed
   * @param[in] videoRecorder VideoRecorder instance which is used for
   * video capturing
   * @param[in] offlineRenderingFrameTime If the value is 0, offline rendering
   * is disabled, otherwise the given frame time is used for every
   * Scene::update call.
   */
  explicit Window(std::shared_ptr<AbstractScene> scene,
                  std::shared_ptr<VideoRecorder> videoRecorder,
                  double offlineRenderingFrameTime,
                  QWindow *parent = 0);
  ~Window();

  std::shared_ptr<QStateMachine> stateMachine;

signals:
  void averageFrameTimeUpdated();
  void uiGotFocus();
  void uiLostFocus();

 protected slots:
  void resizeOpenGL();
  void render();
  void update();

  void toggleFullscreen();
  void contextAboutToBeDestroyed();
  void onInvalidated();

  void uiFocusChanged(bool hasFocus);

 private:
  QSurfaceFormat createSurfaceFormat();
  void onMessageLogged(QOpenGLDebugMessage message);
  void handleLazyInitialization();
  void initializeOpenGL();
  void updateAverageFrameTime(double frameTime);

  QElapsedTimer timer;
  QOpenGLContext *context;
  Graphics::Gl *gl = nullptr;
  std::shared_ptr<AbstractScene> scene;
  QOpenGLDebugLogger *logger;
  bool updatePending;
  QSet<Qt::Key> keysPressed;

  int framesInSecond = 0;
  double runningTime = 0;
  double avgFrameTime = 0;

  std::shared_ptr<VideoRecorder> videoRecorder;
  double offlineRenderingFrameTime;
};

#endif  // SRC_WINDOW_H_
