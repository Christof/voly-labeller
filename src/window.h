#ifndef SRC_WINDOW_H_

#define SRC_WINDOW_H_

#include <QWindow>
#include <QScopedPointer>

/**
 * \brief
 *
 *
 */
class Window : public QWindow
{
  Q_OBJECT
 public:
  Window(QScreen *screen = 0);
  virtual ~Window();
 protected slots:
  void resizeOpenGL();
  void render();
  void update();

 private:
  void initializeOpenGL();

  QOpenGLContext *context;
  // QScopedPointer<AbstractScene> scene;
};

#endif  // SRC_WINDOW_H_
