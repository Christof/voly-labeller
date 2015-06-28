#include "./window.h"
#include <QOpenGLContext>
#include <QtOpenGLExtensions>
#include <QDebug>
#include <QCoreApplication>
#include <QKeyEvent>
#include <QStateMachine>
#include <QAbstractState>
#include <QAbstractTransition>
#include <iostream>
#include "./gl.h"
#include "./abstract_scene.h"

Window::Window(std::shared_ptr<AbstractScene> scene, QWindow *parent)
  : QQuickView(parent), scene(scene), frameCount(0)
{
  setClearBeforeRendering(false);

  connect(this, SIGNAL(widthChanged(int)), this, SLOT(resizeOpenGL()));
  connect(this, SIGNAL(heightChanged(int)), this, SLOT(resizeOpenGL()));

  connect(reinterpret_cast<QObject *>(engine()), SIGNAL(quit()), this,
          SLOT(close()));

  connect(this, SIGNAL(beforeRendering()), this, SLOT(render()),
          Qt::DirectConnection);

  auto format = createSurfaceFormat();
  setFormat(format);

  timer.start();
}

Window::~Window()
{
  delete gl;
}

QSurfaceFormat Window::createSurfaceFormat()
{
  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  format.setMajorVersion(4);
  format.setMinorVersion(3);
  format.setSamples(4);

  return format;
}

void Window::initializeOpenGL()
{
  context = openglContext();
  gl = new Gl();
  gl->initialize(size());

  qWarning() << "Has GL_NV_shader_buffer_load:"
             << context->hasExtension("GL_NV_shader_buffer_load");
  QOpenGLExtension_NV_shader_buffer_load *b =
      new QOpenGLExtension_NV_shader_buffer_load();
  b->initializeOpenGLFunctions();
  glCheckError();

  gl->glEnable(GL_DEPTH_TEST);
  gl->glEnable(GL_BLEND);
  gl->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Window::keyReleaseEvent(QKeyEvent *event)
{
  QQuickView::keyReleaseEvent(event);
  keysPressed -= static_cast<Qt::Key>(event->key());
}

void Window::keyPressEvent(QKeyEvent *event)
{
  QQuickView::keyPressEvent(event);
  keysPressed += static_cast<Qt::Key>(event->key());
}

void Window::handleLazyInitialization()
{
  static bool initialized = false;
  if (!initialized)
  {
    initializeOpenGL();

    scene->setContext(context, gl);
    scene->resize(size().width(), size().height());
    scene->initialize();
    initialized = true;
  }
}

void Window::render()
{
  handleLazyInitialization();

  update();
  scene->render();

  // Use to check for missing release calls
  // resetOpenGLState();

  ++frameCount;

  QQuickView::update();
}

void Window::resizeOpenGL()
{
  if (!gl)
    return;

  scene->resize(width(), height());
  gl->setSize(this->size());
}

void Window::update()
{
  double frameTime = timer.restart() / 1000.0;
  updateAverageFrameTime(frameTime);

  scene->update(frameTime, keysPressed);
}

void Window::toggleFullscreen()
{
  setVisibility(visibility() == QWindow::Windowed ? QWindow::FullScreen
                                                  : QWindow::Windowed);
}

void Window::updateAverageFrameTime(double frameTime)
{
  runningTime += frameTime;
  ++framesInSecond;

  if (runningTime > 1.0)
  {
    avgFrameTime = runningTime / framesInSecond;
    emit averageFrameTimeUpdated();

    framesInSecond = 0;
    runningTime = 0;
  }
}

