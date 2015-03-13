#include "./window.h"
#include "./abstract_scene.h"
#include <QOpenGLContext>
#include <QTimer>

Window::Window(std::shared_ptr<AbstractScene> scene, QScreen *screen)
  : QWindow(screen), scene(scene)
{
  setSurfaceType(OpenGLSurface);

  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  format.setMajorVersion(4);
  format.setMinorVersion(3);
  format.setSamples(4);
  format.setProfile(QSurfaceFormat::CoreProfile);

  setFormat(format);
  create();

  context = new QOpenGLContext();
  context->setFormat(format);
  context->create();

  scene->setContext(context);

  initializeOpenGL();

  resize(QSize(1280, 720));

  connect(this, SIGNAL(widthChanged(int)), this, SLOT(resizeOpenGL()));
  connect(this, SIGNAL(heightChanged(int)), this, SLOT(resizeOpenGL()));

  QTimer *timer = new QTimer(this);
  connect(timer, SIGNAL(timeout()), this, SLOT(update()));
  timer->start(16);
}

Window::~Window()
{
}

void Window::initializeOpenGL()
{
  context->makeCurrent(this);
  scene->initialize();
}

void Window::render()
{
  if (!isExposed())
    return;
  context->makeCurrent(this);
  scene->render();
  context->swapBuffers(this);
}

void Window::resizeOpenGL()
{
  context->makeCurrent(this);
  scene->resize(width(), height());
}

void Window::update()
{
  scene->update(0.0f);
  render();
}

