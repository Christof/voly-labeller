#include "./window.h"
#include <QOpenGLContext>

Window::Window(QScreen *screen) : QWindow(screen)
{
  setSurfaceType(OpenGLSurface);

  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  format.setMajorVersion(4);
  format.setMinorVersion(4);
  format.setSamples(4);
  format.setProfile(QSurfaceFormat::CoreProfile);

  setFormat(format);
  create();

  context = new QOpenGLContext();
  context->setFormat(format);
  context->create();

  // scene->setContext(context);

  initializeOpenGL();

  resize(QSize(1280, 720));

  connect(this, SIGNAL(widthChanged(int)), this, SLOT(resizeOpenGL()));
  connect(this, SIGNAL(heightChanged(int)), this, SLOT(resizeOpenGL()));
}

Window::~Window()
{
}

void Window::initializeOpenGL()
{
  context->makeCurrent(this);
}

void Window::render()
{
  if (!isExposed())
    return;
  context->makeCurrent(this);
  // scene->render();
  context->swapBuffers(this);
}

void Window::resizeOpenGL()
{
  context->makeCurrent(this);
  // scene->resize(width(), height());
}

void Window::update()
{
  // scene->update(0.0f);
  render();
}
