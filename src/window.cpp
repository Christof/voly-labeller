#include "./window.h"
#include "./abstract_scene.h"
#include <QOpenGLContext>
#include <QOpenGLFunctions_4_3_Core>
#include <QCoreApplication>
#include "./gl_assert.h"

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

  initializeOpenGL();

  scene->setContext(context, gl);
  scene->initialize();

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
  gl = context->versionFunctions<QOpenGLFunctions_4_3_Core>();
  if (!gl)
  {
    qWarning() << "Could not obtain required OpenGL context version";
    exit(1);
  }
  gl->initializeOpenGLFunctions();
  glCheckError();
}

void Window::renderLater()
{
  if (!updatePending)
  {
    updatePending = true;
    QCoreApplication::postEvent(this, new QEvent(QEvent::UpdateRequest));
  }
}

bool Window::event(QEvent *event)
{
  switch (event->type())
  {
  case QEvent::UpdateRequest:
    updatePending = false;
    render();
    return true;
  default:
    return QWindow::event(event);
  }
}

void Window::exposeEvent(QExposeEvent *event)
{
  Q_UNUSED(event);

  if (isExposed())
    render();
}

void Window::render()
{
  if (!isExposed())
    return;

  update();
  context->makeCurrent(this);
  scene->render();
  context->swapBuffers(this);

  renderLater();
}

void Window::resizeOpenGL()
{
  context->makeCurrent(this);
  scene->resize(width(), height());
}

void Window::update()
{
  scene->update(0.0f);
}

