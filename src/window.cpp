#include "./window.h"
#include <QOpenGLContext>
#include <QOpenGLFunctions_4_3_Core>
#include <QCoreApplication>
#include <QKeyEvent>
#include "./abstract_scene.h"
#include "./gl_assert.h"

Window::Window(std::shared_ptr<AbstractScene> scene, QWindow *parent)
  : QQuickView(parent), scene(scene), frameCount(0)
{
  setSurfaceType(OpenGLSurface);

  auto format = createSurfaceFormat();
  setFormat(format);
  create();

  initializeContext(format);
  initializeOpenGL();

  scene->setContext(context, gl);
  scene->initialize();

  resize(QSize(1280, 720));

  connect(this, SIGNAL(widthChanged(int)), this, SLOT(resizeOpenGL()));
  connect(this, SIGNAL(heightChanged(int)), this, SLOT(resizeOpenGL()));

  timer.start();
}

Window::~Window()
{
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

void Window::initializeContext(QSurfaceFormat format)
{
  context = new QOpenGLContext();
  context->setFormat(format);
  context->create();
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

  gl->glEnable(GL_DEPTH_TEST);
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

void Window::keyReleaseEvent(QKeyEvent *event)
{
  if (event->key() == Qt::Key_Escape)
  {
    close();
  }

  keysPressed -= static_cast<Qt::Key>(event->key());
}

void Window::keyPressEvent(QKeyEvent *event)
{
  keysPressed += static_cast<Qt::Key>(event->key());
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
  ++frameCount;
}

void Window::resizeOpenGL()
{
  context->makeCurrent(this);
  scene->resize(width(), height());
}

void Window::update()
{
  scene->update(timer.restart() / 1000.0, keysPressed);
}

