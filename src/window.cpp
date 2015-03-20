#include "./window.h"
#include <QOpenGLContext>
#include <QCoreApplication>
#include <QKeyEvent>
#include "./abstract_scene.h"

Window::Window(std::shared_ptr<AbstractScene> scene, QWindow *parent)
  : QQuickView(parent), scene(scene), frameCount(0)
{
  setClearBeforeRendering(false);
  // setSurfaceType(OpenGLSurface);

  // create();

  // initializeContext(format);
  /*
  initializeOpenGL();

  scene->setContext(context, gl);
  scene->initialize();
  */

  resize(QSize(1280, 720));

  connect(this, SIGNAL(widthChanged(int)), this, SLOT(resizeOpenGL()));
  connect(this, SIGNAL(heightChanged(int)), this, SLOT(resizeOpenGL()));

  connect(this, SIGNAL(beforeRendering()), this, SLOT(render()),
          Qt::DirectConnection);

  auto format = createSurfaceFormat();
  setFormat(format);

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
  setPersistentOpenGLContext(true);
  context = new QOpenGLContext(this);
  context->setFormat(format);
  context->create();
}

void Window::initializeOpenGL()
{
  context = openglContext();
  // context->makeCurrent(this);
  gl = context->versionFunctions<Gl>();
  // context->makeCurrent(this);
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
  /*
case QEvent::UpdateRequest:
  updatePending = false;
  render();
  return true;
  */
  default:
    return QWindow::event(event);
  }
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
  static bool initialized = false;
  if (!initialized)
  {
    initializeOpenGL();

    scene->setContext(context, gl);
    scene->initialize();
    initialized = true;
  }

  /*
  if (!isExposed())
    return;
    */

  update();
  // context->makeCurrent(this);
  scene->render();
  // context->swapBuffers(this);
  // resetOpenGLState();

  // renderLater();
  ++frameCount;

  QQuickView::update();
}

void Window::resizeOpenGL()
{
  // context->makeCurrent(this);
  scene->resize(width(), height());
}

void Window::update()
{
  scene->update(timer.restart() / 1000.0, keysPressed);
}

