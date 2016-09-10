#include "./window.h"
#include <QOpenGLContext>
#include <QOpenGLDebugLogger>
#include <QDebug>
#include <QCoreApplication>
#include <QStateMachine>
#include <QAbstractState>
#include <QAbstractTransition>
#include <QLoggingCategory>
#include "./graphics/gl.h"
#include "./abstract_scene.h"
#include "./video_recorder.h"
#include "./utils/image_persister.h"

QLoggingCategory openGlChan("OpenGl");

Window::Window(std::shared_ptr<AbstractScene> scene,
               std::shared_ptr<VideoRecorder> videoRecorder,
               double offlineRenderingFrameTime, QWindow *parent)
  : QQuickView(parent), scene(scene), videoRecorder(videoRecorder),
    offlineRenderingFrameTime(offlineRenderingFrameTime)
{
  setClearBeforeRendering(false);

  connect(this, SIGNAL(widthChanged(int)), this, SLOT(resizeOpenGL()));
  connect(this, SIGNAL(heightChanged(int)), this, SLOT(resizeOpenGL()));
  connect(this, &Window::sceneGraphInvalidated, this, &Window::onInvalidated,
          Qt::DirectConnection);

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
  disconnect(logger, &QOpenGLDebugLogger::messageLogged, this,
             &Window::onMessageLogged);
}

QSurfaceFormat Window::createSurfaceFormat()
{
  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  format.setStencilBufferSize(8);
  format.setMajorVersion(4);
  format.setMinorVersion(5);
  format.setSamples(4);
  format.setOption(QSurfaceFormat::DebugContext);
  format.setSwapInterval(0);

  return format;
}

void Window::initializeOpenGL()
{
  qCInfo(openGlChan) << "initializeOpenGL";
  context = openglContext();
  bool success = context->makeCurrent(this);
  qCInfo(openGlChan) << "success: " << success;

  gl = new Graphics::Gl();
  gl->initialize(context, size());

  logger = new QOpenGLDebugLogger(context);
  connect(context, &QOpenGLContext::aboutToBeDestroyed, this,
          &Window::contextAboutToBeDestroyed, Qt::DirectConnection);

  connect(logger, &QOpenGLDebugLogger::messageLogged, this,
          &Window::onMessageLogged, Qt::DirectConnection);

  if (logger->initialize())
  {
    logger->startLogging(QOpenGLDebugLogger::SynchronousLogging);
    logger->enableMessages();
  }

  glAssert(gl->glDisable(GL_CULL_FACE));
  glAssert(gl->glDisable(GL_DEPTH_TEST));
  glAssert(gl->glDisable(GL_STENCIL_TEST));
  glAssert(gl->glDisable(GL_BLEND));
  glAssert(gl->glDepthMask(GL_FALSE));

  videoRecorder->initialize(gl);
  videoRecorder->resize(this->width(), this->height());
}

void Window::onInvalidated()
{
  qCInfo(openGlChan) << "on invalidated: delete logger";
  scene->cleanup();
  delete logger;
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

  QQuickView::update();

  if (takeScreenshot)
  {
    int width = size().width();
    int height = size().height();
    std::vector<unsigned char> pixels(width * height * 4);
    gl->glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                     pixels.data());
    ImagePersister::flipAndSaveRGBA8I(pixels.data(), width, height,
                                      "screenshot.png");
    takeScreenshot = false;
  }

  videoRecorder->captureVideoFrame();
}

void Window::resizeOpenGL()
{
  if (!gl)
    return;

  qCWarning(openGlChan) << "Resize not supported for now";
  /*
  scene->resize(width(), height());
  gl->setSize(this->size());
  */
}

void Window::update()
{
  double frameTime = timer.restart() / 1000.0;
  updateAverageFrameTime(frameTime);

  scene->update(offlineRenderingFrameTime != 0.0f ? offlineRenderingFrameTime
                                                  : frameTime);
}

void Window::toggleFullscreen()
{
  setVisibility(visibility() == QWindow::Windowed ? QWindow::FullScreen
                                                  : QWindow::Windowed);
}

void Window::contextAboutToBeDestroyed()
{
  qCInfo(openGlChan) << "Closing rendering thread";
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

void Window::uiFocusChanged(bool hasFocus)
{
  qCDebug(openGlChan) << "uiFocusChanged" << hasFocus;
  if (hasFocus)
    emit uiGotFocus();
  else
    emit uiLostFocus();
}

void Window::takeScreenshotOfNextFrame()
{
  takeScreenshot = true;
}

void Window::onMessageLogged(QOpenGLDebugMessage message)
{
  // Ignore buffer detailed info which cannot be fixed
  if (message.id() == 131185)
    return;

  // Ignore buffer performance warning
  if (message.id() == 131186)
    return;

  // Ignore generic vertex attribute array 1 uses a pointer with a small value
  if (message.id() == 131076)
    return;

  switch (message.severity())
  {
  case QOpenGLDebugMessage::Severity::NotificationSeverity:
    qCInfo(openGlChan) << message;
    break;

  case QOpenGLDebugMessage::Severity::LowSeverity:
    qCDebug(openGlChan) << message;
    break;
  case QOpenGLDebugMessage::Severity::MediumSeverity:
    qCWarning(openGlChan) << message;
    break;
  case QOpenGLDebugMessage::Severity::HighSeverity:
    qCCritical(openGlChan) << message;
    throw std::runtime_error(message.message().toStdString());
    break;
  default:
    return;
  }
}

