#include "./window.h"
#include <iostream>
#include <QOpenGLContext>
#include <QCoreApplication>
#include <QKeyEvent>
#include <QStateMachine>
#include <QAbstractState>
#include <QAbstractTransition>
#include "./abstract_scene.h"
#include "./input/scxml_importer.h"

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

  ScxmlImporter importer(QUrl::fromLocalFile("../config/simple_state.xml"),
                         this);
  stateMachine = importer.getStateMachine();
  stateMachine->start();

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
  gl = context->versionFunctions<Gl>();
  if (!gl)
  {
    qWarning() << "Could not obtain required OpenGL context version";
    exit(1);
  }
  gl->initializeOpenGLFunctions();
  glCheckError();

  gl->glEnable(GL_DEPTH_TEST);
}

void Window::keyReleaseEvent(QKeyEvent *event)
{
  keysPressed -= static_cast<Qt::Key>(event->key());
}

void Window::keyPressEvent(QKeyEvent *event)
{
  keysPressed += static_cast<Qt::Key>(event->key());
}

void Window::handleLazyInitialization()
{
  static bool initialized = false;
  if (!initialized)
  {
    initializeOpenGL();

    scene->setContext(context, gl);
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
  scene->resize(width(), height());
}

void Window::update()
{
  scene->update(timer.restart() / 1000.0, keysPressed);
}

void Window::printCurrentState()
{
  auto currentState = stateMachine->configuration();
  std::cout << "current state(s): " << std::endl;
  for (auto state : currentState)
  {
    std::cout << "\t" << state->property("name").toString().toStdString()
              << std::endl;
    auto castState = dynamic_cast<QState *>(state);
    if (!castState)
      continue;

    auto transitions = castState->transitions();
    std::cout << "\t\ttransition count: " << transitions.length() << std::endl;
    for (auto transition : transitions)
    {
      std::cout << "\t\tsource: "
                << transition->sourceState()
                       ->property("name")
                       .toString()
                       .toStdString() << std::endl;
      auto targetState = transition->targetState();
      std::cout << "\t\ttarget: "
                << targetState->property("name").toString().toStdString()
                << std::endl;
    }
  }
}

