#include <QtGui/QGuiApplication>
#include <QQmlContext>
#include <QStateMachine>
#include <QDebug>
#include <memory>
#include "./window.h"
#include "./demo_scene.h"
#include "./nodes.h"
#include "./input/invoke_manager.h"
#include "./input/signal_manager.h"
#include "./input/scxml_importer.h"
#include "./mouse_shape_controller.h"

int main(int argc, char **argv)
{
  qputenv("QT_MESSAGE_PATTERN",
          QString("%{time [yyyy'-'MM'-'dd' 'hh':'mm':'ss]} - %{threadid} "
                  "%{if-category}%{category}: %{endif}%{message}").toUtf8());
  if (qgetenv("QT_LOGGING_CONF").size() == 0)
    qputenv("QT_LOGGING_CONF", "../config/logging.ini");

  QGuiApplication application(argc, argv);

  auto invokeManager = std::shared_ptr<InvokeManager>(new InvokeManager());
  auto nodes = std::make_shared<Nodes>();
  auto scene = std::make_shared<DemoScene>(invokeManager, nodes);

  Window window(scene);
  window.rootContext()->setContextProperty("window", &window);
  window.rootContext()->setContextProperty("nodes", nodes.get());
  window.setSource(QUrl("qrc:ui.qml"));

  MouseShapeController mouseShapeController(window);

  auto signalManager = std::shared_ptr<SignalManager>(new SignalManager());
  ScxmlImporter importer(QUrl::fromLocalFile("../config/states.xml"),
                         invokeManager, signalManager);

  invokeManager->addHandler(&window);
  invokeManager->addHandler("mouseShape", &mouseShapeController);
  signalManager->addSender("KeyboardEventSender", &window);

  auto stateMachine = importer.import();

  // just for printCurrentState slot for debugging
  window.stateMachine = stateMachine;

  stateMachine->start();

  window.show();

  return application.exec();
}
