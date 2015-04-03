#include <QtGui/QGuiApplication>
#include <QQmlContext>
#include <QStateMachine>
#include <memory>
#include "./window.h"
#include "./demo_scene.h"
#include "./input/invoke_manager.h"
#include "./input/signal_manager.h"
#include "./input/scxml_importer.h"

int main(int argc, char **argv)
{
  QGuiApplication application(argc, argv);

  auto scene = std::make_shared<DemoScene>();
  Window window(scene);
  window.rootContext()->setContextProperty("window", &window);
  window.setSource(QUrl("qrc:ui.qml"));

  auto invokeManager = std::shared_ptr<InvokeManager>(new InvokeManager());
  auto signalManager = std::shared_ptr<SignalManager>(new SignalManager());
  ScxmlImporter importer(QUrl::fromLocalFile("../config/simple_state.xml"),
                         invokeManager, signalManager);

  invokeManager->addHandler(&window);
  signalManager->addSender("KeyboardEventSender", &window);

  auto stateMachine = importer.import();

  // just for printCurrentState slot for debugging
  window.stateMachine = stateMachine;

  stateMachine->start();

  window.show();

  return application.exec();
}
