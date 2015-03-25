#include <QtGui/QGuiApplication>
#include <QQmlContext>
#include <memory>
#include "./window.h"
#include "./demo_scene.h"

int main(int argc, char **argv)
{
  QGuiApplication application(argc, argv);
  auto scene = std::make_shared<DemoScene>();
  Window window(scene);
  window.rootContext()->setContextProperty("window", &window);
  window.setSource(QUrl("qrc:ui.qml"));

  window.show();

  return application.exec();
}
