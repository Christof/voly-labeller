#include "../test.h"
#include <QUrl>
#include <QStateMachine>
#include <QGuiApplication>
#include <QWindow>
#include <QKeyEvent>
#include <thread>
#include <chrono>
#include "../../src/input/scxml_importer.h"

TEST(Test_ScxmlImporter, foo)
{
  int zero = 0;
  QGuiApplication application(zero, static_cast<char **>(nullptr));
  QWindow window;

  ScxmlImporter importer(QUrl::fromLocalFile("../config/simple_state.xml"),
                         &window);
  auto stateMachine = importer.getStateMachine();
  std::cout << "running:" << stateMachine->isRunning() << std::endl;

  stateMachine->start();

  window.show();

  std::cout << "running:" << stateMachine->isRunning() << std::endl;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  std::cout << "error:" << stateMachine->errorString().toStdString() << std::endl;
  std::cout << "running:" << stateMachine->isRunning() << std::endl;

  stateMachine->postEvent(
      new QKeyEvent(QEvent::KeyPress, Qt::Key_A, Qt::NoModifier));

  application.exec();
}
