#include "../test.h"
#include <QUrl>
#include <QStateMachine>
#include <QGuiApplication>
#include <QtQuick/QQuickView>
#include <QKeyEvent>
#include <thread>
#include <chrono>
#include "../../src/input/scxml_importer.h"

/**
 * \brief
 *
 *
 */
class MyWindow : public QQuickView
{
 public:
  MyWindow() : QQuickView(nullptr){};
  virtual ~MyWindow() {
  }

 signals:
  void keyPressed(QKeyEvent *e) {Q_UNUSED(e)}

 protected:
  inline void keyPressEvent(QKeyEvent * e) Q_DECL_OVERRIDE
  {
    std::cout << "key event: " << e->text().toStdString() << std::endl;
    emit keyPressed(e);
    //QQuickView::event(e);
  }
};

TEST(Test_ScxmlImporter, foo)
{
  int zero = 0;
  QGuiApplication application(zero, static_cast<char **>(nullptr));
  MyWindow window;
  window.setSource(QUrl("qrc:ui.qml"));

  window.show();

  auto invokeManager = std::shared_ptr<InvokeManager>(new InvokeManager());
  ScxmlImporter importer(QUrl::fromLocalFile("../config/simple_state.xml"),
                         &window, invokeManager);
  auto stateMachine = importer.getStateMachine();
  std::cout << "running:" << stateMachine->isRunning() << std::endl;

  stateMachine->start();

  std::cout << "running:" << stateMachine->isRunning() << std::endl;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  std::cout << "error:" << stateMachine->errorString().toStdString()
            << std::endl;
  std::cout << "running:" << stateMachine->isRunning() << std::endl;

  stateMachine->postEvent(
      new QKeyEvent(QEvent::KeyPress, Qt::Key_A, Qt::NoModifier));

  application.exec();
}

