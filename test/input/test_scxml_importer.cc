#include "../test.h"
#include <QUrl>
#include <QStateMachine>
#include <QCoreApplication>
#include <QtQuick/QQuickView>
#include <QKeyEvent>
#include <thread>
#include <chrono>
#include "../../src/input/scxml_importer.h"

class Test_ScxmlImporter : public ::testing::Test
{
 protected:
  virtual void SetUp()
  {
    int zero = 0;
    application = new QCoreApplication(zero, static_cast<char **>(nullptr));
    eventSender = new QObject();

    invokeManager = std::shared_ptr<InvokeManager>(new InvokeManager());
    ScxmlImporter importer(QUrl::fromLocalFile("../config/simple_state.xml"),
                           eventSender, invokeManager);
    stateMachine = importer.getStateMachine();

    stateMachine->start();

    application->processEvents(QEventLoop::ExcludeSocketNotifiers);
  }

  virtual void TearDown()
  {
    delete eventSender;
    delete application;
  }

  QAbstractState* expectSingleStateWithName(std::string name)
  {
    auto configuration = stateMachine->configuration();
    EXPECT_EQ(1, configuration.size());
    auto state = (*configuration.begin());
    EXPECT_EQ(name, state->property("name").toString().toStdString());

    return state;
  }

  void sendKeyPressEvent(Qt::Key key)
  {
    application->sendEvent(
        eventSender, new QKeyEvent(QEvent::KeyPress, key, Qt::NoModifier));
    application->processEvents();
  }

  QCoreApplication *application;
  QObject *eventSender;
  ScxmlImporter *importer;
  std::shared_ptr<QStateMachine> stateMachine;
  std::shared_ptr<InvokeManager> invokeManager;
};

TEST_F(Test_ScxmlImporter, SwitchFromInitialToFinalStateWithKeyPress)
{
  EXPECT_TRUE(stateMachine->isRunning());
  expectSingleStateWithName("idle");

  sendKeyPressEvent(Qt::Key_B);

  expectSingleStateWithName("exit");
  EXPECT_FALSE(stateMachine->isRunning());
}

class MockHandler : public QObject
{
  Q_OBJECT
 public slots:
  void printCurrentState()
  {
    wasPrintCurrentStateCalled = true;
  }

 public:
  bool wasPrintCurrentStateCalled = false;
};

TEST_F(Test_ScxmlImporter,
       SwitchFromInitialToAnotherStateAndInvokeWithStateChange)
{
  MockHandler handler;
  invokeManager->addHandler("Window", &handler);

  sendKeyPressEvent(Qt::Key_V);

  expectSingleStateWithName("base");

  sendKeyPressEvent(Qt::Key_Y);

  expectSingleStateWithName("idle");
  EXPECT_TRUE(handler.wasPrintCurrentStateCalled);
}

TEST_F(Test_ScxmlImporter,
       SwitchFromInitialToAnotherStateAndInvokeWithoutStateChange)
{
  MockHandler handler;
  invokeManager->addHandler("Window", &handler);

  sendKeyPressEvent(Qt::Key_V);

  expectSingleStateWithName("base");

  sendKeyPressEvent(Qt::Key_N);

  expectSingleStateWithName("base");
  EXPECT_TRUE(handler.wasPrintCurrentStateCalled);
}

TEST_F(Test_ScxmlImporter, SwitchToNestedStateWithInitialElement)
{
  sendKeyPressEvent(Qt::Key_N);

  auto configuration = stateMachine->configuration();
  EXPECT_EQ(2, configuration.size());

  for (auto state : configuration)
  {
    auto name = state->property("name").toString().toStdString();
    EXPECT_TRUE(name == "nested" || name == "with-nesting");
  }
}

TEST_F(Test_ScxmlImporter, SwitchToNestedStateWithInitialElementAndInvoke)
{
  MockHandler handler;
  invokeManager->addHandler("Window", &handler);

  sendKeyPressEvent(Qt::Key_I);

  auto configuration = stateMachine->configuration();
  EXPECT_EQ(2, configuration.size());

  for (auto state : configuration)
  {
    auto name = state->property("name").toString().toStdString();
    EXPECT_TRUE(name == "nested-invoke" || name == "with-nesting-invoke");
  }

  EXPECT_TRUE(handler.wasPrintCurrentStateCalled);
}

TEST_F(Test_ScxmlImporter, OnEntryInNestedState)
{
  MockHandler handler;
  invokeManager->addHandler("Window", &handler);

  sendKeyPressEvent(Qt::Key_E);

  expectSingleStateWithName("on-entry");

  EXPECT_TRUE(handler.wasPrintCurrentStateCalled);
}

#include "test_scxml_importer.moc"
