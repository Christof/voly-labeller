#include "../test.h"
#include <QUrl>
#include <QStateMachine>
#include <QCoreApplication>
#include <QtQuick/QQuickView>
#include <QKeyEvent>
#include <thread>
#include <chrono>
#include "../../src/input/scxml_importer.h"

class SignalSender : public QObject
{
  Q_OBJECT

signals:
  void signal();
};

class Test_ScxmlImporter : public ::testing::Test
{
 protected:
  virtual void SetUp()
  {
    int zero = 0;
    application = new QCoreApplication(zero, static_cast<char **>(nullptr));
    eventSender = new QObject();

    invokeManager = std::shared_ptr<InvokeManager>(new InvokeManager());
    signalManager = std::shared_ptr<SignalManager>(new SignalManager());
    signalManager->addSender(&sender);
    signalManager->addSender("KeyboardEventSender", eventSender);
    ScxmlImporter importer(QUrl::fromLocalFile("../config/simple_state.xml"),
                           invokeManager, signalManager);
    stateMachine = importer.import();

    stateMachine->start();

    application->processEvents(QEventLoop::ExcludeSocketNotifiers);
  }

  virtual void TearDown()
  {
    delete eventSender;
    delete application;
  }

  QAbstractState *expectSingleStateWithName(std::string name)
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

  void sendMouseButtonPressEvent(Qt::MouseButton button)
  {
    application->sendEvent(
        eventSender, new QMouseEvent(QEvent::MouseButtonPress, QPointF(0, 0),
                                     button, button, Qt::NoModifier));
    application->processEvents();
  }

  void sendKeyReleaseEvent(Qt::Key key)
  {
    application->sendEvent(
        eventSender, new QKeyEvent(QEvent::KeyRelease, key, Qt::NoModifier));
    application->processEvents();
  }

  QCoreApplication *application;
  QObject *eventSender;
  ScxmlImporter *importer;
  SignalSender sender;
  std::shared_ptr<QStateMachine> stateMachine;
  std::shared_ptr<InvokeManager> invokeManager;
  std::shared_ptr<SignalManager> signalManager;
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
    printCurrentStateCallCount++;
  }

  void somethingElse()
  {
    somethingElseCallCount++;
  }

 public:
  int printCurrentStateCallCount = 0;
  int somethingElseCallCount = 0;
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
  EXPECT_TRUE(handler.printCurrentStateCallCount);
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
  EXPECT_TRUE(handler.printCurrentStateCallCount);
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

  EXPECT_TRUE(handler.printCurrentStateCallCount);
}

TEST_F(Test_ScxmlImporter, OnEntryInNestedState)
{
  MockHandler handler;
  invokeManager->addHandler("Window", &handler);

  sendKeyPressEvent(Qt::Key_E);

  expectSingleStateWithName("on-entry");

  EXPECT_TRUE(handler.printCurrentStateCallCount);
}

TEST_F(Test_ScxmlImporter, MultipleInvokesInOneTransition)
{
  MockHandler handler;
  invokeManager->addHandler("Window", &handler);

  sendKeyPressEvent(Qt::Key_J);

  EXPECT_EQ(1, handler.printCurrentStateCallCount);
  EXPECT_EQ(1, handler.somethingElseCallCount);
}

TEST_F(Test_ScxmlImporter, OnEntryWithMultipleInvokes)
{
  MockHandler handler;
  invokeManager->addHandler("Window", &handler);

  sendKeyPressEvent(Qt::Key_F);

  EXPECT_EQ(1, handler.printCurrentStateCallCount);
  EXPECT_EQ(1, handler.somethingElseCallCount);
}

TEST_F(Test_ScxmlImporter, OnExitInNestedState)
{
  MockHandler handler;
  invokeManager->addHandler("Window", &handler);

  sendKeyPressEvent(Qt::Key_G);

  expectSingleStateWithName("on-exit");

  sendKeyPressEvent(Qt::Key_G);

  EXPECT_EQ(1, handler.somethingElseCallCount);
}

TEST_F(Test_ScxmlImporter, TransitionOnKeyReleaseEvent)
{
  expectSingleStateWithName("idle");
  sendKeyPressEvent(Qt::Key_A);
  expectSingleStateWithName("idle");

  sendKeyReleaseEvent(Qt::Key_A);

  expectSingleStateWithName("exit");
}

TEST_F(Test_ScxmlImporter, TransitionOnMousePressEvent)
{
  expectSingleStateWithName("idle");

  sendMouseButtonPressEvent(Qt::MouseButton::LeftButton);

  expectSingleStateWithName("exit");
}

TEST_F(Test_ScxmlImporter, EventFromCustomSignal)
{
  MockHandler handler;
  invokeManager->addHandler("Window", &handler);
  expectSingleStateWithName("idle");

  emit sender.signal();

  expectSingleStateWithName("exit");
  EXPECT_EQ(1, handler.somethingElseCallCount);
}

TEST_F(Test_ScxmlImporter, StateChangeWithControlKey)
{
  expectSingleStateWithName("idle");

  sendKeyPressEvent(Qt::Key_Alt);

  expectSingleStateWithName("alt");
}

#include "test_scxml_importer.moc"
