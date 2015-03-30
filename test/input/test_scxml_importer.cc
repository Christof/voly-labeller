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

    auto invokeManager = std::shared_ptr<InvokeManager>(new InvokeManager());
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

  QCoreApplication *application;
  QObject *eventSender;
  std::shared_ptr<QStateMachine> stateMachine;
};

TEST_F(Test_ScxmlImporter, SwitchFromInitialToFinalStateWithKeyPress)
{
  EXPECT_TRUE(stateMachine->isRunning());

  auto configuration = stateMachine->configuration();
  EXPECT_EQ(1, configuration.size());
  EXPECT_EQ("idle", (*configuration.begin())->property("name").toString());

  application->sendEvent(
      eventSender, new QKeyEvent(QEvent::KeyPress, Qt::Key_B, Qt::NoModifier));
  application->processEvents();

  configuration = stateMachine->configuration();
  EXPECT_EQ(1, configuration.size());
  EXPECT_EQ("exit", (*configuration.begin())->property("name").toString());

  EXPECT_FALSE(stateMachine->isRunning());
}

