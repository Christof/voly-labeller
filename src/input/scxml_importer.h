#ifndef SRC_INPUT_SCXML_IMPORTER_H_

#define SRC_INPUT_SCXML_IMPORTER_H_

#include <QUrl>
#include <QString>
#include <QObject>
#include <memory>
#include <map>
#include <stack>
#include <vector>
#include <tuple>
#include "./invoke_manager.h"

class QStateMachine;
class QAbstractState;
class QAbstractTransition;
class QXmlStreamReader;
class QState;

/**
 * \brief
 *
 *
 */
class ScxmlImporter : public QObject
{
  Q_OBJECT
 public:
  ScxmlImporter(QUrl url, QObject *keyboardEventReceiver,
                std::shared_ptr<InvokeManager> invokeManager);
  virtual ~ScxmlImporter();

  std::shared_ptr<QStateMachine> getStateMachine();

 private:
  QObject *keyboardEventReceiver;
  QString initialState;
  QString activeElement;
  std::stack<QState*> stateStack;
  QAbstractTransition *currentTransition;
  std::unique_ptr<QXmlStreamReader> reader;
  std::shared_ptr<QStateMachine> stateMachine;
  std::map<QString, QAbstractState *> states;
  bool isReadingInitial = false;
  std::map<QState*, QString> initialStateTransitions;
  // transition and target state name
  std::vector<std::tuple<QAbstractTransition *, QString>> transitions;
  std::shared_ptr<InvokeManager> invokeManager;

  void readElement();
  void finishElement();
  void readState();
  void readFinalState();
  void readTransition();
  void readInvoke();

  QString attributeAsString(const char *name);
};

#endif  // SRC_INPUT_SCXML_IMPORTER_H_
