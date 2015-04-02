#ifndef SRC_INPUT_SCXML_IMPORTER_H_

#define SRC_INPUT_SCXML_IMPORTER_H_

#include <QUrl>
#include <QString>
#include <QObject>
#include <QMetaEnum>
#include <QStack>
#include <memory>
#include <map>
#include <stack>
#include <vector>
#include <tuple>
#include "./invoke_manager.h"
#include "./signal_manager.h"

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
  Q_ENUMS(ScxmlElement)
 public:
  ScxmlImporter(QUrl url, QObject *keyboardEventReceiver,
                std::shared_ptr<InvokeManager> invokeManager,
                std::shared_ptr<SignalManager> signalManager);
  virtual ~ScxmlImporter();

  std::shared_ptr<QStateMachine> getStateMachine();

  enum ScxmlElement
  {
    scxml,
    state,
    transition,
    initial,
    onentry,
    onexit,
    invoke,
    final
  };
 private:
  QObject *keyboardEventReceiver;
  QString initialState;
  QMetaEnum metaScxmlElement;
  QStack<ScxmlElement> elementStack;
  std::stack<QState *> stateStack;
  QAbstractTransition *currentTransition;
  std::unique_ptr<QXmlStreamReader> reader;
  std::shared_ptr<QStateMachine> stateMachine;
  std::map<QString, QAbstractState *> states;
  bool isReadingInitial = false;
  std::map<QState *, QString> initialStateTransitions;
  // transition and target state name
  std::vector<std::tuple<QAbstractTransition *, QString>> transitions;
  std::shared_ptr<InvokeManager> invokeManager;
  std::shared_ptr<SignalManager> signalManager;

  void readElement();
  void finishElement();
  void readState();
  void readFinalState();
  void readTransition();
  void readInvoke();

  QString attributeAsString(const char *name);
  ScxmlElement elementFromString(QString name);
};

#endif  // SRC_INPUT_SCXML_IMPORTER_H_
