#ifndef SRC_INPUT_SCXML_IMPORTER_H_

#define SRC_INPUT_SCXML_IMPORTER_H_

#include <QUrl>
#include <QString>
#include <QObject>
#include <QMetaEnum>
#include <QStack>
#include <memory>
#include <map>
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
 * \brief Import a SCXML file and converts it into a QStateMachine.
 *
 * The given InvokeManager is used to call arbitrary slots.
 * The given SignalManager is used to react to arbitrary signals.
 *
 * To handle events from the keyboards a `"KeyboardEventSender"` must
 * be registered in the SignalManager.
 */
class ScxmlImporter : public QObject
{
  Q_OBJECT
  Q_ENUMS(ScxmlElement)
 public:
  ScxmlImporter(QUrl url, std::shared_ptr<InvokeManager> invokeManager,
                std::shared_ptr<SignalManager> signalManager);
  virtual ~ScxmlImporter();

  std::shared_ptr<QStateMachine> import();

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
  QUrl url;
  QString initialState;
  QMetaEnum metaScxmlElement;
  QMetaEnum metaEventType;
  QStack<ScxmlElement> elementStack;
  QStack<QState *> stateStack;
  QAbstractTransition *currentTransition;
  std::unique_ptr<QXmlStreamReader> reader;
  std::shared_ptr<QStateMachine> stateMachine;
  std::map<QString, QAbstractState *> states;
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

  void parse();
  void addTransitions();
  void setInitialStates();
  QAbstractTransition *createKeyEventTransition(const QString &event);
  QAbstractTransition *createMouseButtonEventTransition(const QString &event);
  QAbstractTransition *createEventTransition(const QString &event);
  QAbstractTransition *createSignalTransition(const QString &event);
  QString attributeAsString(const char *name);
  ScxmlElement elementFromString(QString name);
};

#endif  // SRC_INPUT_SCXML_IMPORTER_H_
