#ifndef SRC_INPUT_SCXML_IMPORTER_H_

#define SRC_INPUT_SCXML_IMPORTER_H_

#include <QUrl>
#include <QString>
#include <QObject>
#include <memory>
#include <map>
#include <vector>

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
  ScxmlImporter(QUrl url, QObject *keyboardEventReceiver);
  virtual ~ScxmlImporter();

  std::shared_ptr<QStateMachine> getStateMachine();

 private:
  QObject *keyboardEventReceiver;
  QString initialState;
  QState *state;
  std::unique_ptr<QXmlStreamReader> reader;
  std::shared_ptr<QStateMachine> stateMachine;
  std::map<QString, QAbstractState *> states;
  std::vector<QAbstractTransition *> transitions;

  void readElement();
  void readState();
  void readTransition();

  QString attributeAsString(const char *name);
};

#endif  // SRC_INPUT_SCXML_IMPORTER_H_
