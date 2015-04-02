#ifndef SRC_INPUT_SIGNAL_MANAGER_H_

#define SRC_INPUT_SIGNAL_MANAGER_H_

#include <QString>
#include <QObject>
#include <map>

/**
 * \brief Manages objects that have signals which can be
 * used to trigger SCXML transitions.
 *
 */
class SignalManager
{
public:
  SignalManager();
  virtual ~SignalManager();

  void addSender(QObject *sender);
  void addSender(QString name, QObject *sender);

  QObject* getFor(QString name);
private:
  std::map<QString, QObject*> senders;
};

#endif  // SRC_INPUT_SIGNAL_MANAGER_H_
