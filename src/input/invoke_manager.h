#ifndef SRC_INPUT_INVOKE_MANAGER_H_

#define SRC_INPUT_INVOKE_MANAGER_H_

#include <QString>
#include <QObject>
#include <functional>
#include <map>
#include <vector>
#include "./invoke.h"


#include "./mouse_wheel_transition.h"

class QAbstractTransition;
class QAbstractState;

/**
 * \brief Manages SCXML invoke calls by storing handler objects which
 * have methods that are invoked by state transitions.
 *
 */
class InvokeManager : public QObject
{
  Q_OBJECT
 public:
  InvokeManager();
  virtual ~InvokeManager();

  void addFor(QAbstractTransition *transition, QString targetType,
              QString source);

  template <typename Signal>
  void addForSignal(const QAbstractState *sender, Signal signal,
                    QString targetType, QString source)
  {
    connect(sender, signal,
            std::bind(&InvokeManager::invokeMethod, this, targetType, source));
  }

  void invokeFor(QAbstractTransition *transition);
  void invokeForWithArg(MouseWheelTransition *transition);

  void addHandler(QObject *handlerObject);
  void addHandler(QString targetType, QObject *handlerObject);

 private:
  std::map<QAbstractTransition *, std::vector<Invoke>> invokes;
  std::map<QString, QObject *> handlers;

  void invokeMethod(QString targetType, QString source);
};

#endif  // SRC_INPUT_INVOKE_MANAGER_H_
