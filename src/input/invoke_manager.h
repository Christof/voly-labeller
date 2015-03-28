#ifndef SRC_INPUT_INVOKE_MANAGER_H_

#define SRC_INPUT_INVOKE_MANAGER_H_

#include <QString>
#include <QObject>
#include <map>
#include <vector>
#include "invoke.h"

class QAbstractTransition;

/**
 * \brief
 *
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

  void invokeFor(QAbstractTransition *transition);

  void addHandler(QObject* handlerObject);

 private:
  std::map<QAbstractTransition *, std::vector<Invoke>> invokes;
  std::map<QString, QObject*> handlers;
};

#endif  // SRC_INPUT_INVOKE_MANAGER_H_
