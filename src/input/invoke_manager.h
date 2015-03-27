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

 private:
  std::map<QAbstractTransition *, std::vector<Invoke>> invokes;
};

#endif  // SRC_INPUT_INVOKE_MANAGER_H_
