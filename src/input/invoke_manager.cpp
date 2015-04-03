#include "./invoke_manager.h"
#include <QAbstractTransition>
#include <QAbstractState>
#include <QVariant>

InvokeManager::InvokeManager()
{
}

InvokeManager::~InvokeManager()
{
}

void InvokeManager::addFor(QAbstractTransition *transition, QString targetType,
                           QString source)
{
  if (invokes.count(transition) == 0)
    connect(transition, &QAbstractTransition::triggered,
            std::bind(&InvokeManager::invokeFor, this, transition));

  invokes[transition].push_back(Invoke(targetType, source));
}

void InvokeManager::invokeFor(QAbstractTransition *transition)
{
  for (auto &invoke : invokes[transition])
  {
    invokeMethod(invoke.targetType, invoke.source);
  }
}

void InvokeManager::addHandler(QString targetType, QObject *handlerObject)
{
  handlers[targetType] = handlerObject;
}

void InvokeManager::addHandler(QObject *handlerObject)
{
  auto className = handlerObject->metaObject()->className();
  addHandler(className, handlerObject);
}

void InvokeManager::invokeMethod(QString targetType, QString source)
{
  auto object = handlers[targetType];
  QMetaObject::invokeMethod(object, source.toStdString().c_str(),
                            Qt::AutoConnection);
}

