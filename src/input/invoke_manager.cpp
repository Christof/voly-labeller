#include "./invoke_manager.h"
#include <QAbstractTransition>
#include <QAbstractState>
#include <QVariant>
#include <iostream>

InvokeManager::InvokeManager()
{
}

InvokeManager::~InvokeManager()
{
}

void InvokeManager::addFor(QAbstractTransition *transition, QString targetType,
                           QString source)
{
  invokes[transition].push_back(Invoke(targetType, source));

  connect(transition, &QAbstractTransition::triggered,
          std::bind(&InvokeManager::invokeFor, this, transition));
}

void InvokeManager::invokeFor(QAbstractTransition *transition)
{
  for (auto &invoke : invokes[transition])
  {
    std::cout << invoke.targetType.toStdString()
              << " src=" << invoke.source.toStdString();
    auto object = handlers[invoke.targetType];
    std::cout << "handler size: " << handlers.size() << std::endl;
    std::cout << (long)object << std::endl;

    QMetaObject::invokeMethod(object, invoke.source.toStdString().c_str(),
                              Qt::AutoConnection);
  }
}

void InvokeManager::addHandler(QObject *handlerObject)
{
  auto className = handlerObject->metaObject()->className();
  std::cout << "Add handler " << className << std::endl;
  handlers[className] = handlerObject;
}

