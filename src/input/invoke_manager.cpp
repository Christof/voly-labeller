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
    std::cout << invoke.targetType.toStdString()
              << " src=" << invoke.source.toStdString();
}

