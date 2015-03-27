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
  std::cout << invokes.size() << std::endl;
  invokes[transition].push_back(Invoke(targetType, source));
  std::cout << "ADDING" << invokes.size() << std::endl;
  std::cout << "\tsize: " << invokes.size() << std::endl;
  std::cout << "\tinvokes size: " << invokes[transition].size() << std::endl;
  std::cout << (long)this << std::endl;

  connect(transition, &QAbstractTransition::triggered, [transition, this]()
          {
    std::cout << (long)this << std::endl;
    std::cout << "size: " << this->invokes.size() << std::endl;
    std::cout << "in callback value of transition to: "
              << transition->targetState()->property("name").toString().toStdString()
              << std::endl;
    this->invokeFor(transition);
  });
  /*
  connect(transition, &QAbstractTransition::triggered,
      std::bind(&InvokeManager::invokeFor, this, transition));
      */
}

void InvokeManager::invokeFor(QAbstractTransition *transition)
{
  std::cout << (long)this << std::endl;
  std::cout << "size: " << invokes.size() << std::endl;
  std::cout << "invokes size: " << invokes[transition].size() << std::endl;
  for (auto &invoke : invokes[transition])
    std::cout << invoke.targetType.toStdString()
              << " src=" << invoke.source.toStdString();
}

