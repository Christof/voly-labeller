#include "./signal_manager.h"

SignalManager::SignalManager()
{
}

SignalManager::~SignalManager()
{
}

void SignalManager::addSender(QString name, QObject *sender)
{
  senders[name] = sender;
}

void SignalManager::addSender(QObject *sender)
{
  addSender(sender->metaObject()->className(), sender);
}

QObject* SignalManager::getFor(QString name)
{
  return senders[name];
}

