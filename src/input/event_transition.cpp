#include "./event_transition.h"
#include <QStateMachine>
#include <QEvent>

EventTransition::EventTransition(QObject *object, QEvent::Type type,
                                 QState *sourceState)
  : QEventTransition(object, type, sourceState)
{
}

EventTransition::~EventTransition()
{
}

void EventTransition::onTransition(QEvent *event)
{
  QStateMachine::WrappedEvent *we =
      static_cast<QStateMachine::WrappedEvent *>(event);
  this->event = we->event();
}
