#include "./mouse_wheel_transition.h"
#include <QStateMachine>
#include <QEvent>

MouseWheelTransition::MouseWheelTransition(QObject *object, QEvent::Type type,
                                           QState *sourceState)
  : QEventTransition(object, type, sourceState)
{
}

MouseWheelTransition::~MouseWheelTransition()
{
}

void MouseWheelTransition::onTransition(QEvent *event)
{
  QStateMachine::WrappedEvent *we =
      static_cast<QStateMachine::WrappedEvent *>(event);
  this->event = we->event();
}
