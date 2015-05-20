#include "./mouse_wheel_transition.h"
#include <QStateMachine>

MouseWheelTransition::MouseWheelTransition(QObject *object, QState *sourceState)
  : QEventTransition(object, QEvent::Wheel, sourceState)
{
}

MouseWheelTransition::~MouseWheelTransition()
{
}

void MouseWheelTransition::onTransition(QEvent *event)
{
  QStateMachine::WrappedEvent *we = static_cast<QStateMachine::WrappedEvent*>(event);
  this->event = we->event();
}
