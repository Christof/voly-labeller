#include "./mouse_wheel_transition.h"

MouseWheelTransition::MouseWheelTransition(QObject *object, QState *sourceState)
  : QEventTransition(object, QEvent::Scroll, sourceState)
{
}

MouseWheelTransition::~MouseWheelTransition()
{
}

bool MouseWheelTransition::eventTest(QEvent *event)
{
  return event->type() == QEvent::Scroll;
}

void MouseWheelTransition::onTransition(QEvent *event)
{
  this->event = event;
}
