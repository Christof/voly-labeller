#include "./mouse_wheel_transition.h"
#include <QDebug>

MouseWheelTransition::MouseWheelTransition(QObject *object, QState *sourceState)
  : QEventTransition(object, QEvent::Wheel, sourceState)
{
}

MouseWheelTransition::~MouseWheelTransition()
{
}

void MouseWheelTransition::onTransition(QEvent *event)
{
  this->event = event;
}
