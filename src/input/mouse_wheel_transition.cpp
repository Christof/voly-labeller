#include "./mouse_wheel_transition.h"
#include <QDebug>

MouseWheelTransition::MouseWheelTransition(QObject *object, QState *sourceState)
  : QEventTransition(object, QEvent::Wheel, sourceState)
{
}

MouseWheelTransition::~MouseWheelTransition()
{
}

bool MouseWheelTransition::eventTest(QEvent *event)
{
  qWarning() << "eventTest" << event->type();
  return event->type() == QEvent::Scroll;
}

void MouseWheelTransition::onTransition(QEvent *event)
{
  this->event = event;
}
