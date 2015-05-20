#ifndef SRC_INPUT_MOUSE_WHEEL_TRANSITION_H_

#define SRC_INPUT_MOUSE_WHEEL_TRANSITION_H_

#include <QEventTransition>

/**
 * \brief
 *
 *
 */
class MouseWheelTransition : public QEventTransition
{
 public:
  MouseWheelTransition(QObject *object, QEvent::Type type,
                       QState *sourceState = 0);
  virtual ~MouseWheelTransition();

  QEvent *event;

 protected:
  void onTransition(QEvent *event);

 private:
  /* data */
};

#endif  // SRC_INPUT_MOUSE_WHEEL_TRANSITION_H_
