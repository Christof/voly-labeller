#ifndef SRC_INPUT_EVENT_TRANSITION_H_

#define SRC_INPUT_EVENT_TRANSITION_H_

#include <QEventTransition>

/**
 * \brief Transition triggered on given QEvent::Type which stores the triggering
 * event so that the values can be used in an invoke slot
 *
 */
class EventTransition : public QEventTransition
{
 public:
  EventTransition(QObject *object, QEvent::Type type, QState *sourceState = 0);
  virtual ~EventTransition();

  QEvent *event;

 protected:
  void onTransition(QEvent *event);
};

#endif  // SRC_INPUT_EVENT_TRANSITION_H_
