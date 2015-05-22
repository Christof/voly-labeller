#ifndef SRC_INPUT_EVENT_TRANSITION_H_

#define SRC_INPUT_EVENT_TRANSITION_H_

#include <QEventTransition>

/**
 * \brief
 *
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
