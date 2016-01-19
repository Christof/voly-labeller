#ifndef SRC_MOUSE_DRAGGING_CONTROLLER_H_

#define SRC_MOUSE_DRAGGING_CONTROLLER_H_

#include <QObject>
#include <memory>
#include "./math/eigen.h"

class Camera;

/**
 * \brief Base class for controllers which are based on mouse dragging
 *
 * It is enabled via the state machine, which calls the slots.
 * The mouse position is gathered using QCursor.
 *
 * MouseDraggingController::setFrameTime must be called each frame
 * to ensure a steady camera speed.
 */
class MouseDraggingController : public QObject
{
  Q_OBJECT
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MouseDraggingController(std::shared_ptr<Camera> camera, double speedFactor);

  void update(std::shared_ptr<Camera> camera, double frameTime);

 public slots:
  void startDragging();
  void updateDragging(QEvent *event);

 protected:
  virtual void updateFromDiff(Eigen::Vector2f diff) = 0;

  std::shared_ptr<Camera> camera;
  Eigen::Vector2f lastMousePosition;
  double frameTime;
  double speedFactor;

 private:
  bool start = false;
};

#endif  // SRC_MOUSE_DRAGGING_CONTROLLER_H_
