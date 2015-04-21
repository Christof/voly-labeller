#ifndef SRC_MOUSE_DRAGGING_CONTROLLER_H_

#define SRC_MOUSE_DRAGGING_CONTROLLER_H_

#include <QObject>
#include <Eigen/Core>

/**
 * \brief Base class for controllers which are based on mouse dragging
 *
 * MouseDraggingController::setFrameTime must be called each frame
 * to ensure a steady camera speed.
 */
class MouseDraggingController : public QObject
{
  Q_OBJECT
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MouseDraggingController() = default;

  void setFrameTime(double frameTime);

 public slots:
  void startDragging();
  void updateDragging();

 protected:
  virtual void update(Eigen::Vector2f diff) = 0;

  Eigen::Vector2f lastMousePosition;
  double frameTime;
};

#endif  // SRC_MOUSE_DRAGGING_CONTROLLER_H_
