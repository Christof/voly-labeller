#ifndef SRC_PICKING_CONTROLLER_H_

#define SRC_PICKING_CONTROLLER_H_

#include <QObject>
#include <memory>
#include "./math/eigen.h"

class Scene;
class Label;

/**
 * \brief Triggers picking of 3d position on mouse click and set anchor position
 *
 */
class PickingController : public QObject
{
  Q_OBJECT
 public:
  explicit PickingController(std::shared_ptr<Scene> scene);

 public slots:
  void startPicking(Label *label);
  void pick(QEvent *event);

  void pickedPosition(Eigen::Vector3f position);

 private:
  std::shared_ptr<Scene> scene;
  Label *label;
};

#endif  // SRC_PICKING_CONTROLLER_H_
