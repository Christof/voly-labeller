#ifndef SRC_PICKING_CONTROLLER_H_

#define SRC_PICKING_CONTROLLER_H_

#include <QObject>
#include <memory>

class Scene;

/**
 * \brief
 *
 *
 */
class PickingController : public QObject
{
  Q_OBJECT
 public:
  explicit PickingController(std::shared_ptr<Scene> scene);

 public slots:
  void pick(QEvent *event);

 private:
  std::shared_ptr<Scene> scene;
};

#endif  // SRC_PICKING_CONTROLLER_H_
