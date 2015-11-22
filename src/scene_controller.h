#ifndef SRC_SCENE_CONTROLLER_H_

#define SRC_SCENE_CONTROLLER_H_

#include <QObject>
#include <memory>

class Scene;

/**
 * \brief
 *
 *
 */
class SceneController : public QObject
{
  Q_OBJECT
 public:
  SceneController(std::shared_ptr<Scene> scene);

 public slots:
  void toggleBufferViews();
  void toggleConstraintOverlay();

 private:
  std::shared_ptr<Scene> scene;

  bool showBufferDebuggingViews = false;
  bool showConstraintOverlay = false;
};

#endif  // SRC_SCENE_CONTROLLER_H_
