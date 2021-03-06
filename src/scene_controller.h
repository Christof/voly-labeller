#ifndef SRC_SCENE_CONTROLLER_H_

#define SRC_SCENE_CONTROLLER_H_

#include <QObject>
#include <memory>

class Scene;

/**
 * \brief Wrapper around Scene to provide access to its settings for the UI
 *
 */
class SceneController : public QObject
{
  Q_OBJECT
 public:
  explicit SceneController(std::shared_ptr<Scene> scene);

 public slots:
  void toggleBufferViews();
  void toggleConstraintOverlay();
  void toggleLabelling();
  void compositeLayers();
  void renderFirstLayer();
  void renderSecondLayer();
  void renderThirdLayer();
  void renderFourthLayer();
  void accumulatedLayers();

 private:
  std::shared_ptr<Scene> scene;

  bool showBufferDebuggingViews = false;
  bool showConstraintOverlay = false;
  bool labellingEnabled = true;
};

#endif  // SRC_SCENE_CONTROLLER_H_
