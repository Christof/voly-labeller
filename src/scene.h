#ifndef SRC_SCENE_H_

#define SRC_SCENE_H_

#include <memory>
#include <functional>
#include "./abstract_scene.h"
#include "./camera.h"
#include "./frustum_optimizer.h"
#include "./forces/labeller.h"
#include "./placement/labeller.h"
#include "./labelling/labels.h"
#include "./graphics/screen_quad.h"
#include "./graphics/frame_buffer_object.h"
#include "./graphics/ha_buffer.h"
#include "./graphics/object_manager.h"
#include "./placement/cuda_texture_mapper.h"
#include "./graphics/managers.h"
#include "./graphics/standard_texture_2d.h"

class Nodes;
class InvokeManager;
class CameraControllers;
class TextureMapperManager;

/**
 * \brief Default implementation of AbstractScene
 *
 * It uses Nodes to manage which nodes are visible.
 * In initialize a default scene is loaded for now
 * (and also persisted as `config/scene.xml`).
 */
class Scene : public AbstractScene
{
 public:
  Scene(std::shared_ptr<InvokeManager> invokeManager,
        std::shared_ptr<Nodes> nodes, std::shared_ptr<Labels> labels,
        std::shared_ptr<Forces::Labeller> forcesLabeller,
        std::shared_ptr<Placement::Labeller> placementLabeller,
        std::shared_ptr<TextureMapperManager> textureMapperManager);
  ~Scene();

  virtual void initialize();
  virtual void cleanup();
  virtual void update(double frameTime, QSet<Qt::Key> keysPressed);
  virtual void render();
  virtual void resize(int width, int height);

  void pick(int id, Eigen::Vector2f position);

 private:
  Camera camera;
  std::shared_ptr<CameraControllers> cameraControllers;
  double frameTime;

  std::shared_ptr<Nodes> nodes;
  std::shared_ptr<Labels> labels;
  std::shared_ptr<Forces::Labeller> forcesLabeller;
  std::shared_ptr<Placement::Labeller> placementLabeller;
  std::shared_ptr<Graphics::ScreenQuad> quad;
  std::shared_ptr<Graphics::ScreenQuad> positionQuad;
  std::shared_ptr<Graphics::FrameBufferObject> fbo;
  std::shared_ptr<Graphics::HABuffer> haBuffer;
  std::shared_ptr<Graphics::Managers> managers;
  FrustumOptimizer frustumOptimizer;

  int width;
  int height;
  bool shouldResize = false;

  void renderQuad(std::shared_ptr<Graphics::ScreenQuad> quad,
                  Eigen::Matrix4f modelMatrix);
  void renderScreenQuad();

  bool performPicking;
  Eigen::Vector2f pickingPosition;
  int pickingLabelId;
  void renderDebuggingViews(const RenderData &renderData);
  void doPick();

  std::shared_ptr<TextureMapperManager> textureMapperManager;
};

#endif  // SRC_SCENE_H_
