#ifndef SRC_SCENE_H_

#define SRC_SCENE_H_

#include <functional>
#include "./utils/memory.h"
#include "./abstract_scene.h"
#include "./camera.h"
#include "./frustum_optimizer.h"
#include "./forces/labeller.h"
#include "./placement/labeller.h"
#include "./graphics/screen_quad.h"
#include "./graphics/frame_buffer_object.h"
#include "./graphics/ha_buffer.h"
#include "./graphics/object_manager.h"
#include "./placement/cuda_texture_mapper.h"
#include "./graphics/managers.h"
#include "./graphics/standard_texture_2d.h"
#include "./picker.h"
#include "./labelling/clustering.h"

class Nodes;
class InvokeManager;
class CameraControllers;
class TextureMapperManager;
class ConstraintBufferObject;
class Labels;
class PersistentConstraintUpdater;

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

  void enableBufferDebuggingViews(bool enable);
  void enableConstraingOverlay(bool enable);
  void setRenderLayer(int layerNumber);

 private:
  std::shared_ptr<CameraControllers> cameraControllers;
  double frameTime;

  std::shared_ptr<Nodes> nodes;
  std::shared_ptr<Labels> labels;
  std::shared_ptr<Forces::Labeller> forcesLabeller;
  std::vector<std::shared_ptr<Placement::Labeller>> placementLabellers;
  std::vector<std::shared_ptr<LabelsContainer>> labelsInLayer;
  std::shared_ptr<Graphics::ScreenQuad> quad;
  std::shared_ptr<Graphics::ScreenQuad> screenQuad;
  std::shared_ptr<Graphics::ScreenQuad> positionQuad;
  std::shared_ptr<Graphics::ScreenQuad> distanceTransformQuad;
  std::shared_ptr<Graphics::ScreenQuad> transparentQuad;
  std::shared_ptr<Graphics::FrameBufferObject> fbo;
  std::shared_ptr<ConstraintBufferObject> constraintBufferObject;
  std::shared_ptr<PersistentConstraintUpdater> persistentConstraintUpdater;
  std::shared_ptr<Graphics::HABuffer> haBuffer;
  std::shared_ptr<Graphics::Managers> managers;
  std::unique_ptr<Picker> picker;
  FrustumOptimizer frustumOptimizer;
  Clustering clustering;

  int width;
  int height;
  bool shouldResize = false;
  bool showBufferDebuggingViews = false;
  bool showConstraintOverlay = false;
  int activeLayerNumber = 0;

  void updateLabelling();

  void renderNodesWithHABufferIntoFBO(const RenderData &renderData);
  void renderQuad(std::shared_ptr<Graphics::ScreenQuad> quad,
                  Eigen::Matrix4f modelMatrix);
  void renderScreenQuad();

  void renderDebuggingViews(const RenderData &renderData);
  RenderData createRenderData();
  std::shared_ptr<Camera> getCamera();

  std::shared_ptr<TextureMapperManager> textureMapperManager;
};

#endif  // SRC_SCENE_H_
