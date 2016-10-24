#ifndef SRC_SCENE_H_

#define SRC_SCENE_H_

#include <functional>
#include <vector>
#include "./utils/memory.h"
#include "./abstract_scene.h"
#include "./frustum_optimizer.h"
#include "./graphics/screen_quad.h"
#include "./graphics/frame_buffer_object.h"
#include "./graphics/ha_buffer.h"
#include "./graphics/object_manager.h"
#include "./graphics/managers.h"
#include "./picker.h"

class Nodes;
class InvokeManager;
class Camera;
class CameraControllers;
class TextureMapperManager;
class ConstraintBufferObject;
class Labels;
class LabellingCoordinator;
class RecordingAutomation;

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
  Scene(int layerCount, std::shared_ptr<InvokeManager> invokeManager,
        std::shared_ptr<Nodes> nodes, std::shared_ptr<Labels> labels,
        std::shared_ptr<LabellingCoordinator> labellingCoordinator,
        std::shared_ptr<TextureMapperManager> textureMapperManager,
        std::shared_ptr<RecordingAutomation> recordingAutomationController);
  ~Scene();

  virtual void initialize();
  virtual void cleanup();
  virtual void update(double frameTime);
  virtual void render();
  virtual void resize(int width, int height);

  void pick(int id, Eigen::Vector2f position);

  void enableBufferDebuggingViews(bool enable);
  void enableConstraingOverlay(bool enable);
  void enableLabelling(bool enable);
  void setRenderLayer(int layerNumber);

 private:
  std::shared_ptr<CameraControllers> cameraControllers;
  double frameTime;

  std::shared_ptr<Nodes> nodes;
  std::shared_ptr<Labels> labels;
  std::shared_ptr<Graphics::ScreenQuad> quad;
  std::shared_ptr<Graphics::ScreenQuad> screenQuad;
  std::shared_ptr<Graphics::ScreenQuad> positionQuad;
  std::shared_ptr<Graphics::ScreenQuad> distanceTransformQuad;
  std::shared_ptr<Graphics::ScreenQuad> transparentQuad;
  std::shared_ptr<Graphics::ScreenQuad> sliceQuad;
  std::shared_ptr<Graphics::FrameBufferObject> fbo;
  std::shared_ptr<ConstraintBufferObject> constraintBufferObject;
  std::shared_ptr<Graphics::HABuffer> haBuffer;
  std::shared_ptr<Graphics::Managers> managers;
  std::shared_ptr<LabellingCoordinator> labellingCoordinator;
  std::unique_ptr<Picker> picker;
  FrustumOptimizer frustumOptimizer;

  int width;
  int height;
  bool shouldResize = false;
  bool showBufferDebuggingViews = false;
  bool showConstraintOverlay = false;
  bool labellingEnabled = true;
  int activeLayerNumber = 0;

  void updateLabelling();

  void renderNodesWithHABufferIntoFBO();
  void renderQuad(std::shared_ptr<Graphics::ScreenQuad> quad,
                  Eigen::Matrix4f modelMatrix);
  void renderSliceIntoQuad(Eigen::Matrix4f modelMatrix, int slice);
  void renderScreenQuad();

  void renderDebuggingViews(const RenderData &renderData);
  RenderData createRenderData();
  std::shared_ptr<Camera> getCamera();

  std::shared_ptr<TextureMapperManager> textureMapperManager;
  std::shared_ptr<RecordingAutomation> recordingAutomation;
  RenderData renderData;
};

#endif  // SRC_SCENE_H_
