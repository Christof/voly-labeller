#if _WIN32
#pragma warning(disable : 4522 4996 4267)
#endif

#include "./scene.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>
#include <map>
#include "./graphics/gl.h"
#include "./input/invoke_manager.h"
#include "./graphics/render_data.h"
#include "./graphics/managers.h"
#include "./graphics/volume_manager.h"
#include "./graphics/shader_program.h"
#include "./graphics/buffer_drawer.h"
#include "./camera_controllers.h"
#include "./nodes.h"
#include "./camera_node.h"
#include "./label_node.h"
#include "./eigen_qdebug.h"
#include "./utils/path_helper.h"
#include "./placement/to_gray.h"
#include "./placement/distance_transform.h"
#include "./placement/apollonius.h"
#include "./texture_mapper_manager.h"
#include "./constraint_buffer_object.h"
#include "./labelling_coordinator.h"

Scene::Scene(int layerCount, std::shared_ptr<InvokeManager> invokeManager,
             std::shared_ptr<Nodes> nodes, std::shared_ptr<Labels> labels,
             std::shared_ptr<LabellingCoordinator> labellingCoordinator,
             std::shared_ptr<TextureMapperManager> textureMapperManager)

  : nodes(nodes), labels(labels), labellingCoordinator(labellingCoordinator),
    frustumOptimizer(nodes), textureMapperManager(textureMapperManager)
{
  cameraControllers =
      std::make_shared<CameraControllers>(invokeManager, getCamera());

  fbo = std::make_shared<Graphics::FrameBufferObject>(layerCount);
  constraintBufferObject = std::make_shared<ConstraintBufferObject>();
  managers = std::make_shared<Graphics::Managers>();
}

Scene::~Scene()
{
  nodes->clear();
  qInfo() << "Destructor of Scene"
          << "Remaining managers instances" << managers.use_count();
}

void Scene::initialize()
{
  quad = std::make_shared<Graphics::ScreenQuad>(
      ":shader/pass.vert", ":shader/textureForRenderBuffer.frag");
  screenQuad = std::make_shared<Graphics::ScreenQuad>(
      ":shader/pass.vert", ":shader/combineLayers.frag");
  positionQuad = std::make_shared<Graphics::ScreenQuad>(
      ":shader/pass.vert", ":shader/positionRenderTarget.frag");
  distanceTransformQuad = std::make_shared<Graphics::ScreenQuad>(
      ":shader/pass.vert", ":shader/distanceTransform.frag");
  transparentQuad = std::make_shared<Graphics::ScreenQuad>(
      ":shader/pass.vert", ":shader/transparentOverlay.frag");

  fbo->initialize(gl, width, height);
  picker = std::make_unique<Picker>(fbo, gl, labels);
  picker->resize(width, height);
  constraintBufferObject->initialize(gl, textureMapperManager->getBufferSize(),
                                     textureMapperManager->getBufferSize());
  haBuffer =
      std::make_shared<Graphics::HABuffer>(Eigen::Vector2i(width, height));
  managers->getShaderManager()->initialize(gl, haBuffer);

  managers->getObjectManager()->initialize(gl, 128, 10000000);
  haBuffer->initialize(gl, managers);
  quad->initialize(gl, managers);
  screenQuad->initialize(gl, managers);
  positionQuad->initialize(gl, managers);
  distanceTransformQuad->initialize(gl, managers);
  transparentQuad->initialize(gl, managers);

  managers->getTextureManager()->initialize(gl, true, 8);

  textureMapperManager->createTextureMappersForLayers(fbo->getLayerCount());
  textureMapperManager->resize(width, height);
  textureMapperManager->initialize(gl, fbo, constraintBufferObject);

  auto drawer = std::make_shared<Graphics::BufferDrawer>(
      textureMapperManager->getBufferSize(),
      textureMapperManager->getBufferSize(), gl, managers->getShaderManager());

  labellingCoordinator->initialize(textureMapperManager->getBufferSize(),
                                   drawer, textureMapperManager, width, height);
}

void Scene::cleanup()
{
  labellingCoordinator->cleanup();

  textureMapperManager->cleanup();
}

void Scene::update(double frameTime)
{
  auto camera = getCamera();

  if (camera->needsResizing())
    camera->resize(width, height);

  this->frameTime = frameTime;
  cameraControllers->update(camera, frameTime);

  frustumOptimizer.update(camera->getViewMatrix());
  camera->updateNearAndFarPlanes(frustumOptimizer.getNear(),
                                 frustumOptimizer.getFar());
  haBuffer->updateNearAndFarPlanes(frustumOptimizer.getNear(),
                                   frustumOptimizer.getFar());

  labellingCoordinator->update(frameTime, camera->getProjectionMatrix(),
                               camera->getViewMatrix(), activeLayerNumber);
}

void Scene::render()
{
  auto camera = getCamera();

  if (shouldResize)
  {
    camera->resize(width, height);
    fbo->resize(width, height);
    shouldResize = false;
  }

  glAssert(gl->glViewport(0, 0, width, height));
  glAssert(gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f));
  glAssert(gl->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  RenderData newRenderData = createRenderData();
  bool isIdle =
      newRenderData.viewProjectionMatrix == renderData.viewProjectionMatrix;
  if (isIdle)
    qInfo() << "Idle";

  renderData = newRenderData;

  renderNodesWithHABufferIntoFBO();

  textureMapperManager->update();

  constraintBufferObject->bind();

  labellingCoordinator->updatePlacement();

  constraintBufferObject->unbind();

  glAssert(gl->glViewport(0, 0, width, height));

  fbo->bind();
  glAssert(gl->glDisable(GL_DEPTH_TEST));
  nodes->renderLabels(gl, managers, renderData);
  fbo->unbind();
  renderScreenQuad();
  nodes->renderOverlays(gl, managers, renderData);

  if (showConstraintOverlay)
  {
    constraintBufferObject->bindTexture(GL_TEXTURE0);
    renderQuad(transparentQuad, Eigen::Matrix4f::Identity());
  }

  if (showBufferDebuggingViews)
    renderDebuggingViews(renderData);

  glAssert(gl->glEnable(GL_DEPTH_TEST));
}

void Scene::renderNodesWithHABufferIntoFBO()
{
  fbo->bind();
  glAssert(gl->glViewport(0, 0, width, height));
  glAssert(gl->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  haBuffer->clearAndPrepare(managers);

  nodes->render(gl, managers, renderData);

  managers->getObjectManager()->render(renderData);

  std::vector<float> zValues = labellingCoordinator->updateClusters();

  haBuffer->setLayerZValues(zValues);
  haBuffer->render(managers, renderData);

  picker->doPick(renderData.viewProjectionMatrix);

  fbo->unbind();
}

void Scene::renderDebuggingViews(const RenderData &renderData)
{
  for (int i = 0; i < fbo->getLayerCount(); ++i)
  {
    fbo->bindColorTexture(i, GL_TEXTURE0);
    auto transformation = Eigen::Affine3f(
        Eigen::Translation3f(Eigen::Vector3f(-0.8f + 0.4f * i, -0.4f, 0)) *
        Eigen::Scaling(Eigen::Vector3f(0.2f, 0.2f, 1.0f)));
    renderQuad(quad, transformation.matrix());
  }

  fbo->bindAccumulatedLayersTexture(GL_TEXTURE0);
  auto transformation =
      Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f(
                          -0.8f + 0.4f * fbo->getLayerCount(), -0.4f, 0.0f)) *
                      Eigen::Scaling(Eigen::Vector3f(0.2f, 0.2f, 1.0f)));
  renderQuad(quad, transformation.matrix());

  fbo->bindDepthTexture(GL_TEXTURE0);
  transformation = Eigen::Affine3f(
      Eigen::Translation3f(Eigen::Vector3f(-0.8f, -0.8f, 0.0f)) *
      Eigen::Scaling(Eigen::Vector3f(0.2f, 0.2f, 1.0f)));
  renderQuad(quad, transformation.matrix());

  textureMapperManager->bindOcclusionTexture();
  transformation = Eigen::Affine3f(
      Eigen::Translation3f(Eigen::Vector3f(-0.4f, -0.8f, 0.0f)) *
      Eigen::Scaling(Eigen::Vector3f(0.2f, 0.2f, 1.0f)));
  renderQuad(quad, transformation.matrix());

  int layerIndex = activeLayerNumber == 0 ? 0 : activeLayerNumber - 1;
  textureMapperManager->bindSaliencyTexture();
  transformation =
      Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f(0.0f, -0.8f, 0.0f)) *
                      Eigen::Scaling(Eigen::Vector3f(0.2f, 0.2f, 1.0f)));
  renderQuad(distanceTransformQuad, transformation.matrix());

  textureMapperManager->bindApollonius(layerIndex);
  transformation =
      Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f(0.4f, -0.8f, 0.0f)) *
                      Eigen::Scaling(Eigen::Vector3f(0.2f, 0.2f, 1.0f)));
  renderQuad(quad, transformation.matrix());

  constraintBufferObject->bindTexture(GL_TEXTURE0);
  transformation =
      Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f(0.8f, -0.8f, 0.0f)) *
                      Eigen::Scaling(Eigen::Vector3f(0.2f, 0.2f, 1.0f)));
  renderQuad(quad, transformation.matrix());
}

void Scene::renderQuad(std::shared_ptr<Graphics::ScreenQuad> quad,
                       Eigen::Matrix4f modelMatrix)
{
  RenderData renderData;
  renderData.modelMatrix = modelMatrix;

  quad->getShaderProgram()->bind();
  quad->getShaderProgram()->setUniform("textureSampler", 0);
  quad->renderImmediately(gl, managers, renderData);
}

void Scene::renderScreenQuad()
{
  if (activeLayerNumber == 0)
  {
    fbo->bindColorTexture(0, GL_TEXTURE0);
    fbo->bindColorTexture(1, GL_TEXTURE1);
    fbo->bindColorTexture(2, GL_TEXTURE2);
    fbo->bindColorTexture(3, GL_TEXTURE3);

    screenQuad->getShaderProgram()->setUniform("layer1", 0);
    screenQuad->getShaderProgram()->setUniform("layer2", 1);
    screenQuad->getShaderProgram()->setUniform("layer3", 2);
    screenQuad->getShaderProgram()->setUniform("layer4", 3);
    renderQuad(screenQuad, Eigen::Matrix4f::Identity());

    gl->glActiveTexture(GL_TEXTURE0);
  }
  else if (activeLayerNumber == 9)
  {
    fbo->bindAccumulatedLayersTexture(GL_TEXTURE0);
    renderQuad(quad, Eigen::Matrix4f::Identity());
  }
  else
  {
    fbo->bindColorTexture(activeLayerNumber - 1, GL_TEXTURE0);
    renderQuad(quad, Eigen::Matrix4f::Identity());
  }
}

void Scene::resize(int width, int height)
{
  this->width = width;
  this->height = height;

  shouldResize = true;

  labellingCoordinator->resize(width, height);
}

RenderData Scene::createRenderData()
{
  auto camera = getCamera();

  return RenderData(camera->getProjectionMatrix(), camera->getViewMatrix(),
                    camera->getPosition(), Eigen::Vector2f(width, height));
}

void Scene::pick(int id, Eigen::Vector2f position)
{
  if (picker.get())
    picker->pick(id, position);
}

void Scene::enableBufferDebuggingViews(bool enable)
{
  showBufferDebuggingViews = enable;
}

void Scene::enableConstraingOverlay(bool enable)
{
  showConstraintOverlay = enable;
}

std::shared_ptr<Camera> Scene::getCamera()
{
  return nodes->getCameraNode()->getCamera();
}

void Scene::setRenderLayer(int layerNumber)
{
  activeLayerNumber = layerNumber;
}

