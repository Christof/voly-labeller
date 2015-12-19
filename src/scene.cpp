#include "./scene.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>
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
#include "./labelling/labeller_frame_data.h"
#include "./label_node.h"
#include "./eigen_qdebug.h"
#include "./utils/path_helper.h"
#include "./placement/to_gray.h"
#include "./placement/distance_transform.h"
#include "./placement/occupancy.h"
#include "./placement/apollonius.h"
#include "./texture_mapper_manager.h"
#include "./constraint_buffer_object.h"
#include "./placement/constraint_updater.h"

Scene::Scene(std::shared_ptr<InvokeManager> invokeManager,
             std::shared_ptr<Nodes> nodes, std::shared_ptr<Labels> labels,
             std::shared_ptr<Forces::Labeller> forcesLabeller,
             std::shared_ptr<Placement::Labeller> placementLabeller,
             std::shared_ptr<TextureMapperManager> textureMapperManager)

  : nodes(nodes), labels(labels), forcesLabeller(forcesLabeller),
    placementLabeller(placementLabeller), frustumOptimizer(nodes),
    textureMapperManager(textureMapperManager)
{
  cameraControllers =
      std::make_shared<CameraControllers>(invokeManager, getCamera());

  fbo = std::make_shared<Graphics::FrameBufferObject>();
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
  glAssert(gl->glClearColor(0.9f, 0.9f, 0.8f, 1.0f));

  quad = std::make_shared<Graphics::ScreenQuad>(
      ":shader/pass.vert", ":shader/textureForRenderBuffer.frag");
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
  positionQuad->initialize(gl, managers);
  distanceTransformQuad->initialize(gl, managers);
  transparentQuad->initialize(gl, managers);

  managers->getTextureManager()->initialize(gl, true, 8);

  textureMapperManager->resize(width, height);
  textureMapperManager->initialize(gl, fbo, constraintBufferObject);

  auto drawer = std::make_shared<Graphics::BufferDrawer>(
      textureMapperManager->getBufferSize(),
      textureMapperManager->getBufferSize(), gl, managers->getShaderManager());

  auto constraintUpdater = std::make_shared<ConstraintUpdater>(
      drawer, textureMapperManager->getBufferSize(),
      textureMapperManager->getBufferSize());

  placementLabeller->initialize(
      textureMapperManager->getOccupancyTextureMapper(),
      textureMapperManager->getDistanceTransformTextureMapper(),
      textureMapperManager->getApolloniusTextureMapper(),
      textureMapperManager->getConstraintTextureMapper(), constraintUpdater);
}

void Scene::cleanup()
{
  placementLabeller->cleanup();
  textureMapperManager->cleanup();
}

void Scene::update(double frameTime, QSet<Qt::Key> keysPressed)
{
  auto camera = getCamera();

  this->frameTime = frameTime;
  cameraControllers->update(camera, frameTime);

  // frustumOptimizer.update(camera->getViewMatrix());
  // camera->updateNearAndFarPlanes(frustumOptimizer.getNear(),
  //                                frustumOptimizer.getFar());
  // haBuffer->updateNearAndFarPlanes(frustumOptimizer.getNear(),
  //                                  frustumOptimizer.getFar());

  /*
  auto newPositions = forcesLabeller->update(LabellerFrameData(
      frameTime, camera.getProjectionMatrix(), camera.getViewMatrix()));
      */
  auto newPositions = placementLabeller->getLastPlacementResult();
  for (auto &labelNode : nodes->getLabelNodes())
  {
    labelNode->labelPosition = newPositions[labelNode->label.id];
  }
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

  if (camera->needsResizing())
    camera->resize(width, height);

  glAssert(gl->glViewport(0, 0, width, height));
  glAssert(gl->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  RenderData renderData = createRenderData();

  renderNodesWithHABufferIntoFBO(renderData);

  glAssert(gl->glDisable(GL_DEPTH_TEST));
  renderScreenQuad();

  textureMapperManager->update();

  constraintBufferObject->bind();

  placementLabeller->update(LabellerFrameData(
      frameTime, camera->getProjectionMatrix(), camera->getViewMatrix()));

  constraintBufferObject->unbind();

  glAssert(gl->glViewport(0, 0, width, height));

  if (showConstraintOverlay)
  {
    constraintBufferObject->bindTexture(GL_TEXTURE0);
    renderQuad(transparentQuad, Eigen::Matrix4f::Identity());
  }

  if (showBufferDebuggingViews)
    renderDebuggingViews(renderData);

  glAssert(gl->glEnable(GL_DEPTH_TEST));

  nodes->renderLabels(gl, managers, renderData);
}

void Scene::renderNodesWithHABufferIntoFBO(const RenderData &renderData)
{
  fbo->bind();
  glAssert(gl->glViewport(0, 0, width, height));
  glAssert(gl->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  haBuffer->clearAndPrepare(managers);

  nodes->render(gl, managers, renderData);

  managers->getObjectManager()->render(renderData);

  haBuffer->render(managers, renderData);

  picker->doPick(renderData.projectionMatrix * renderData.viewMatrix);

  fbo->unbind();
}

void Scene::renderDebuggingViews(const RenderData &renderData)
{
  fbo->bindDepthTexture(GL_TEXTURE0);
  auto transformation =
      Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f(-0.8, -0.8, 0)) *
                      Eigen::Scaling(Eigen::Vector3f(0.2, 0.2, 1)));
  renderQuad(quad, transformation.matrix());

  textureMapperManager->bindOccupancyTexture();
  transformation =
      Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f(-0.4, -0.8, 0)) *
                      Eigen::Scaling(Eigen::Vector3f(0.2, 0.2, 1)));
  renderQuad(quad, transformation.matrix());

  textureMapperManager->bindDistanceTransform();
  transformation =
      Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f(0.0, -0.8, 0)) *
                      Eigen::Scaling(Eigen::Vector3f(0.2, 0.2, 1)));
  renderQuad(distanceTransformQuad, transformation.matrix());

  textureMapperManager->bindApollonius();
  transformation =
      Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f(0.4, -0.8, 0)) *
                      Eigen::Scaling(Eigen::Vector3f(0.2, 0.2, 1)));
  renderQuad(quad, transformation.matrix());

  constraintBufferObject->bindTexture(GL_TEXTURE0);
  transformation =
      Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f(0.8, -0.8, 0)) *
                      Eigen::Scaling(Eigen::Vector3f(0.2, 0.2, 1)));
  renderQuad(quad, transformation.matrix());
}

void Scene::renderQuad(std::shared_ptr<Graphics::ScreenQuad> quad,
                       Eigen::Matrix4f modelMatrix)
{
  RenderData renderData;
  renderData.projectionMatrix = Eigen::Matrix4f::Identity();
  renderData.viewMatrix = Eigen::Matrix4f::Identity();
  renderData.modelMatrix = modelMatrix;

  quad->getShaderProgram()->bind();
  quad->getShaderProgram()->setUniform("textureSampler", 0);
  quad->renderImmediately(gl, managers, renderData);
}

void Scene::renderScreenQuad()
{
  fbo->bindColorTexture(GL_TEXTURE0);

  renderQuad(quad, Eigen::Matrix4f::Identity());
}

void Scene::resize(int width, int height)
{
  this->width = width;
  this->height = height;

  placementLabeller->resize(width, height);

  shouldResize = true;

  forcesLabeller->resize(width, height);
}

RenderData Scene::createRenderData()
{
  RenderData renderData;
  auto camera = getCamera();
  renderData.projectionMatrix = camera->getProjectionMatrix();
  renderData.viewMatrix = camera->getViewMatrix();
  renderData.cameraPosition = camera->getPosition();
  renderData.modelMatrix = Eigen::Matrix4f::Identity();
  renderData.windowPixelSize = Eigen::Vector2f(width, height);

  return renderData;
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

