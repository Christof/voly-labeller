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
#include "./camera_controllers.h"
#include "./nodes.h"
#include "./labelling/labeller_frame_data.h"
#include "./label_node.h"
#include "./eigen_qdebug.h"
#include "./utils/path_helper.h"
#include "./placement/to_gray.h"
#include "./placement/distance_transform.h"
#include "./placement/occupancy.h"
#include "./placement/apollonius.h"
#include "./texture_mapper_manager.h"
#include "./constraint_buffer.h"
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
      std::make_shared<CameraControllers>(invokeManager, camera);

  fbo = std::make_shared<Graphics::FrameBufferObject>();
  constraintBuffer = std::make_shared<ConstraintBuffer>();
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
  constraintBuffer->initialize(gl, textureMapperManager->getBufferSize(),
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
  textureMapperManager->initialize(gl, fbo);

  auto constraintUpdater = std::make_shared<ConstraintUpdater>(
      gl, managers->getShaderManager(), textureMapperManager->getBufferSize(),
      textureMapperManager->getBufferSize());

  placementLabeller->initialize(
      textureMapperManager->getOccupancyTextureMapper(),
      textureMapperManager->getDistanceTransformTextureMapper(),
      textureMapperManager->getApolloniusTextureMapper(), constraintUpdater);
}

void Scene::cleanup()
{
  placementLabeller->cleanup();
  textureMapperManager->cleanup();
}

void Scene::update(double frameTime, QSet<Qt::Key> keysPressed)
{
  this->frameTime = frameTime;
  cameraControllers->update(frameTime);

  frustumOptimizer.update(camera.getViewMatrix());
  camera.updateNearAndFarPlanes(frustumOptimizer.getNear(),
                                frustumOptimizer.getFar());
  haBuffer->updateNearAndFarPlanes(frustumOptimizer.getNear(),
                                   frustumOptimizer.getFar());

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
  if (shouldResize)
  {
    camera.resize(width, height);
    fbo->resize(width, height);
    shouldResize = false;
  }
  glAssert(gl->glViewport(0, 0, width, height));
  glAssert(gl->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  fbo->bind();
  glAssert(gl->glViewport(0, 0, width, height));
  glAssert(gl->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  RenderData renderData;
  renderData.projectionMatrix = camera.getProjectionMatrix();
  renderData.viewMatrix = camera.getViewMatrix();
  renderData.cameraPosition = camera.getPosition();
  renderData.modelMatrix = Eigen::Matrix4f::Identity();
  renderData.windowPixelSize = Eigen::Vector2f(width, height);

  haBuffer->clearAndPrepare(managers);

  nodes->render(gl, managers, renderData);

  managers->getObjectManager()->render(renderData);

  haBuffer->render(managers, renderData);

  // doPick();

  fbo->unbind();

  glAssert(gl->glDisable(GL_DEPTH_TEST));
  renderScreenQuad();

  textureMapperManager->update();

  constraintBuffer->bind();
  glAssert(gl->glViewport(0, 0, textureMapperManager->getBufferSize(),
                          textureMapperManager->getBufferSize()));
  gl->glClearColor(0, 0, 0, 0);
  glAssert(gl->glClear(GL_COLOR_BUFFER_BIT));

  placementLabeller->update(LabellerFrameData(
      frameTime, camera.getProjectionMatrix(), camera.getViewMatrix()));

  constraintBuffer->unbind();

  glAssert(gl->glViewport(0, 0, width, height));
  if (showBufferDebuggingViews)
    renderDebuggingViews(renderData);

  glAssert(gl->glEnable(GL_DEPTH_TEST));

  nodes->renderLabels(gl, managers, renderData);
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

  constraintBuffer->bindTexture(GL_TEXTURE0);
  transformation =
      Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f(0.8, -0.8, 0)) *
                      Eigen::Scaling(Eigen::Vector3f(0.2, 0.2, 1)));
  renderQuad(quad, transformation.matrix());

  renderQuad(transparentQuad, Eigen::Matrix4f::Identity());
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

void Scene::pick(int id, Eigen::Vector2f position)
{
  pickingPosition = position;
  performPicking = true;
  pickingLabelId = id;
}

void Scene::doPick()
{
  if (!performPicking)
    return;

  float depth = -2.0f;

  fbo->bindDepthTexture(GL_TEXTURE0);

  glAssert(gl->glReadPixels(pickingPosition.x(),
                            height - pickingPosition.y() - 1, 1, 1,
                            GL_DEPTH_COMPONENT, GL_FLOAT, &depth));
  Eigen::Vector4f positionNDC(pickingPosition.x() * 2.0f / width - 1.0f,
                              pickingPosition.y() * -2.0f / height + 1.0f,
                              depth * 2.0f - 1.0f, 1.0f);

  Eigen::Matrix4f viewProjection =
      camera.getProjectionMatrix() * camera.getViewMatrix();
  Eigen::Vector4f positionWorld = viewProjection.inverse() * positionNDC;
  positionWorld = positionWorld / positionWorld.w();

  qWarning() << "picked:" << positionWorld;

  performPicking = false;
  auto label = labels->getById(pickingLabelId);
  label.anchorPosition = toVector3f(positionWorld);

  labels->update(label);
}

void Scene::enableBufferDebuggingViews(bool enable)
{
  showBufferDebuggingViews = enable;
}

