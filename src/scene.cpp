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

Scene::Scene(std::shared_ptr<InvokeManager> invokeManager,
             std::shared_ptr<Nodes> nodes, std::shared_ptr<Labels> labels,
             std::shared_ptr<Forces::Labeller> forcesLabeller,
             std::shared_ptr<Placement::Labeller> placementLabeller)

  : nodes(nodes), labels(labels), forcesLabeller(forcesLabeller),
    placementLabeller(placementLabeller), frustumOptimizer(nodes)
{
  cameraControllers =
      std::make_shared<CameraControllers>(invokeManager, camera);

  fbo = std::make_shared<Graphics::FrameBufferObject>();
  managers = std::make_shared<Graphics::Managers>();

  textureMapperManager =
      std::make_shared<TextureMapperManager>(postProcessingTextureSize);
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

  fbo->initialize(gl, width, height);
  haBuffer =
      std::make_shared<Graphics::HABuffer>(Eigen::Vector2i(width, height));
  managers->getShaderManager()->initialize(gl, haBuffer);

  managers->getObjectManager()->initialize(gl, 128, 10000000);
  haBuffer->initialize(gl, managers);
  quad->initialize(gl, managers);
  positionQuad->initialize(gl, managers);

  managers->getTextureManager()->initialize(gl, true, 8);

  textureMapperManager->resize(width, height);
  textureMapperManager->initialize(gl, fbo);

  placementLabeller->initialize(textureMapperManager->getOccupancyTextureMapper(),
      textureMapperManager->getDistanceTransformTextureMapper());
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

  renderDebuggingViews(renderData);

  glAssert(gl->glEnable(GL_DEPTH_TEST));
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
  renderQuad(quad, transformation.matrix());

  placementLabeller->update(LabellerFrameData(
      frameTime, camera.getProjectionMatrix(), camera.getViewMatrix()));

  transformation =
      Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f(0.4, -0.8, 0)) *
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

