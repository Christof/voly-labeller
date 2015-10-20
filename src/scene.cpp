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
#include "./placement/summed_area_table.h"
#include "./placement/to_gray.h"
#include "./placement/distance_transform.h"
#include "./placement/occupancy.h"
#include "./placement/apollonius.h"
#include "./placement/cost_function_calculator.h"

Scene::Scene(std::shared_ptr<InvokeManager> invokeManager,
             std::shared_ptr<Nodes> nodes, std::shared_ptr<Labels> labels,
             std::shared_ptr<Forces::Labeller> forcesLabeller)

  : nodes(nodes), labels(labels), forcesLabeller(forcesLabeller),
    frustumOptimizer(nodes)
{
  cameraControllers =
      std::make_shared<CameraControllers>(invokeManager, camera);

  fbo = std::unique_ptr<Graphics::FrameBufferObject>(
      new Graphics::FrameBufferObject());
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

  fbo->initialize(gl, width, height);
  haBuffer =
      std::make_shared<Graphics::HABuffer>(Eigen::Vector2i(width, height));
  managers->getShaderManager()->initialize(gl, haBuffer);

  managers->getObjectManager()->initialize(gl, 128, 10000000);
  haBuffer->initialize(gl, managers);
  quad->initialize(gl, managers);
  positionQuad->initialize(gl, managers);

  managers->getTextureManager()->initialize(gl, true, 8);

  occupancyTexture =
      std::make_shared<Graphics::StandardTexture2d>(width, height, GL_R32F);
  occupancyTexture->initialize(gl);
  distanceTransformTexture =
      std::make_shared<Graphics::StandardTexture2d>(width, height, GL_RGBA32F);
  distanceTransformTexture->initialize(gl);
}

void Scene::cleanup()
{
  colorTextureMapper.reset();
  positionsTextureMapper.reset();
  occupancyTextureMapper.reset();
  distanceTransformTextureMapper.reset();
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

  auto newPositions = forcesLabeller->update(LabellerFrameData(
      frameTime, camera.getProjectionMatrix(), camera.getViewMatrix()));

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

  renderDebuggingViews(renderData);

  glAssert(gl->glEnable(GL_DEPTH_TEST));
}

void Scene::renderDebuggingViews(const RenderData &renderData)
{
  if (!colorTextureMapper.get())
  {
    colorTextureMapper = std::shared_ptr<CudaTextureMapper>(
        CudaTextureMapper::createReadWriteMapper(fbo->getRenderTextureId(),
                                                 width, height));
    positionsTextureMapper = std::shared_ptr<CudaTextureMapper>(
        CudaTextureMapper::createReadOnlyMapper(fbo->getPositionTextureId(),
                                                width, height));
    distanceTransformTextureMapper = std::shared_ptr<CudaTextureMapper>(
        CudaTextureMapper::createReadWriteDiscardMapper(
            distanceTransformTexture->getId(), width, height));
    occupancyTextureMapper = std::shared_ptr<CudaTextureMapper>(
        CudaTextureMapper::createReadWriteDiscardMapper(
            occupancyTexture->getId(), width, height));
  }

  fbo->bindDepthTexture(GL_TEXTURE0);
  auto transformation =
      Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f(-0.8, -0.8, 0)) *
                      Eigen::Scaling(Eigen::Vector3f(0.2, 0.2, 1)));
  renderQuad(quad, transformation.matrix());

  Occupancy(positionsTextureMapper, occupancyTextureMapper).runKernel();
  occupancyTexture->bind();
  transformation =
      Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f(-0.4, -0.8, 0)) *
                      Eigen::Scaling(Eigen::Vector3f(0.2, 0.2, 1)));
  renderQuad(quad, transformation.matrix());

  DistanceTransform distanceTransform(occupancyTextureMapper,
                                      distanceTransformTextureMapper);
  distanceTransform.run();
  distanceTransformTexture->bind();
  transformation =
      Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f(0.0, -0.8, 0)) *
                      Eigen::Scaling(Eigen::Vector3f(0.2, 0.2, 1)));
  renderQuad(quad, transformation.matrix());

  auto seedBuffer = Apollonius::createSeedBufferFromLabels(
      labels->getLabels(), renderData.projectionMatrix * renderData.viewMatrix,
      Eigen::Vector2i(width, height));
  Apollonius(distanceTransformTextureMapper, seedBuffer,
             distanceTransform.getResults(), labels->count()).run();
  transformation =
      Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f(0.4, -0.8, 0)) *
                      Eigen::Scaling(Eigen::Vector3f(0.2, 0.2, 1)));
  renderQuad(quad, transformation.matrix());

  CostFunctionCalculator calc(width, height);
  calc.calculateCosts(distanceTransform.getResults());
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

