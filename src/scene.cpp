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
#include "./camera_controller.h"
#include "./camera_rotation_controller.h"
#include "./camera_zoom_controller.h"
#include "./camera_move_controller.h"
#include "./nodes.h"
#include "./forces/labeller_frame_data.h"
#include "./label_node.h"
#include "./eigen_qdebug.h"
#include "./utils/path_helper.h"

Scene::Scene(std::shared_ptr<InvokeManager> invokeManager,
             std::shared_ptr<Nodes> nodes, std::shared_ptr<Labels> labels,
             std::shared_ptr<Forces::Labeller> labeller)

  : nodes(nodes), labels(labels), labeller(labeller), frustumOptimizer(nodes)
{
  cameraController = std::make_shared<CameraController>(camera);
  cameraRotationController = std::make_shared<CameraRotationController>(camera);
  cameraZoomController = std::make_shared<CameraZoomController>(camera);
  cameraMoveController = std::make_shared<CameraMoveController>(camera);

  invokeManager->addHandler("cam", cameraController.get());
  invokeManager->addHandler("cameraRotation", cameraRotationController.get());
  invokeManager->addHandler("cameraZoom", cameraZoomController.get());
  invokeManager->addHandler("cameraMove", cameraMoveController.get());

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

  fbo->initialize(gl, width, height);
  haBuffer =
      std::make_shared<Graphics::HABuffer>(Eigen::Vector2i(width, height));
  managers->getShaderManager()->initialize(gl, haBuffer);

  managers->getObjectManager()->initialize(gl, 128, 10000000);
  haBuffer->initialize(gl, managers);
  quad->initialize(gl, managers);

  managers->getTextureManager()->initialize(gl, true, 8);
}

void Scene::update(double frameTime, QSet<Qt::Key> keysPressed)
{
  this->frameTime = frameTime;
  cameraController->setFrameTime(frameTime);
  cameraRotationController->setFrameTime(frameTime);
  cameraZoomController->setFrameTime(frameTime);
  cameraMoveController->setFrameTime(frameTime);

  frustumOptimizer.update(camera.getViewMatrix());
  camera.updateNearAndFarPlanes(frustumOptimizer.getNear(),
                                frustumOptimizer.getFar());
  haBuffer->updateNearAndFarPlanes(frustumOptimizer.getNear(),
                                   frustumOptimizer.getFar());

  auto newPositions = labeller->update(Forces::LabellerFrameData(
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

  haBuffer->clearAndPrepare(managers);

  nodes->render(gl, managers, renderData);

  managers->getObjectManager()->render(renderData);

  haBuffer->render(managers, renderData);

  // doPick();

  fbo->unbind();

  renderScreenQuad();

  Eigen::Affine3f transformation(
      Eigen::Translation3f(Eigen::Vector3f(-0.8, -0.8, 0)) *
      Eigen::Scaling(Eigen::Vector3f(0.2, 0.2, 1)));
  fbo->bindColorTexture(GL_TEXTURE0);
  renderQuad(transformation.matrix());
}

void Scene::renderQuad(Eigen::Matrix4f modelMatrix)
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

  renderQuad(Eigen::Matrix4f::Identity());
}

void Scene::resize(int width, int height)
{
  this->width = width;
  this->height = height;

  shouldResize = true;
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

