#include "./scene.h"
#include <QCursor>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>
#include "./graphics/gl.h"
#include "./input/invoke_manager.h"
#include "./graphics/render_data.h"
#include "./camera_controller.h"
#include "./camera_rotation_controller.h"
#include "./camera_zoom_controller.h"
#include "./camera_move_controller.h"
#include "./nodes.h"
#include "./forces/labeller_frame_data.h"
#include "./label_node.h"
#include "./eigen_qdebug.h"

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
  bufferManager =
      std::unique_ptr<Graphics::BufferManager>(new Graphics::BufferManager());
  textureManager = std::make_shared<Graphics::TextureManager>();
}

Scene::~Scene()
{
  qDebug() << "Destructor of Scene";
}

void Scene::initialize()
{
  glAssert(gl->glClearColor(0.9f, 0.9f, 0.8f, 1.0f));

  quad = std::make_shared<Graphics::Quad>(":shader/label.vert",
                                          ":shader/texture.frag");

  fbo->initialize(gl, width, height);
  haBuffer =
      std::make_shared<Graphics::HABuffer>(Eigen::Vector2i(width, height));
  haBuffer->initialize(gl);

  bufferManager->initialize(gl, 10, 100);
  std::vector<float> positions = { 1.0f, 1.0f,  0.0f, -1.0f, 1.0f,  0.0f,
                                   1.0f, -1.0f, 0.0f, -1.0f, -1.0f, 0.0f };
  std::vector<float> normals = { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
                                 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f };
  std::vector<float> colors = {
    1.0f, 0.0f, 1.0f, 0.5f, 0.0f, 1.0f, 1.0f, 0.5f,
    0.0f, 0.0f, 1.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.5f
  };
  std::vector<float> texCoords = { 1.0f, 0.0f, 0.0f, 0.0f,
                                   1.0f, 1.0f, 0.0f, 1.0f };
  std::vector<uint> indices = { 0, 1, 2, 1, 3, 2 };
  objectId =
      bufferManager->addObject(positions, normals, colors, texCoords, indices);
  shader = std::make_shared<Graphics::ShaderProgram>(gl, ":/shader/pass.vert",
                                                     ":/shader/test.frag");
  textureManager->initialize(gl, true, 8);
  textureManager->addTexture(
      "/home/christof/Documents/master/volyHA5/scenes/tiger/tiger-atlas.jpg");
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

  haBuffer->clearAndPrepare();

  nodes->render(gl, haBuffer, renderData);

  shader->bind();
  Eigen::Matrix4f mvp = renderData.projectionMatrix * renderData.viewMatrix;
  shader->setUniform("modelViewProjectionMatrix", mvp);
  // shader->setUniform("modelMatrix", renderData.modelMatrix);
  haBuffer->begin(shader);

  bufferManager->render();
  shader->release();

  haBuffer->render();

  // doPick();

  fbo->unbind();

  renderScreenQuad();
}

void Scene::renderScreenQuad()
{
  RenderData renderData;
  renderData.projectionMatrix = Eigen::Matrix4f::Identity();
  renderData.viewMatrix = Eigen::Matrix4f::Identity();
  renderData.modelMatrix =
      Eigen::Affine3f(Eigen::AlignedScaling3f(1, -1, 1)).matrix();

  fbo->bindColorTexture(GL_TEXTURE0);
  // fbo->bindDepthTexture(GL_TEXTURE0);

  quad->renderToFrameBuffer(gl, renderData);
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

