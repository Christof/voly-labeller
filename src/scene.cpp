#include "./scene.h"
#include <QCursor>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>
#include "./gl.h"
#include "./input/invoke_manager.h"
#include "./mesh.h"
#include "./mesh_node.h"
#include "./label_node.h"
#include "./render_data.h"
#include "./importer.h"
#include "./camera_controller.h"
#include "./camera_rotation_controller.h"
#include "./camera_zoom_controller.h"
#include "./camera_move_controller.h"
#include "./nodes.h"
#include "./quad.h"
#include "./frame_buffer_object.h"
#include "./utils/persister.h"
#include "./forces/labeller_frame_data.h"
#include "./eigen_qdebug.h"

BOOST_CLASS_EXPORT_GUID(LabelNode, "LabelNode")
BOOST_CLASS_EXPORT_GUID(MeshNode, "MeshNode")

Scene::Scene(std::shared_ptr<InvokeManager> invokeManager,
             std::shared_ptr<Nodes> nodes,
             std::shared_ptr<Labels> labels,
             std::shared_ptr<Forces::Labeller> labeller)

  : nodes(nodes), labels(labels), labeller(labeller)
{
  cameraController = std::make_shared<CameraController>(camera);
  cameraRotationController = std::make_shared<CameraRotationController>(camera);
  cameraZoomController = std::make_shared<CameraZoomController>(camera);
  cameraMoveController = std::make_shared<CameraMoveController>(camera);

  invokeManager->addHandler("cam", cameraController.get());
  invokeManager->addHandler("cameraRotation", cameraRotationController.get());
  invokeManager->addHandler("cameraZoom", cameraZoomController.get());
  invokeManager->addHandler("cameraMove", cameraMoveController.get());

  fbo = std::unique_ptr<FrameBufferObject>(new FrameBufferObject());
}

Scene::~Scene()
{
  qDebug() << "Destructor of Scene";
}

void Scene::initialize()
{
  glAssert(gl->glClearColor(0.9f, 0.9f, 0.8f, 1.0f));

  const std::string filename = "assets/assets.dae";
  Importer importer;

  std::vector<std::shared_ptr<Node>> meshNodes;
  for (unsigned int meshIndex = 0; meshIndex < 2; ++meshIndex)
  {
    auto mesh = importer.import(filename, meshIndex);
    auto node =
        new MeshNode(filename, meshIndex, mesh, Eigen::Matrix4f::Identity());
    meshNodes.push_back(std::unique_ptr<MeshNode>(node));
  }
  auto label = Label(1, "Shoulder", Eigen::Vector3f(0.174f, 0.553f, 0.02f));
  meshNodes.push_back(std::make_shared<LabelNode>(label));

  auto label2 = Label(2, "Ellbow", Eigen::Vector3f(0.334f, 0.317f, -0.013f));
  meshNodes.push_back(std::make_shared<LabelNode>(label2));

  auto label3 = Label(3, "Wound", Eigen::Vector3f(0.262f, 0.422f, 0.058f),
                      Eigen::Vector2f(0.14f, 0.14f));
  meshNodes.push_back(std::make_shared<LabelNode>(label3));

  auto label4 = Label(4, "Wound 2", Eigen::Vector3f(0.034f, 0.373f, 0.141f));
  meshNodes.push_back(std::make_shared<LabelNode>(label4));

  Persister::save(meshNodes, "config/scene.xml");

  nodes->addSceneNodesFrom("config/scene.xml");

  for (auto &labelNode : nodes->getLabelNodes())
    labels->add(labelNode->getLabel());

  quad = std::make_shared<Quad>();

  fbo->initialize(gl, width, height);
}

void Scene::update(double frameTime, QSet<Qt::Key> keysPressed)
{
  this->frameTime = frameTime;
  cameraController->setFrameTime(frameTime);
  cameraRotationController->setFrameTime(frameTime);
  cameraZoomController->setFrameTime(frameTime);
  cameraMoveController->setFrameTime(frameTime);

  auto newPositions = labeller->update(Forces::LabellerFrameData(
      frameTime, camera.getProjectionMatrix(), camera.getViewMatrix()));

  for (auto &labelNode : nodes->getLabelNodes())
  {
    labelNode->labelPosition = newPositions[labelNode->getLabel().id];
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

  nodes->render(gl, renderData);

  doPick();

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

  quad->render(gl, renderData);
}

void Scene::resize(int width, int height)
{
  this->width = width;
  this->height = height;

  shouldResize = true;
}

void Scene::pick(int id, Eigen::Vector2f position,
                 std::function<void(Eigen::Vector3f)> callback)
{
  pickingCallback = callback;
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
  Eigen::Vector3f anchorPosition = toVector3f(positionWorld);
  if (pickingCallback)
    pickingCallback(anchorPosition);

  labeller->updateLabel(pickingLabelId, anchorPosition);
}

