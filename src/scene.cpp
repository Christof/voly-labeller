#include "./scene.h"

#include <QDebug>
#include <QOpenGLFramebufferObject>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>
#include "./gl.h"
#include "./input/invoke_manager.h"
#include "./mesh.h"
#include "./mesh_node.h"
#include "./label_node.h"
#include "./forces_visualizer_node.h"
#include "./render_data.h"
#include "./importer.h"
#include "./camera_controller.h"
#include "./camera_rotation_controller.h"
#include "./camera_zoom_controller.h"
#include "./camera_move_controller.h"
#include "./nodes.h"
#include "./quad.h"
#include "./utils/persister.h"
#include "./forces/labeller_frame_data.h"

BOOST_CLASS_EXPORT_GUID(LabelNode, "LabelNode")
BOOST_CLASS_EXPORT_GUID(MeshNode, "MeshNode")

Scene::Scene(std::shared_ptr<InvokeManager> invokeManager,
             std::shared_ptr<Nodes> nodes,
             std::shared_ptr<Forces::Labeller> labeller)

  : nodes(nodes), labeller(labeller)
{
  cameraController = std::make_shared<CameraController>(camera);
  cameraRotationController = std::make_shared<CameraRotationController>(camera);
  cameraZoomController = std::make_shared<CameraZoomController>(camera);
  cameraMoveController = std::make_shared<CameraMoveController>(camera);

  invokeManager->addHandler("cam", cameraController.get());
  invokeManager->addHandler("cameraRotation", cameraRotationController.get());
  invokeManager->addHandler("cameraZoom", cameraZoomController.get());
  invokeManager->addHandler("cameraMove", cameraMoveController.get());
}

Scene::~Scene()
{
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
  {
    auto label = labelNode->getLabel();
    labeller->addLabel(label.id, label.text, label.anchorPosition, label.size);
  }

  nodes->addNode(std::make_shared<ForcesVisualizerNode>(labeller));

  quad = std::make_shared<Quad>();
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
  glAssert(gl->glViewport(0, 0, width, height));
  glAssert(gl->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  glAssert(fbo->bind());
  glAssert(gl->glViewport(0, 0, width, height));
  glAssert(gl->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT |
                       GL_STENCIL_BUFFER_BIT));

  RenderData renderData;
  renderData.projectionMatrix = camera.getProjectionMatrix();
  renderData.viewMatrix = camera.getViewMatrix();
  renderData.cameraPosition = camera.getPosition();
  renderData.modelMatrix = Eigen::Matrix4f::Identity();

  nodes->render(gl, renderData);

  glAssert(fbo->release());

  renderData.projectionMatrix = Eigen::Matrix4f::Identity();
  renderData.viewMatrix = Eigen::Matrix4f::Identity();
  renderData.modelMatrix =
      Eigen::Affine3f(Eigen::AlignedScaling3f(1, -1, 1)).matrix();

  glAssert(gl->glActiveTexture(GL_TEXTURE0));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, fbo->texture()));
  // glAssert(gl->glBindTexture(GL_TEXTURE_2D, depthTexture));

  quad->render(gl, renderData);
}

void Scene::resize(int width, int height)
{
  this->width = width;
  this->height = height;

  glAssert(glViewport(0, 0, width, height));
  camera.resize(width, height);
  if (fbo.get())
    fbo->release();

  fbo = std::unique_ptr<QOpenGLFramebufferObject>(new QOpenGLFramebufferObject(
      width, height, QOpenGLFramebufferObject::Depth));
  qWarning() << "create fbo";

  glAssert(fbo->bind());
  glAssert(glViewport(0, 0, width, height));

  glAssert(gl->glGenTextures(1, &depthTexture));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, depthTexture));
  glAssert(
      gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
  glAssert(
      gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  glAssert(
      gl->glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE));
  glAssert(gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width,
                            height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE,
                            NULL));

  glAssert(gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                      GL_TEXTURE_2D, depthTexture, 0));

  fbo->release();
  /*
  glAssert(fbo->bind());
  glAssert(glViewport(0, 0, width, height));

  glAssert(fbo->bindDefault());
  */
}

