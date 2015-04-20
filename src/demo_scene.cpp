#include "./demo_scene.h"

#include <QDebug>
#include <Eigen/Core>
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
#include "./nodes.h"
#include "./utils/persister.h"

BOOST_CLASS_EXPORT_GUID(LabelNode, "LabelNode")
BOOST_CLASS_EXPORT_GUID(MeshNode, "MeshNode")

DemoScene::DemoScene(std::shared_ptr<InvokeManager> invokeManager,
                     std::shared_ptr<Nodes> nodes)
  : nodes(nodes)
{
  cameraController = std::make_shared<CameraController>(camera);
  cameraRotationController = std::make_shared<CameraRotationController>(camera);

  invokeManager->addHandler("cam", cameraController.get());
  invokeManager->addHandler("cameraRotation", cameraRotationController.get());
}

DemoScene::~DemoScene()
{
}

void DemoScene::initialize()
{
  glAssert(gl->glClearColor(0.9f, 0.9f, 0.8f, 1.0f));

  const std::string filename = "../assets/assets.dae";
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

  auto label3 = Label(3, "Wound", Eigen::Vector3f(0.262f, 0.422f, 0.058f));
  meshNodes.push_back(std::make_shared<LabelNode>(label3));

  Persister::save(meshNodes, "../config/scene.xml");

  nodes->addSceneNodesFrom("../config/scene.xml");
}

void DemoScene::update(double frameTime, QSet<Qt::Key> keysPressed)
{
  this->frameTime = frameTime;
  cameraController->setFrameTime(frameTime);
  cameraRotationController->setFrameTime(frameTime);
}

void DemoScene::render()
{
  glAssert(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  RenderData renderData;
  renderData.projectionMatrix = camera.getProjectionMatrix();
  renderData.viewMatrix = camera.getViewMatrix();
  renderData.cameraPosition = camera.getPosition();
  renderData.modelMatrix = Eigen::Matrix4f::Identity();

  nodes->render(gl, renderData);
}

void DemoScene::resize(int width, int height)
{
  glAssert(glViewport(0, 0, width, height));
}

