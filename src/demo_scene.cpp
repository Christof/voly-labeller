#include "./demo_scene.h"

#include <QObject>
#include <QDebug>
#include <QPainter>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include "./gl.h"
#include "./input/invoke_manager.h"
#include "./mesh.h"
#include "./mesh_node.h"
#include "./label_node.h"
#include "./render_data.h"
#include "./importer.h"
#include "./utils/persister.h"

BOOST_CLASS_EXPORT_GUID(LabelNode, "LabelNode")
BOOST_CLASS_EXPORT_GUID(MeshNode, "MeshNode")

DemoScene::DemoScene(std::shared_ptr<InvokeManager> invokeManager,
                     std::shared_ptr<Nodes> nodes)
  : nodes(nodes)
{
  cameraController =
      std::shared_ptr<CameraController>(new CameraController(camera));

  invokeManager->addHandler("cam", cameraController.get());
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
    auto transformation = Eigen::Affine3f::Identity();
    auto node =
        new MeshNode(filename, meshIndex, mesh, transformation.matrix());
    meshNodes.push_back(std::unique_ptr<MeshNode>(node));
  }
  auto label = Label(1, "My label 1", Eigen::Vector3f(0.174f, 0.553f, 0.02f));
  meshNodes.push_back(std::shared_ptr<LabelNode>(new LabelNode(label)));

  Persister::save(meshNodes, "../config/scene.xml");

  nodes->addSceneNodesFrom("../config/scene.xml");
}

void DemoScene::update(double frameTime, QSet<Qt::Key> keysPressed)
{
  this->frameTime = frameTime;
  cameraController->setFrameTime(frameTime);
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

