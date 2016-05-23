#include "./default_scene_creator.h"
#include <Eigen/Geometry>
#include <string>
#include <vector>
#include "./importer.h"
#include "./nodes.h"
#include "./mesh_node.h"
#include "./label_node.h"
#include "./volume_node.h"
#include "./camera_node.h"
#include "./utils/persister.h"

BOOST_CLASS_EXPORT_GUID(LabelNode, "LabelNode")
BOOST_CLASS_EXPORT_GUID(MeshNode, "MeshNode")
BOOST_CLASS_EXPORT_GUID(VolumeNode, "VolumeNode")
BOOST_CLASS_EXPORT_GUID(CameraNode, "CameraNode")

DefaultSceneCreator::DefaultSceneCreator(std::shared_ptr<Nodes> nodes,
                                         std::shared_ptr<Labels> labels)
  : nodes(nodes), labels(labels)
{
}

void DefaultSceneCreator::create()
{
  std::vector<std::shared_ptr<Node>> sceneNodes;
  // addMeshNodesTo(sceneNodes);
  // addLabelNodesTo(sceneNodes);
  Eigen::Affine3f trans(
      Eigen::AngleAxisf(0.5 * M_PI, Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(-0.5 * M_PI, Eigen::Vector3f::UnitX()));
  sceneNodes.push_back(std::make_shared<VolumeNode>(
      "assets/datasets/v1.mhd", "assets/transferfunctions/v1-custom.gra",
      trans.matrix(), true));
  // addMultiVolumeNodesTo(sceneNodes);

  Persister::save(sceneNodes, "config/scene.xml");

  // nodes->addSceneNodesFrom("config/scene.xml");
  for (auto &node : sceneNodes)
    nodes->addNode(node);
}

void DefaultSceneCreator::addMeshNodesTo(
    std::vector<std::shared_ptr<Node>> &sceneNodes)
{
  const std::string filename = "assets/human-edited.dae";
  Importer importer;

  for (unsigned int meshIndex = 0; meshIndex < 2; ++meshIndex)
  {
    auto mesh = importer.import(filename, meshIndex);
    auto transformation = importer.getTransformationFor(filename, meshIndex);
    auto node = new MeshNode(filename, meshIndex, mesh, transformation);
    sceneNodes.push_back(std::unique_ptr<MeshNode>(node));
  }
}

void DefaultSceneCreator::addLabelNodesTo(
    std::vector<std::shared_ptr<Node>> &sceneNodes)
{
  auto label = Label(1, "Shoulder", Eigen::Vector3f(0.174f, 0.55f, 0.034f));
  sceneNodes.push_back(std::make_shared<LabelNode>(label));

  auto label2 = Label(2, "Ellbow", Eigen::Vector3f(0.34f, 0.322f, -0.007f));
  sceneNodes.push_back(std::make_shared<LabelNode>(label2));

  auto label3 = Label(3, "Wound", Eigen::Vector3f(0.262f, 0.422f, 0.058f),
                      Eigen::Vector2i(128, 128));
  sceneNodes.push_back(std::make_shared<LabelNode>(label3));

  auto label4 = Label(4, "Wound 2", Eigen::Vector3f(0.034f, 0.373f, 0.141f));
  sceneNodes.push_back(std::make_shared<LabelNode>(label4));
}

void DefaultSceneCreator::addMultiVolumeNodesTo(
    std::vector<std::shared_ptr<Node>> &sceneNodes)
{
  sceneNodes.push_back(std::make_shared<VolumeNode>(
      "assets/datasets/GRCH_Abdomen.mhd",
      "assets/transferfunctions/scapula4.gra", Eigen::Matrix4f::Identity()));
  sceneNodes.push_back(std::make_shared<VolumeNode>(
      "assets/datasets/GRCH_Schaedel_fein_H31.mhd",
      "assets/transferfunctions/scapula4.gra", Eigen::Matrix4f::Identity()));
}

