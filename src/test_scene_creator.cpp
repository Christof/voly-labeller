#include "./test_scene_creator.h"
#include <string>
#include <vector>
#include "./importer.h"
#include "./nodes.h"
#include "./mesh_node.h"
#include "./label_node.h"
#include "./volume_node.h"
#include "./utils/persister.h"

BOOST_CLASS_EXPORT_GUID(LabelNode, "LabelNode")
BOOST_CLASS_EXPORT_GUID(MeshNode, "MeshNode")
BOOST_CLASS_EXPORT_GUID(VolumeNode, "VolumeNode")

TestSceneCreator::TestSceneCreator(std::shared_ptr<Nodes> nodes,
                                   std::shared_ptr<Labels> labels)
  : nodes(nodes), labels(labels)
{
}

void TestSceneCreator::create()
{
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

  meshNodes.push_back(
      std::make_shared<VolumeNode>("assets/datasets/MANIX.mhd"));

  Persister::save(meshNodes, "config/scene.xml");

  nodes->addSceneNodesFrom("config/scene.xml");

  for (auto &labelNode : nodes->getLabelNodes())
  {
    labels->add(labelNode->label);
  }
}

