#include "./default_scene_creator.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>
#include <random>
#include "./importer.h"
#include "./nodes.h"
#include "./mesh_node.h"
#include "./meshes_node.h"
#include "./label_node.h"
#include "./volume_node.h"
#include "./camera_node.h"
#include "./utils/persister.h"

BOOST_CLASS_EXPORT_GUID(LabelNode, "LabelNode")
BOOST_CLASS_EXPORT_GUID(MeshNode, "MeshNode")
BOOST_CLASS_EXPORT_GUID(MeshesNode, "MeshesNode")
BOOST_CLASS_EXPORT_GUID(VolumeNode, "VolumeNode")
BOOST_CLASS_EXPORT_GUID(CameraNode, "CameraNode")

DefaultSceneCreator::DefaultSceneCreator(std::shared_ptr<Nodes> nodes,
                                         std::shared_ptr<Labels> labels)
  : nodes(nodes), labels(labels)
{
}

void addPedestrianNodes(std::vector<std::shared_ptr<Node>> &sceneNodes)
{
  Eigen::Affine3f trans(
      Eigen::AngleAxisf(-0.5 * M_PI, Eigen::Vector3f::UnitX()));
  sceneNodes.push_back(std::make_shared<VolumeNode>(
      "assets/datasets/fussgaenger_graz_iterative.mhd",
      "assets/transferfunctions/scapula4.gra", trans.matrix(), true));

  const std::string filename = "assets/models/RekoCT2_01-8.dae";
  Importer importer;

  unsigned int meshIndex = 0;
  auto mesh = importer.import(filename, meshIndex);
  auto transformation = importer.getTransformationFor(filename, meshIndex);
  auto node = new MeshNode(filename, meshIndex, mesh, transformation);
  sceneNodes.push_back(std::unique_ptr<MeshNode>(node));
}

void addLIDCIDRINodes(std::vector<std::shared_ptr<Node>> &sceneNodes)
{
  Eigen::Affine3f trans(
      Eigen::Scaling(2.0f) *
      Eigen::Translation3f(Eigen::Vector3f(0.15, 0.145f, 0.15)) *
      Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(0.5 * M_PI, Eigen::Vector3f::UnitX()));

  sceneNodes.push_back(std::make_shared<VolumeNode>(
      "assets/datasets/LIDC-IDRI-0469_lung2.mha",
      "assets/transferfunctions/LIDC-IDRI-0469.gra", trans.matrix(), false));
  sceneNodes.push_back(std::make_shared<VolumeNode>(
      "assets/datasets/LIDC-IDRI-0469_lesion1_mask_extracted_blurred.mha",
      "assets/transferfunctions/LIDC_IDRI_lesions_mask.gra", trans.matrix(),
      false));
  sceneNodes.push_back(std::make_shared<VolumeNode>(
      "assets/datasets/LIDC-IDRI-0469_lesion2_mask_extracted_blurred.mha",
      "assets/transferfunctions/LIDC_IDRI_lesions_mask.gra", trans.matrix(),
      false));
  sceneNodes.push_back(std::make_shared<VolumeNode>(
      "assets/datasets/LIDC-IDRI-0469_lesion3_mask_extracted_blurred.mha",
      "assets/transferfunctions/LIDC_IDRI_lesions_mask.gra", trans.matrix(),
      false));
  sceneNodes.push_back(std::make_shared<VolumeNode>(
      "assets/datasets/LIDC-IDRI-0469_lesion4_mask_extracted_blurred.mha",
      "assets/transferfunctions/LIDC_IDRI_lesions_mask.gra", trans.matrix(),
      false));
}

void addSponza(std::vector<std::shared_ptr<Node>> &sceneNodes)
{
  const std::string filename = "assets/models/crytek-sponza/sponza.obj";
  Eigen::Affine3f scaling(Eigen::Scaling(0.01f));
  Eigen::Matrix4f scalingMatrix = scaling.matrix();

  sceneNodes.push_back(std::make_shared<MeshesNode>(filename, scalingMatrix));
}

void addJetEngine(std::vector<std::shared_ptr<Node>> &sceneNodes)
{
  const std::string filename = "assets/models/jet_engine.dae";
  Eigen::Affine3f trans(
      Eigen::Translation3f(Eigen::Vector3f(0, 0, 1)) * Eigen::Scaling(0.01f) *
      Eigen::AngleAxisf(-0.5 * M_PI, Eigen::Vector3f::UnitX()));
  Eigen::Matrix4f matrix = trans.matrix();

  sceneNodes.push_back(std::make_shared<MeshesNode>(filename, matrix));
}

void addPlane(std::vector<std::shared_ptr<Node>> &sceneNodes)
{
  const std::string filename = "assets/models/plane.dae";
  Eigen::Affine3f trans(Eigen::Scaling(0.5f) *
                        Eigen::AngleAxisf(-M_PI, Eigen::Vector3f::UnitY()));
  Eigen::Matrix4f matrix = trans.matrix();

  sceneNodes.push_back(std::make_shared<MeshesNode>(filename, matrix));
}

void addArtificial(std::vector<std::shared_ptr<Node>> &sceneNodes)
{
  const std::string filename = "assets/models/artificial.dae";
  Eigen::Affine3f scaling(Eigen::Scaling(1.0f));
  Eigen::Matrix4f scalingMatrix = scaling.matrix();

  sceneNodes.push_back(std::make_shared<MeshesNode>(filename, scalingMatrix));
}

void add100Labels(std::vector<std::shared_ptr<Node>> &sceneNodes)
{
  Eigen::Affine3f trans(
      Eigen::AngleAxisf(-0.5 * M_PI, Eigen::Vector3f::UnitX()));
  sceneNodes.push_back(std::make_shared<VolumeNode>(
      "assets/datasets/MANIX.mhd", "assets/transferfunctions/scapula2.gra",
      trans.matrix(), true));

  std::default_random_engine gen;
  std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
  for (int i = 0; i < 100; ++i)
  {
    auto label = Label(i, "L " + std::to_string(i),
                       Eigen::Vector3f(dist(gen), dist(gen), dist(gen)));
    label.size = Eigen::Vector2f(64, 32);
    sceneNodes.push_back(std::make_shared<LabelNode>(label));
  }
}

void addKnife(std::vector<std::shared_ptr<Node>> &sceneNodes)
{
  Eigen::Affine3f trans(
      Eigen::Translation3f(Eigen::Vector3f(0, -1.4, 0)) *
      Eigen::AngleAxisf(-0.5 * M_PI, Eigen::Vector3f::UnitX()));
  sceneNodes.push_back(std::make_shared<VolumeNode>(
      "assets/datasets/heidelberg_delikt_messer_masked3.mha",
      "assets/transferfunctions/Siemens_Pulmo3D_scapula.gra", trans.matrix(),
      false));
  sceneNodes.push_back(std::make_shared<VolumeNode>(
      "assets/datasets/heidelberg_delikt_messer_air2.mha",
      "assets/transferfunctions/air_cavities_6.gra", trans.matrix(), false));
}

void DefaultSceneCreator::create()
{
  std::vector<std::shared_ptr<Node>> sceneNodes;
  // addMeshNodesTo(sceneNodes);
  // addLabelNodesTo(sceneNodes);
  Eigen::Affine3f trans(
      Eigen::AngleAxisf(-0.5 * M_PI, Eigen::Vector3f::UnitX()));
  sceneNodes.push_back(std::make_shared<VolumeNode>(
      "assets/datasets/dice.mha", "assets/transferfunctions/dice4.gra",
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
  Eigen::Affine3f trans(
      Eigen::Translation3f(Eigen::Vector3f(0, 0.6f, 0)) *
      Eigen::AngleAxisf(-0.5 * M_PI, Eigen::Vector3f::UnitX()));
  sceneNodes.push_back(std::make_shared<VolumeNode>(
      "assets/datasets/GRCH_Abdomen.mhd",
      "assets/transferfunctions/scapula2.gra", trans.matrix()));
  sceneNodes.push_back(std::make_shared<VolumeNode>(
      "assets/datasets/GRCH_Schaedel_fein_H31.mhd",
      "assets/transferfunctions/scapula2.gra", trans.matrix()));
}

