#ifndef SRC_NODES_H_

#define SRC_NODES_H_

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "./node.h"
#include "./graphics/render_data.h"
#include "./graphics/managers.h"
#include "./graphics/gl.h"

class LabelNode;
class CameraNode;
class MeshNode;

/**
 * \brief Manages a collection of nodes which is rendered
 * in the scene
 *
 * The collection can be extended by loading a scene
 * file (Nodes::addSceneNodesFrom), which was previously saved
 * using Nodes::saveSceneTo.
 *
 * Also an asset file can be directly added using Nodes::importFrom.
 */
class Nodes
{
 public:
  Nodes();
  ~Nodes();
  void render(Graphics::Gl *gl, std::shared_ptr<Graphics::Managers> managers,
              RenderData renderData);
  void renderLabels(Graphics::Gl *gl,
                    std::shared_ptr<Graphics::Managers> managers,
                    RenderData renderData);
  void renderOverlays(Graphics::Gl *gl,
                      std::shared_ptr<Graphics::Managers> managers,
                      RenderData renderData);

  std::vector<std::shared_ptr<LabelNode>> getLabelNodes();
  void removeNode(std::shared_ptr<Node> node);
  std::vector<std::shared_ptr<Node>> getNodes();
  std::vector<std::shared_ptr<Node>> getNodesForObb();
  std::shared_ptr<CameraNode> getCameraNode();
  void setCameraNode(std::shared_ptr<CameraNode> node);

  void addSceneNodesFrom(std::string filename);
  void importMeshFrom(std::string filename);
  void saveSceneTo(std::string filename);
  void importVolume(std::string volumeFilename,
                    std::string transferFunctionFilename);
  void clear();
  void clearForShutdown();
  void toggleBoundingVolumes();
  void toggleCoordinateSystem();
  void toggleCameraOriginVisualizer();

  void addNode(std::shared_ptr<Node> node);
  void addForcesVisualizerNode(std::shared_ptr<Node> node);
  void removeForcesVisualizerNode();

  /**
   * \brief Applies the given transformation to (nearly) all nodes
   *
   * %Nodes considered are VolumeNode%s, MeshNode%s and LabelNode%s.
   *
   */
  void applyTransformationToAllNodes(Eigen::Matrix4f transformation);

  void setOnNodeAdded(std::function<void(std::shared_ptr<Node>)> onNodeAdded);

  std::string getSceneName() const;

 private:
  std::vector<std::shared_ptr<Node>> nodes;
  bool showBoundingVolumes = false;
  std::vector<std::shared_ptr<Node>> obbNodes;
  std::vector<std::shared_ptr<LabelNode>> labelNodes;
  std::shared_ptr<Node> coordinateSystemNode;
  std::shared_ptr<Node> forcesVisualizerNode;
  std::shared_ptr<MeshNode> cameraOriginVisualizerNode;
  std::shared_ptr<CameraNode> cameraNode;
  std::string sceneName;

  std::function<void(std::shared_ptr<Node>)> onNodeAdded;

  void createCameraOriginVisualizer();
  void
  renderCameraOriginVisualizer(Graphics::Gl *gl,
                               std::shared_ptr<Graphics::Managers> managers,
                               const RenderData &renderData);
};

#endif  // SRC_NODES_H_
