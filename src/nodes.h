#ifndef SRC_NODES_H_

#define SRC_NODES_H_

#include <QObject>
#include <QUrl>
#include <string>
#include <vector>
#include <memory>
#include "./node.h"
#include "./graphics/render_data.h"
#include "./graphics/ha_buffer.h"
#include "./graphics/gl.h"

class LabelNode;

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
class Nodes : public QObject
{
  Q_OBJECT
 public:
  Nodes();
  void render(Graphics::Gl *gl, std::shared_ptr<Graphics::HABuffer> haBuffer,
              RenderData renderData);

  std::vector<std::shared_ptr<LabelNode>> getLabelNodes();
  void removeNode(std::shared_ptr<Node> node);
  std::vector<std::shared_ptr<Node>> getNodes();
 public slots:
  void addSceneNodesFrom(std::string filename);
  void addSceneNodesFrom(QUrl url);

  void importFrom(std::string filename);
  void importFrom(QUrl url);

  void saveSceneTo(QUrl url);
  void saveSceneTo(std::string filename);

  void clear();

  void toggleBoundingVolumes();

  void addNode(std::shared_ptr<Node> node);
 signals:
  void nodesChanged();

 private:
  std::vector<std::shared_ptr<Node>> nodes;
  bool showBoundingVolumes = false;
  std::vector<std::shared_ptr<Node>> obbNodes;
};

#endif  // SRC_NODES_H_
