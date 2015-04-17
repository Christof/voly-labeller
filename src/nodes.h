#ifndef SRC_NODES_H_

#define SRC_NODES_H_

#include <QObject>
#include <QUrl>
#include <string>
#include <vector>
#include <memory>
#include "./node.h"
#include "./render_data.h"

class Gl;

/**
 * \brief Manages a collection of nodes which is rendered
 * in the scene
 *
 * The collection can be extended by loading a scene
 * file (Nodes::addSceneNodesFrom), which was previously saved
 * using Nodes::saveSceneTo.
 *
 * Also an asset file can be dirctly added using Nodes::importFrom.
 */
class Nodes : public QObject
{
  Q_OBJECT
 public:
  Nodes();
  void render(Gl *gl, RenderData renderData);

 public slots:
  void addSceneNodesFrom(std::string filename);
  void addSceneNodesFrom(QUrl url);

  void importFrom(std::string filename);
  void importFrom(QUrl url);

  void saveSceneTo(QUrl url);
  void saveSceneTo(std::string filename);

  void clear();

 private:
  std::vector<std::shared_ptr<Node>> nodes;
};

#endif  // SRC_NODES_H_
