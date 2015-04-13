#ifndef SRC_NODES_H_

#define SRC_NODES_H_

#include "./node.h"
#include "./render_data.h"
#include <QObject>
#include <QUrl>
#include <string>
#include <memory>

/**
 * \brief
 *
 *
 */
class Nodes : public QObject
{
  Q_OBJECT
 public:
  Nodes();
  void render(RenderData renderData);

 public slots:
  void addSceneNodesFrom(std::string filename);
  void addSceneNodesFrom(QUrl url);

 private:
  std::vector<std::shared_ptr<Node>> nodes;
};

#endif  // SRC_NODES_H_
