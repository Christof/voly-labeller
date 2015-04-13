#ifndef SRC_NODES_H_

#define SRC_NODES_H_

#include "./node.h"
#include "./render_data.h"
#include <QObject>
#include <QUrl>
#include <string>
#include <memory>

class Gl;

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
