#ifndef SRC_NODES_CONTROLLER_H_

#define SRC_NODES_CONTROLLER_H_

#include <QObject>
#include <QUrl>
#include <memory>

class Nodes;

/**
 * \brief
 *
 *
 */
class NodesController : public QObject
{
  Q_OBJECT
 public:
  NodesController(std::shared_ptr<Nodes> nodes);

 public slots:
  void addSceneNodesFrom(QUrl url);
  void importMeshFrom(QUrl url);
  void setVolumeToImport(QUrl url);
  void importVolume(QUrl transferFunctionUrl);

  void saveSceneTo(QUrl url);

  void clear();

  void toggleBoundingVolumes();

 private:
  std::shared_ptr<Nodes> nodes;

  std::string volumeToImport;
};

#endif  // SRC_NODES_CONTROLLER_H_
