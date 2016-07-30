#ifndef SRC_NODES_CONTROLLER_H_

#define SRC_NODES_CONTROLLER_H_

#include <QObject>
#include <QUrl>
#include <memory>
#include <string>

class Nodes;

/**
 * \brief Wrapper around Nodes to provide access to node loading and saving
 * functions to the UI
 *
 */
class NodesController : public QObject
{
  Q_OBJECT
  Q_PROPERTY(float anchorSize MEMBER labelAnchorSize NOTIFY anchorSizeChanged)
 public:
  explicit NodesController(std::shared_ptr<Nodes> nodes);

  float getAnchorSize();

 signals:
  void anchorSizeChanged();

 public slots:
  void addSceneNodesFrom(QUrl url);
  void importMeshFrom(QUrl url);
  void setVolumeToImport(QUrl url);
  void importVolume(QUrl transferFunctionUrl);

  void saveSceneTo(QUrl url);

  void clear();

  void toggleBoundingVolumes();
  void toggleCoordinateSystem();
  void toggleCameraOriginVisualizer();

  void changeAnchorSize(float size);

 private:
  std::shared_ptr<Nodes> nodes;

  std::string volumeToImport;

  float labelAnchorSize = 10.0f;
};

#endif  // SRC_NODES_CONTROLLER_H_
