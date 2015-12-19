#ifndef SRC_APPLICATION_H_

#define SRC_APPLICATION_H_

#include <QtGui/QGuiApplication>
#include <QObject>
#include "./placement/labeller.h"
#include "./forces/labeller.h"

class InvokeManager;
class Node;
class Nodes;
class Labels;
class TextureMapperManager;
class TextureMapperManagerController;

/**
 * \brief
 *
 *
 */
class Application : public QObject
{
  Q_OBJECT;

 public:
  Application(int &argc, char **argv);
  virtual ~Application();

  int execute();

 private:
  QGuiApplication application;
  std::shared_ptr<InvokeManager> invokeManager;
  std::shared_ptr<Nodes> nodes;
  std::shared_ptr<Labels> labels;
  std::shared_ptr<Forces::Labeller> forcesLabeller;
  std::shared_ptr<Placement::Labeller> placementLabeller;
  std::shared_ptr<TextureMapperManager> textureMapperManager;
  std::shared_ptr<TextureMapperManagerController>
      textureMapperManagerController;

  void onNodesChanged(std::shared_ptr<Node> node);
  void onLabelChangedUpdateLabelNodes(Labels::Action action,
                                      const Label &label);
};

#endif  // SRC_APPLICATION_H_
