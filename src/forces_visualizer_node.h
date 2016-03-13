#ifndef SRC_FORCES_VISUALIZER_NODE_H_

#define SRC_FORCES_VISUALIZER_NODE_H_

#include <memory>
#include <map>
#include <string>
#include "./node.h"
#include "./forces/labeller.h"
#include "./graphics/connector.h"
#include "./graphics/gl.h"

/**
 * \brief Node to visualize forces of a Forces::Labeller instance
 *
 * The forces are displayed as lines, with a displayed magnitude
 * 10 times larger than the real one.
 */
class ForcesVisualizerNode : public Node
{
 public:
  explicit ForcesVisualizerNode(std::shared_ptr<Forces::Labeller> labeller);

  virtual void render(Graphics::Gl *gl,
                      std::shared_ptr<Graphics::Managers> managers,
                      RenderData renderData);

 private:
  std::shared_ptr<Forces::Labeller> labeller;

  std::map<std::string, std::shared_ptr<Graphics::Connector>> connectors;

  void renderForce(std::shared_ptr<Graphics::Connector> connector,
                   Eigen::Vector2f labelPosition, Eigen::Vector2f force,
                   Graphics::Gl *gl,
                   std::shared_ptr<Graphics::Managers> managers,
                   RenderData renderData);
};

#endif  // SRC_FORCES_VISUALIZER_NODE_H_
