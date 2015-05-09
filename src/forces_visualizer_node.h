#ifndef SRC_FORCES_VISUALIZER_NODE_H_

#define SRC_FORCES_VISUALIZER_NODE_H_

#include <memory>
#include "./node.h"
#include "./forces/labeller.h"

class Connector;

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

  void render(Gl *gl, RenderData renderData);

 private:
  std::shared_ptr<Forces::Labeller> labeller;

  std::shared_ptr<Connector> connector;

  void renderForce(Eigen::Vector3f labelPosition, Eigen::Vector2f force, Gl *gl,
                   RenderData renderData);
};

#endif  // SRC_FORCES_VISUALIZER_NODE_H_
