#include "./forces_visualizer_node.h"
#include <Eigen/Geometry>
#include <QDebug>

ForcesVisualizerNode::ForcesVisualizerNode(
    std::shared_ptr<Forces::Labeller> labeller)
  : labeller(labeller)
{
  persistable = false;
}

void ForcesVisualizerNode::render(Graphics::Gl *gl,
                                  std::shared_ptr<Graphics::Managers> managers,
                                  RenderData renderData)
{
  for (auto &label : labeller->getLabels())
  {
    for (auto &forcePair : label.forces)
    {
      auto force = forcePair.first;
      if (!connectors.count(force->name))
      {
        auto connector = std::make_shared<Graphics::Connector>(
            Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0));
        connector->zOffset = -0.02f;
        connector->color.head<3>() = forcePair.first->color;
        connectors[force->name] = connector;
      }

      renderForce(connectors[force->name], label.labelPosition2D,
                  forcePair.second, gl, managers, renderData);
    }
  }
}

void ForcesVisualizerNode::renderForce(
    std::shared_ptr<Graphics::Connector> connector,
    Eigen::Vector2f labelPosition, Eigen::Vector2f force, Graphics::Gl *gl,
    std::shared_ptr<Graphics::Managers> managers, RenderData renderData)
{
  auto length = force.norm() * 10;
  auto rotation = Eigen::Quaternionf::FromTwoVectors(
      Eigen::Vector3f::UnitX(), Eigen::Vector3f(force.x(), force.y(), 0));
  auto translation = Eigen::Vector3f(labelPosition.x(), labelPosition.y(), 0);
  Eigen::Affine3f connectorTransform(Eigen::Translation3f(translation) *
                                     rotation * Eigen::Scaling(length));

  // Use inverse view and projection matrices to get everything directly into
  // view space.
  renderData.modelMatrix = renderData.viewMatrix.inverse() *
                           renderData.projectionMatrix.inverse() *
                           connectorTransform.matrix();

  connector->render(gl, managers, renderData);
}

