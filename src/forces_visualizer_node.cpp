#include "./forces_visualizer_node.h"
#include <Eigen/Geometry>
#include <QDebug>
#include "./connector.h"

ForcesVisualizerNode::ForcesVisualizerNode(
    std::shared_ptr<Forces::Labeller> labeller)
  : labeller(labeller)
{
  connector = std::make_shared<Connector>(Eigen::Vector3f(0, 0, 0),
                                          Eigen::Vector3f(1, 0, 0));
  connector->color = Eigen::Vector4f(1.0f, 0.8f, 0, 1);
}

void ForcesVisualizerNode::render(Gl *gl, RenderData renderData)
{
  for (auto &label : labeller->getLabels())
  {
    for (auto &forcePair : label.forces)
    {
      connector->color.head<3>() = forcePair.first->color;
      renderForce(label.labelPosition, forcePair.second, gl, renderData);
    }
  }
}

void ForcesVisualizerNode::renderForce(Eigen::Vector3f labelPosition,
                                       Eigen::Vector2f force, Gl *gl,
                                       RenderData renderData)
{
  auto length = force.norm() * 10;
  auto rotation = Eigen::Quaternionf::FromTwoVectors(
      Eigen::Vector3f::UnitX(), Eigen::Vector3f(force.x(), force.y(), 0));
  Eigen::Affine3f connectorTransform(Eigen::Translation3f(labelPosition) *
                                     rotation * Eigen::Scaling(length));
  renderData.modelMatrix = connectorTransform.matrix();

  connector->render(gl, renderData);
}
