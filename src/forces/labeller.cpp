#include "./labeller.h"
#include <Eigen/LU>
#include <map>
#include <vector>
#include <string>
#include "./center_force.h"
#include "../math/eigen.h"
#include "./anchor_force.h"
#include "./label_collision_force.h"
#include "./lines_crossing_force.h"

namespace Forces
{
Labeller::Labeller(std::shared_ptr<Labels> labels) : labels(labels)
{
  srand(9);
  addForce(new CenterForce());
  addForce(new AnchorForce());
  addForce(new LabelCollisionForce());
  addForce(new LinesCrossingForce());

  unsubscribeLabelChanges = labels->subscribe(std::bind(
      &Labeller::setLabel, this, std::placeholders::_1, std::placeholders::_2));
}

Labeller::~Labeller()
{
  unsubscribeLabelChanges();
}

void Labeller::resize(int width, int height)
{
  size = Eigen::Vector2f(width, height);
}

void Labeller::setLabel(Labels::Action action, const Label &label)
{
  auto predicate = [label](const LabelState labelState)
  {
    return labelState.id == label.id;
  };

  auto oldLabelState =
      std::find_if(labelStates.begin(), labelStates.end(), predicate);

  LabelState labelState(label.id, label.text, label.anchorPosition, label.size);
  if (oldLabelState != labelStates.end() &&
      oldLabelState->anchorPosition == label.anchorPosition)
    labelState.labelPosition = oldLabelState->labelPosition;

  labelStates.erase(
      std::remove_if(labelStates.begin(), labelStates.end(), predicate),
      labelStates.end());

  if (action != Labels::Action::Delete)
    labelStates.push_back(labelState);
}

void Labeller::updateLabel(int id, Eigen::Vector3f anchorPosition)
{
  for (auto &labelState : labelStates)
  {
    if (labelState.id != id)
      continue;

    labelState.anchorPosition = anchorPosition;
    labelState.labelPosition = 1.3f * anchorPosition.normalized();
  }
}

std::map<int, Eigen::Vector3f>
Labeller::update(const LabellerFrameData &frameData)
{
  std::map<int, Eigen::Vector3f> positions;

  for (auto &force : forces)
    force->beforeAll(labelStates);

  Eigen::Matrix4f inverseViewProjection = frameData.viewProjection.inverse();

  for (auto &label : labelStates)
  {
    auto anchor2D = frameData.project(label.anchorPosition);
    label.anchorPosition2D = anchor2D.head<2>();

    auto label2D = frameData.project(label.labelPosition);
    label.labelPosition2D = label2D.head<2>();
    label.labelPositionDepth = label2D.z();

    auto forceOnLabel = Eigen::Vector2f(0, 0);
    for (auto &force : forces)
      forceOnLabel += force->calculateForce(label, labelStates, frameData);

    if (updatePositions)
      label.labelPosition2D += forceOnLabel * frameData.frameTime;

    Eigen::Vector4f reprojected =
        inverseViewProjection * Eigen::Vector4f(label.labelPosition2D.x(),
                                                label.labelPosition2D.y(),
                                                label.labelPositionDepth, 1);
    reprojected /= reprojected.w();

    label.labelPosition = reprojected.head<3>();

    enforceAnchorDepthForLabel(label, frameData.view);

    positions[label.id] = label.labelPosition;
  }

  return positions;
}

std::vector<LabelState> Labeller::getLabels()
{
  return labelStates;
}

template <class T> void Labeller::addForce(T *force)
{
  forces.push_back(std::unique_ptr<T>(force));
}

void Labeller::enforceAnchorDepthForLabel(LabelState &label,
                                          const Eigen::Matrix4f &viewMatrix)
{
  Eigen::Vector4f anchorView = mul(viewMatrix, label.anchorPosition);
  Eigen::Vector4f labelView = mul(viewMatrix, label.labelPosition);
  labelView.z() = anchorView.z();

  label.labelPosition = (viewMatrix.inverse() * labelView).head<3>();
}

}  // namespace Forces
