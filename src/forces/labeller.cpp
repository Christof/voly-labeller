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
#include "./placement_force.h"

namespace Forces
{
Labeller::Labeller(std::shared_ptr<Labels> labels) : labels(labels)
{
  srand(9);
  addForce(new CenterForce(), false);
  addForce(new AnchorForce(), false);
  addForce(new LabelCollisionForce(), false);
  addForce(new LinesCrossingForce(), false);
  addForce(new PlacementForce());

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

  Eigen::Vector2f sizeNDC = 2.0f * label.size.cwiseQuotient(size);
  LabelState labelState(label.id, label.text, label.anchorPosition, sizeNDC);
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
Labeller::update(const LabellerFrameData &frameData,
                 std::map<int, Eigen::Vector3f> placementPositionsNDC,
                 std::map<int, Eigen::Vector3f> placementPositions)
{
  std::map<int, Eigen::Vector3f> positions;

  for (auto &force : forces)
    force->beforeAll(labelStates);

  Eigen::Matrix4f inverseViewProjection = frameData.viewProjection.inverse();

  for (auto &label : labelStates)
  {
    if (placementPositions.count(label.id))
    {
      label.placementPosition = placementPositions[label.id];
      label.placementPosition2D = placementPositionsNDC[label.id].head<2>();
      label.labelPositionDepth = placementPositionsNDC[label.id].z();
    }

    label.update2dValues(frameData);

    auto forceOnLabel = Eigen::Vector2f(0, 0);
    for (auto &force : forces)
      forceOnLabel += force->calculateForce(label, labelStates, frameData);

    if (updatePositions && forceOnLabel.norm() > epsilon)
    {
      auto delta = forceOnLabel * frameData.frameTime;
      label.labelPosition2D += delta;
    }

    Eigen::Vector3f positionNDC(label.labelPosition2D.x(),
                                label.labelPosition2D.y(),
                                label.labelPositionDepth);
    Eigen::Vector3f reprojected = project(inverseViewProjection, positionNDC);
    reprojected.z() = label.placementPosition.z();

    label.labelPosition = reprojected;
    positions[label.id] = positionNDC;
  }

  return positions;
}

void Labeller::setPositions(const LabellerFrameData &frameData,
                            std::map<int, Eigen::Vector3f> positions)
{
  for (auto &label : labelStates)
  {
    if (positions.count(label.id))
      label.labelPosition = positions[label.id];

    label.update2dValues(frameData);
  }
}

std::vector<LabelState> Labeller::getLabels()
{
  return labelStates;
}

template <class T> void Labeller::addForce(T *force, bool enabled)
{
  force->isEnabled = enabled;
  forces.push_back(std::unique_ptr<T>(force));
}

}  // namespace Forces
