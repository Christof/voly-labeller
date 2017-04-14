#include "./labeller.h"
#include <Eigen/LU>
#include <map>
#include <deque>
#include <vector>
#include <string>
#include <algorithm>
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

LabelPositions Labeller::update(const LabellerFrameData &frameData,
                                const LabelPositions &placementPositions)
{
  LabelPositions positions;

  for (auto &force : forces)
    force->beforeAll(labelStates);

  Eigen::Matrix4f inverseViewProjection = frameData.viewProjection.inverse();

  for (auto &label : labelStates)
  {
    if (placementPositions.count(label.id))
    {
      label.placementPosition = placementPositions.get3dFor(label.id);
      auto placementNDC = placementPositions.getNDCFor(label.id);
      label.placementPosition2D = placementNDC.head<2>();
      label.labelPositionDepth = placementNDC.z();
    }

    label.update2dValues(frameData);

    if (updatePositions)
    {
      bool onPlacementResult = false;
      double remainingFrameTime = frameData.frameTime;
      LabellerFrameData partialFrameData = frameData;
      const double stepTime = 0.01;
      do
      {
        partialFrameData.frameTime = std::min(stepTime, remainingFrameTime);
        auto forceOnLabel = Eigen::Vector2f(0, 0);
        for (auto &force : forces)
          forceOnLabel +=
              force->calculateForce(label, labelStates, partialFrameData);

        auto delta =
            overallForceFactor * forceOnLabel * partialFrameData.frameTime;
        if (delta.norm() > epsilon)
        {
          label.labelPosition2D += delta;

          remainingFrameTime -= stepTime;
        }
        else
        {
          label.labelPosition2D = label.placementPosition2D;
          label.labelPosition = label.placementPosition;
          Eigen::Vector3f positionNDC(label.labelPosition2D.x(),
                                      label.labelPosition2D.y(),
                                      label.labelPositionDepth);
          positions.update(label.id, positionNDC, label.placementPosition);
          onPlacementResult = true;
          break;
        }
      } while (remainingFrameTime > 0);

      if (onPlacementResult)
        continue;
    }

    auto history = oldPositions[label.id];
    history.push_front(label.labelPosition2D);
    if (history.size() > 10)
      history.pop_back();

    Eigen::Vector2f sum(0, 0);
    for (auto pos : history)
      sum += pos;

    label.labelPosition2D = sum / history.size();

    Eigen::Vector3f positionNDC(label.labelPosition2D.x(),
                                label.labelPosition2D.y(),
                                label.labelPositionDepth);
    Eigen::Vector3f reprojected = project(inverseViewProjection, positionNDC);

    label.labelPosition = reprojected;
    positions.update(label.id, positionNDC, reprojected);
  }

  return positions;
}

void Labeller::setPositions(const LabellerFrameData &frameData,
                            LabelPositions placementPositions)
{
  for (auto &label : labelStates)
  {
    if (placementPositions.count(label.id))
    {
      label.placementPosition = placementPositions.get3dFor(label.id);
      auto placementNDC = placementPositions.getNDCFor(label.id);
      label.placementPosition2D = placementNDC.head<2>();
      label.labelPositionDepth = placementNDC.z();

      label.labelPosition2D = label.placementPosition2D;
      label.labelPosition = label.placementPosition;
    }

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
