#include "./persistent_constraint_updater.h"
#include <QLoggingCategory>
#include <chrono>
#include <vector>
#include "./constraint_updater.h"

QLoggingCategory pcuChan("Placement.PersistentConstraintUpdater");

PersistentConstraintUpdater::PersistentConstraintUpdater(
    std::shared_ptr<ConstraintUpdater> constraintUpdater)
  : constraintUpdater(constraintUpdater)
{
}

void PersistentConstraintUpdater::setAnchorPositions(
    std::vector<Eigen::Vector2i> anchorPositions)
{
  this->anchorPositions = anchorPositions;
}

void PersistentConstraintUpdater::updateConstraints(
    int labelId, Eigen::Vector2i anchorForBuffer,
    Eigen::Vector2i labelSizeForBuffer)
{
  auto startTime = std::chrono::high_resolution_clock::now();
  constraintUpdater->clear();

  constraintUpdater->drawRegionsForAnchors(anchorPositions, labelSizeForBuffer);

  for (auto &placedLabelPair : placedLabels)
  {
    auto &placedLabel = placedLabelPair.second;
    constraintUpdater->drawConstraintRegionFor(
        anchorForBuffer, labelSizeForBuffer, placedLabel.anchorPosition,
        placedLabel.labelPosition, placedLabel.size);
  }

  constraintUpdater->finish();

  placedLabels[labelId] = PlacedLabelInfo{ labelSizeForBuffer, anchorForBuffer,
                                           Eigen::Vector2i(-1, -1) };

  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> diff = endTime - startTime;

  qCDebug(pcuChan) << "updateConstraints took" << diff.count() << "ms";
}

void PersistentConstraintUpdater::setPosition(int labelId,
                                              Eigen::Vector2i position)
{
  placedLabels[labelId].labelPosition = position;
}

void PersistentConstraintUpdater::clear()
{
  placedLabels.clear();
}

