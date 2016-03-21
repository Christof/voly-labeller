#include "./persistent_constraint_updater.h"
#include "./constraint_updater.h"

PersistentConstraintUpdater::PersistentConstraintUpdater(
    std::shared_ptr<ConstraintUpdater> constraintUpdater)
  : constraintUpdater(constraintUpdater)
{
}

void PersistentConstraintUpdater::updateConstraints(
    int labelId, Eigen::Vector2i anchorForBuffer,
    Eigen::Vector2i labelSizeForBuffer)
{
  constraintUpdater->clear();

  for (auto &placedLabelPair : placedLabels)
  {
    auto &placedLabel = placedLabelPair.second;
    constraintUpdater->drawConstraintRegionFor(
        anchorForBuffer, labelSizeForBuffer, placedLabel.anchorPosition,
        placedLabel.labelPosition, placedLabel.size);
  }

  placedLabels[labelId] = PlacedLabelInfo{ labelSizeForBuffer, anchorForBuffer,
                                           Eigen::Vector2i(-1, -1) };
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

