#ifndef SRC_PLACEMENT_PERSISTENT_CONSTRAINT_UPDATER_H_

#define SRC_PLACEMENT_PERSISTENT_CONSTRAINT_UPDATER_H_

#include <memory>
#include <map>
#include <Eigen/Core>

class ConstraintUpdater;

/**
 * \brief
 *
 *
 */
class PersistentConstraintUpdater
{
 public:
  PersistentConstraintUpdater(
      std::shared_ptr<ConstraintUpdater> constraintUpdater);

  void updateConstraints(int labelId, Eigen::Vector2i anchorForBuffer,
                         Eigen::Vector2i labelSizeForBuffer);
  void setPosition(int labelId, Eigen::Vector2i position);
  void clear();

 private:
  struct PlacedLabelInfo
  {
    Eigen::Vector2i size;
    Eigen::Vector2i anchorPosition;
    Eigen::Vector2i labelPosition;
  };

  std::shared_ptr<ConstraintUpdater> constraintUpdater;

  std::map<int, PlacedLabelInfo> placedLabels;
};

#endif  // SRC_PLACEMENT_PERSISTENT_CONSTRAINT_UPDATER_H_
