#ifndef SRC_PLACEMENT_PERSISTENT_CONSTRAINT_UPDATER_H_

#define SRC_PLACEMENT_PERSISTENT_CONSTRAINT_UPDATER_H_

#include <Eigen/Core>
#include <memory>
#include <map>
#include <vector>

class ConstraintUpdater;

/**
 * \brief Uses the ConstraintUpdater to draw constraints and stores each given
 * label to be used as constraint for the next label
 *
 * The labels are stored and used until #clear is called. After a label has been
 * positioned #setPosition must be called to complete the information necessary
 * to use the label as constraint for the next one.
 *
 * The constraints are drawn by calling #updateConstraints.
 */
class PersistentConstraintUpdater
{
 public:
  PersistentConstraintUpdater(
      std::shared_ptr<ConstraintUpdater> constraintUpdater);

  void setAnchorPositions(std::vector<Eigen::Vector2f> anchorPositions);
  void updateConstraints(int labelId, Eigen::Vector2i anchorForBuffer,
                         Eigen::Vector2i labelSizeForBuffer);
  void setPosition(int labelId, Eigen::Vector2i position);
  void clear();
  void save();

 private:
  struct PlacedLabelInfo
  {
    Eigen::Vector2i size;
    Eigen::Vector2i anchorPosition;
    Eigen::Vector2i labelPosition;
  };

  std::shared_ptr<ConstraintUpdater> constraintUpdater;

  std::map<int, PlacedLabelInfo> placedLabels;
  std::vector<Eigen::Vector2f> anchorPositions;
  int index = 0;
  bool saveConstraints = false;
};

#endif  // SRC_PLACEMENT_PERSISTENT_CONSTRAINT_UPDATER_H_
