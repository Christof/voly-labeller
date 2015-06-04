#ifndef SRC_LABELLING_LABELS_H_

#define SRC_LABELLING_LABELS_H_

#include <map>
#include <vector>
#include <functional>
#include "./label.h"

/**
 * \brief Collection of all labels
 *
 * Provides an observer mechanism to subscribe to changes of
 * labels or the addition of a new one.
 */
class Labels
{
 public:
  enum Action
  {
    Add,
    Update,
    Delete
  };

  Labels() = default;

  std::function<void()>
  subscribe(std::function<void(Action action, const Label &)> subscriber);

  void add(Label label);
  void update(Label label);
  std::vector<Label> getLabels();
  Label getById(int id);
  int count();
  void updateAnchor(int id, Eigen::Vector3f anchorPosition);

 private:
  std::map<int, Label> labels;
  std::vector<std::function<void(Action action, const Label &)>> subscribers;

  void notify(Action action, const Label &label);
};

#endif  // SRC_LABELLING_LABELS_H_
