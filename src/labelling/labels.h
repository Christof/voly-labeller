#ifndef SRC_LABELLING_LABELS_H_

#define SRC_LABELLING_LABELS_H_

#include <map>
#include <vector>
#include <functional>
#include "./label.h"
#include "./labels_container.h"

/**
 * \brief Collection of all labels
 *
 * Provides an observer mechanism to subscribe to changes of
 * labels or the addition of a new one.
 */
class Labels : public LabelsContainer
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

  virtual void add(Label label);
  void update(Label label);
  void remove(Label label);

 private:
  std::vector<std::function<void(Action action, const Label &)>> subscribers;

  void notify(Action action, const Label &label);
};

#endif  // SRC_LABELLING_LABELS_H_
