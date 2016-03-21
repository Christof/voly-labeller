#ifndef SRC_LABELLING_LABELS_CONTAINER_H_

#define SRC_LABELLING_LABELS_CONTAINER_H_

#include <vector>
#include <map>
#include "./label.h"

/**
 * \brief Collection of lables with lookup by id
 *
 * This is used by Labels and also to work on a subset of all labels,
 * e.g. labels of a layer or in cluster.
 */
class LabelsContainer
{
 public:
  LabelsContainer() = default;

  virtual void add(Label label);

  Label getById(int id);

  int count();

  void clear();

  std::vector<Label> getLabels();

 protected:
  std::map<int, Label> labels;
};

#endif  // SRC_LABELLING_LABELS_CONTAINER_H_
