#ifndef SRC_LABELLING_LABELS_CONTAINER_H_

#define SRC_LABELLING_LABELS_CONTAINER_H_

#include <vector>
#include <map>
#include "./label.h"

/**
 * \brief
 *
 *
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
