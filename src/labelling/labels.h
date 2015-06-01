#ifndef SRC_LABELLING_LABELS_H_

#define SRC_LABELLING_LABELS_H_

#include <map>
#include <vector>
#include <functional>
#include "./label.h"

/**
 * \brief
 *
 *
 */
class Labels
{
 public:
  Labels() = default;

  std::function<void()>
  subscribe(std::function<void(const Label &)> subscriber);

  void add(Label label);
  std::vector<Label> getLabels();
  void updateAnchor(int id, Eigen::Vector3f anchorPosition);
 private:
  std::map<int, Label> labels;
  std::vector<std::function<void(const Label &)>> subscribers;

  void notify(const Label& label);
};

#endif  // SRC_LABELLING_LABELS_H_
