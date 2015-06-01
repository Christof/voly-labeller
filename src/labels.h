#ifndef SRC_LABELS_H_

#define SRC_LABELS_H_

#include <map>
#include <vector>
#include <functional>

struct Label;

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
 private:
  std::map<int, Label> labels;
  std::vector<std::function<void(const Label &)>> subscribers;

  void notify(const Label& label);
};

#endif  // SRC_LABELS_H_
