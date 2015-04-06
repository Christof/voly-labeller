#ifndef SRC_LABEL_H_

#define SRC_LABEL_H_

#include <Eigen/Core>

/**
 * \brief Holds basic data of a label: id, text, anchor position
 *
 */
struct Label
{
 public:
  Label(int id, std::string text, Eigen::Vector3f anchorPosition)
    : id(id), text(text), anchorPosition(anchorPosition)
  {
  }

  int id;
  std::string text;
  Eigen::Vector3f anchorPosition;
};

#endif  // SRC_LABEL_H_
