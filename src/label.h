#ifndef SRC_LABEL_H_

#define SRC_LABEL_H_

#include <Eigen/Core>
#include <boost/serialization/nvp.hpp>
#include <string>

/**
 * \brief Holds basic data of a label: id, text, anchor position
 *
 */
struct Label
{
 public:
  /**
   * \brief Default constructor is only implemented for persistence
   *
   * \warn This should not be used.
   */
  Label() : Label(-1, "", Eigen::Vector3f())
  {
  }

  Label(int id, std::string text, Eigen::Vector3f anchorPosition)
    : id(id), text(text), anchorPosition(anchorPosition)
  {
  }

  int id;
  std::string text;
  Eigen::Vector3f anchorPosition;
};

namespace boost
{
namespace serialization
{
template <class Archive>
void serialize(Archive &ar, Label &label, const unsigned int version)
{
  ar &BOOST_SERIALIZATION_NVP(label.id);
  ar &BOOST_SERIALIZATION_NVP(label.text);
  ar &BOOST_SERIALIZATION_NVP(label.anchorPosition);
}
}  // namespace serialization
}  // namespace boost

#endif  // SRC_LABEL_H_
