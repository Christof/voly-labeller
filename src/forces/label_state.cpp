#include "./label_state.h"
#include <string>

namespace Forces
{
LabelState::LabelState(int id, std::string text, Eigen::Vector3f anchorPosition)
  : id(id), anchorPosition(anchorPosition), text(text)
{
  labelPosition = 1.3f * anchorPosition.normalized();
}
}  // namespace Forces
