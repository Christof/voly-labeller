#include "./label_state.h"

namespace Forces
{
LabelState::LabelState(int id, std::string text, Eigen::Vector3f anchorPosition)
  : id(id), anchorPosition(anchorPosition), text(text)
{
}
}  // namespace Forces
