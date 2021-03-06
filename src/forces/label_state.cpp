#include "./label_state.h"
#include <string>

namespace Forces
{
LabelState::LabelState(int id, std::string text, Eigen::Vector3f anchorPosition,
                       Eigen::Vector2f size)
  : id(id), anchorPosition(anchorPosition), size(size), text(text)
{
  labelPosition = 1.3f * anchorPosition.normalized();
}

void LabelState::update2dValues(const LabellerFrameData &frameData)
{
  anchorPosition2D = frameData.project2d(anchorPosition);
}

}  // namespace Forces
