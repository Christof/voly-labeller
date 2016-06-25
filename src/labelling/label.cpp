#include "./label.h"
#include <string>
#include "./labeller_frame_data.h"

Label::Label() : Label(-1, "", Eigen::Vector3f())
{
}

Label::Label(int id, std::string text, Eigen::Vector3f anchorPosition,
             Eigen::Vector2i size)
  : id(id), text(text), anchorPosition(anchorPosition), size(size.cast<float>())
{
}
bool Label::operator==(const Label &other) const
{
  return id == other.id && text == other.text &&
         anchorPosition == other.anchorPosition && size == other.size;
}

bool Label::operator!=(const Label &other) const
{
  return id != other.id || text != other.text ||
         anchorPosition != other.anchorPosition || size != other.size;
}

bool Label::isAnchorInsideFieldOfView(
    const Eigen::Matrix4f &viewProjectionMatrix) const
{
  Eigen::Vector3f anchorNdc = project(viewProjectionMatrix, anchorPosition);

  return std::abs(anchorNdc.x()) <= 1.0f && std::abs(anchorNdc.y()) <= 1.0f &&
         std::abs(anchorNdc.z()) <= 1.0f;
}
