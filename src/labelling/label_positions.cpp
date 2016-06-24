#include "./label_positions.h"
#include <map>

LabelPositions::LabelPositions(std::map<int, Eigen::Vector3f> positionsNDC,
                               std::map<int, Eigen::Vector3f> positions3d)
  : positionsNDC(positionsNDC), positions3d(positions3d)
{
}

void LabelPositions::update(int labelId, Eigen::Vector3f positionNDC,
                            Eigen::Vector3f position3d)
{
  positionsNDC[labelId] = positionNDC;
  positions3d[labelId] = position3d;
}

int LabelPositions::size() const
{
  return positionsNDC.size();
}

int LabelPositions::count(int labelId) const
{
  return positionsNDC.count(labelId);
}

Eigen::Vector3f LabelPositions::getNDCFor(int labelId) const
{
  return positionsNDC.at(labelId);
}

Eigen::Vector3f LabelPositions::get3dFor(int labelId) const
{
  return positions3d.at(labelId);
}
