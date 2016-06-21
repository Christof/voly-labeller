#include "./label_positions.h"

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

int LabelPositions::size()
{
  return positionsNDC.size();
}

int LabelPositions::count(int labelId)
{
  return positionsNDC.count(labelId);
}

Eigen::Vector3f LabelPositions::getNDCFor(int labelId)
{
  return positionsNDC[labelId];
}

Eigen::Vector3f LabelPositions::get3dFor(int labelId)
{
  return positions3d[labelId];
}
