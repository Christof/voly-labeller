#include "./labeller.h"

namespace Placement
{

Labeller::Labeller(std::shared_ptr<Labels> labels) : labels(labels)
{
}

void Labeller::initialize(
    std::shared_ptr<CudaTextureMapper> occupancySummedAreaTable)
{
  this->occupancySummedAreaTable = occupancySummedAreaTable;
}

std::map<int, Eigen::Vector3f>
Labeller::update(const LabellerFrameData &frameData)
{
  return std::map<int, Eigen::Vector3f>();
}

}  // namespace Placement
