#include "./labeller.h"
#include <Eigen/Geometry>
#include "../utils/cuda_array_provider.h"
#include "./summed_area_table.h"

namespace Placement
{

Labeller::Labeller(std::shared_ptr<Labels> labels) : labels(labels)
{
}

void
Labeller::initialize(std::shared_ptr<CudaArrayProvider> occupancyTextureMapper)
{
  occupancySummedAreaTable =
      std::make_shared<SummedAreaTable>(occupancyTextureMapper);
}

std::map<int, Eigen::Vector3f>
Labeller::update(const LabellerFrameData &frameData)
{
  std::map<int, Eigen::Vector3f> result;
  if (!occupancySummedAreaTable.get())
    return result;

  Eigen::Matrix4f inverseViewProjection = frameData.viewProjection.inverse();

  for (auto &label : labels->getLabels())
  {
    auto anchor2D = frameData.project(label.anchorPosition);
    // calc pixel coords
    float x = (anchor2D.x() * 0.5f + 0.5f) * width;
    float y = (anchor2D.y() * 0.5f + 0.5f) * height;

    auto newPosition = costFunctionCalculator.calculateForLabel(label.id, x, y);

    float newX = (std::get<0>(newPosition) / width - 0.5f) * 2.0f;
    float newY = (std::get<1>(newPosition) / height - 0.5f) * 2.0f;
    Eigen::Vector4f reprojected =
        inverseViewProjection * Eigen::Vector4f(newX, newY, anchor2D.z(), 1);
    reprojected /= reprojected.w();

    result[label.id] = toVector3f(reprojected);
  }

  return result;
}

void Labeller::resize(int width, int height)
{
  this->width = width;
  this->height = height;

  costFunctionCalculator.resize(width, height);
}

}  // namespace Placement
