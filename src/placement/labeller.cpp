#include "./labeller.h"
#include <Eigen/Geometry>
#include <QLoggingCategory>
#include <vector>
#include <map>
#include <chrono>
#include "../utils/cuda_array_provider.h"
#include "../utils/logging.h"
#include "./summed_area_table.h"
#include "./apollonius.h"
#include "./occupancy_updater.h"
#include "./constraint_updater.h"
#include "./persistent_constraint_updater.h"
#include "../utils/memory.h"

namespace Placement
{

QLoggingCategory plChan("Placement.Labeller");

Labeller::Labeller(std::shared_ptr<LabelsContainer> labels) : labels(labels)
{
}

void Labeller::initialize(
    std::shared_ptr<CudaArrayProvider> occupancyTextureMapper,
    std::shared_ptr<CudaArrayProvider> distanceTransformTextureMapper,
    std::shared_ptr<CudaArrayProvider> apolloniusTextureMapper,
    std::shared_ptr<CudaArrayProvider> constraintTextureMapper,
    std::shared_ptr<PersistentConstraintUpdater> constraintUpdater)
{
  qCInfo(plChan) << "Initialize";
  if (!integralCosts.get())
    integralCosts =
        std::make_shared<SummedAreaTable>(occupancyTextureMapper);
  occupancyUpdater = std::make_shared<OccupancyUpdater>(occupancyTextureMapper);

  this->distanceTransformTextureMapper = distanceTransformTextureMapper;
  this->apolloniusTextureMapper = apolloniusTextureMapper;
  this->constraintUpdater = constraintUpdater;

  costFunctionCalculator =
      std::make_unique<CostFunctionCalculator>(constraintTextureMapper);
  costFunctionCalculator->resize(size.x(), size.y());
  costFunctionCalculator->setTextureSize(occupancyTextureMapper->getWidth(),
                                         occupancyTextureMapper->getHeight());
}

void Labeller::cleanup()
{
  integralCosts.reset();
  apolloniusTextureMapper.reset();
  occupancyUpdater.reset();
  costFunctionCalculator.reset();
}

std::map<int, Eigen::Vector3f>
Labeller::update(const LabellerFrameData &frameData)
{
  if (labels->count() == 0)
    return std::map<int, Eigen::Vector3f>();

  newPositions.clear();
  if (!integralCosts.get())
    return newPositions;

  Eigen::Vector2i bufferSize(distanceTransformTextureMapper->getWidth(),
                             distanceTransformTextureMapper->getHeight());
  auto insertionOrder = calculateInsertionOrder(frameData, bufferSize);

  Eigen::Matrix4f inverseViewProjection = frameData.viewProjection.inverse();

  auto startTime = std::chrono::high_resolution_clock::now();
  integralCosts->runKernel();
  qCDebug(plChan) << "SAT took" << calculateDurationSince(startTime) << "ms";

  for (size_t i = 0; i < insertionOrder.size(); ++i)
  {
    int id = insertionOrder[i];

    auto label = labels->getById(id);
    auto anchor2D = frameData.project(label.anchorPosition);

    Eigen::Vector2i labelSizeForBuffer =
        label.size.cast<int>().cwiseProduct(bufferSize).cwiseQuotient(size);

    Eigen::Vector2f anchorPixels = toPixel(anchor2D, size);
    Eigen::Vector2i anchorForBuffer = toPixel(anchor2D, bufferSize).cast<int>();

    constraintUpdater->updateConstraints(id, anchorForBuffer,
                                         labelSizeForBuffer);

    auto newPos = costFunctionCalculator->calculateForLabel(
        integralCosts->getResults(), label.id, anchorPixels.x(),
        anchorPixels.y(), label.size.x(), label.size.y());

    Eigen::Vector2i newPosition(std::get<0>(newPos), std::get<1>(newPos));
    constraintUpdater->setPosition(id, newPosition);

    // occupancyUpdater->addLabel(newXPosition, newYPosition,
    //                            labelSizeForBuffer.x(),
    //                            labelSizeForBuffer.y());

    newPositions[label.id] = reprojectTo3d(newPosition, anchor2D.z(),
                                           bufferSize, inverseViewProjection);
  }

  return newPositions;
}

void Labeller::resize(int width, int height)
{
  size = Eigen::Vector2i(width, height);

  if (costFunctionCalculator.get())
    costFunctionCalculator->resize(width, height);
}

std::map<int, Eigen::Vector3f> Labeller::getLastPlacementResult()
{
  return newPositions;
}

std::vector<Eigen::Vector4f>
Labeller::createLabelSeeds(Eigen::Vector2i size, Eigen::Matrix4f viewProjection)
{
  std::vector<Eigen::Vector4f> result;
  for (auto &label : labels->getLabels())
  {
    Eigen::Vector4f pos =
        viewProjection * Eigen::Vector4f(label.anchorPosition.x(),
                                         label.anchorPosition.y(),
                                         label.anchorPosition.z(), 1);
    float x = (pos.x() / pos.w() * 0.5f + 0.5f) * size.x();
    float y = (pos.y() / pos.w() * 0.5f + 0.5f) * size.y();
    result.push_back(Eigen::Vector4f(label.id, x, y, 1));
  }

  return result;
}

std::vector<int>
Labeller::calculateInsertionOrder(const LabellerFrameData &frameData,
                                  Eigen::Vector2i bufferSize)
{
  if (labels->count() == 1)
    return std::vector<int>{ labels->getLabels()[0].id };

  std::vector<Eigen::Vector4f> labelsSeed =
      createLabelSeeds(bufferSize, frameData.viewProjection);

  Apollonius apollonius(distanceTransformTextureMapper, apolloniusTextureMapper,
                        labelsSeed, labels->count());
  apollonius.run();
  return apollonius.calculateOrdering();
}

Eigen::Vector3f Labeller::reprojectTo3d(Eigen::Vector2i newPosition,
                                        float anchorZValue,
                                        Eigen::Vector2i bufferSize,
                                        Eigen::Matrix4f inverseViewProjection)
{
  Eigen::Vector2f newNDC2d =
      2.0f * newPosition.cast<float>().cwiseQuotient(bufferSize.cast<float>()) -
      Eigen::Vector2f(1.0f, 1.0f);

  Eigen::Vector4f newNDC(newNDC2d.x(), newNDC2d.y(), anchorZValue, 1);
  Eigen::Vector4f reprojected = inverseViewProjection * newNDC;
  reprojected /= reprojected.w();

  return toVector3f(reprojected);
}

Eigen::Vector2f Labeller::toPixel(Eigen::Vector3f ndc, Eigen::Vector2i size)
{
  const Eigen::Vector2f half(0.5f, 0.5f);

  auto zeroToOne = ndc.head<2>().cwiseProduct(half) + half;

  return zeroToOne.cwiseProduct(size.cast<float>());
}

}  // namespace Placement
