#if _WIN32
#pragma warning (disable: 4267 4996)
#endif
#include "./labeller.h"
#include <Eigen/Geometry>
#include <QLoggingCategory>
#include <vector>
#include <map>
#include <chrono>
#include "../utils/cuda_array_provider.h"
#include "../utils/logging.h"
#include "./summed_area_table.h"
#if _WIN32
#else
#include "./apollonius.h"
#endif
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
    std::shared_ptr<CudaArrayProvider> integralCostsTextureMapper,
    std::shared_ptr<CudaArrayProvider> distanceTransformTextureMapper,
    std::shared_ptr<CudaArrayProvider> apolloniusTextureMapper,
    std::shared_ptr<CudaArrayProvider> constraintTextureMapper,
    std::shared_ptr<PersistentConstraintUpdater> constraintUpdater)
{
  qCInfo(plChan) << "Initialize";
  if (!integralCosts.get())
    integralCosts =
        std::make_shared<SummedAreaTable>(integralCostsTextureMapper);

  this->distanceTransformTextureMapper = distanceTransformTextureMapper;
  this->apolloniusTextureMapper = apolloniusTextureMapper;
  this->constraintUpdater = constraintUpdater;

  costFunctionCalculator =
      std::make_unique<CostFunctionCalculator>(constraintTextureMapper);
  costFunctionCalculator->resize(size.x(), size.y());
  costFunctionCalculator->setTextureSize(
      integralCostsTextureMapper->getWidth(),
      integralCostsTextureMapper->getHeight());
}

void Labeller::cleanup()
{
  integralCosts.reset();
  apolloniusTextureMapper.reset();
  costFunctionCalculator.reset();
  distanceTransformTextureMapper.reset();
  apolloniusTextureMapper.reset();
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

  Eigen::Matrix4f inverseViewProjection = frameData.viewProjection.inverse();

  auto startTime = std::chrono::high_resolution_clock::now();
  integralCosts->runKernel();
  qCDebug(plChan) << "SAT took" << calculateDurationSince(startTime) << "ms";

  auto labelsInLayer = useApollonius
                           ? getLabelsInApolloniusOrder(frameData, bufferSize)
                           : labels->getLabels();

  for (auto &label : labelsInLayer)
  {
    auto anchor2D = frameData.project(label.anchorPosition);

    Eigen::Vector2i labelSizeForBuffer =
        label.size.cast<int>().cwiseProduct(bufferSize).cwiseQuotient(size);

    Eigen::Vector2f anchorPixels = toPixel(anchor2D, size);
    Eigen::Vector2i anchorForBuffer = toPixel(anchor2D, bufferSize).cast<int>();

    constraintUpdater->updateConstraints(label.id, anchorForBuffer,
                                         labelSizeForBuffer);

    auto newPos = costFunctionCalculator->calculateForLabel(
        integralCosts->getResults(), label.id, anchorPixels.x(),
        anchorPixels.y(), label.size.x(), label.size.y());

    Eigen::Vector2i newPosition(std::get<0>(newPos), std::get<1>(newPos));
    constraintUpdater->setPosition(label.id, newPosition);

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

void Labeller::setCostFunctionWeights(CostFunctionWeights weights)
{
  costFunctionCalculator->weights = weights;
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

std::vector<Label>
Labeller::getLabelsInApolloniusOrder(const LabellerFrameData &frameData,
                                     Eigen::Vector2i bufferSize)
{
#if _WIN32
  return std::vector<Label>(labels->getLabels());
#else
  if (labels->count() == 1)
    return std::vector<Label>{ labels->getLabels()[0] };

  std::vector<Eigen::Vector4f> labelsSeed =
      createLabelSeeds(bufferSize, frameData.viewProjection);

  Apollonius apollonius(distanceTransformTextureMapper, apolloniusTextureMapper,
                        labelsSeed, labels->count());
  apollonius.run();

  std::vector<Label> result;
  for (int id : apollonius.calculateOrdering())
  {
    result.push_back(labels->getById(id));
  }
  return result;
#endif
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
