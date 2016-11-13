#if _WIN32
#pragma warning(disable : 4267 4996)
#endif
#include "./labeller.h"
#include <Eigen/Geometry>
#include <QLoggingCategory>
#include <vector>
#include <map>
#include <chrono>
#include "../math/eigen.h"
#include "../utils/cuda_array_provider.h"
#include "../utils/logging.h"
#include "./summed_area_table.h"
#include "./constraint_updater.h"
#include "./persistent_constraint_updater.h"
#include "./labels_arranger.h"
#include "../utils/memory.h"

namespace Placement
{

QLoggingCategory plChan("Placement.Labeller");

Labeller::Labeller(std::shared_ptr<LabelsContainer> labels) : labels(labels)
{
}

void Labeller::initialize(
    std::shared_ptr<CudaArrayProvider> integralCostsTextureMapper,
    std::shared_ptr<CudaArrayProvider> constraintTextureMapper,
    std::shared_ptr<PersistentConstraintUpdater> constraintUpdater)
{
  qCInfo(plChan) << "Initialize";
  if (!integralCosts.get())
    integralCosts =
        std::make_shared<SummedAreaTable>(integralCostsTextureMapper);

  this->constraintUpdater = constraintUpdater;

  bufferSize = Eigen::Vector2i(integralCostsTextureMapper->getWidth(),
                               integralCostsTextureMapper->getHeight());

  costFunctionCalculator =
      std::make_unique<CostFunctionCalculator>(constraintTextureMapper);
  costFunctionCalculator->resize(size.x(), size.y());
  costFunctionCalculator->setTextureSize(bufferSize.x(), bufferSize.y());
}

void Labeller::cleanup()
{
  integralCosts.reset();
  costFunctionCalculator.reset();
}

std::map<int, Eigen::Vector2f>
Labeller::update(const LabellerFrameData &frameData, bool ignoreOldPosition,
                 const LabelPositions &oldLabelPositions)
{
  if (labels->count() == 0)
    return std::map<int, Eigen::Vector2f>();

  oldPositions.clear();
  for (auto &pair : newPositions)
    oldPositions[pair.first] =
        frameData.project2d(oldLabelPositions.get3dFor(pair.first));

  newPositions.clear();
  if (!integralCosts.get())
    return newPositions;

  auto startTime = std::chrono::high_resolution_clock::now();
  integralCosts->runKernel();
  qCDebug(plChan) << "SAT took" << calculateDurationSince(startTime) << "ms";

  costSum = 0.0f;
  auto labelsInLayer = labelsArranger->getArrangement(frameData, labels);

  for (auto &label : labelsInLayer)
  {
    auto anchor2D = frameData.project(label.anchorPosition).head<2>();

    Eigen::Vector2i labelSizeForBuffer =
        label.size.cast<int>().cwiseProduct(bufferSize).cwiseQuotient(size);

    Eigen::Vector2f anchorPixels = toPixel(anchor2D, size);
    Eigen::Vector2i anchorForBuffer = toPixel(anchor2D, bufferSize).cast<int>();

    constraintUpdater->updateConstraints(label.id, anchorForBuffer,
                                         labelSizeForBuffer);

    bool ignoreOldLabel = ignoreOldPosition || !oldPositions.count(label.id);
    Eigen::Vector2f oldPositionPixel =
        ignoreOldLabel ? Eigen::Vector2f(0, 0)
                          : toPixel(oldPositions.at(label.id), size);

    auto result = costFunctionCalculator->calculateForLabel(
        integralCosts->getResults(), label.id, anchorPixels.x(),
        anchorPixels.y(), label.size.x(), label.size.y(), ignoreOldLabel,
        oldPositionPixel.x(), oldPositionPixel.y());

    constraintUpdater->setPosition(label.id, result.position);
    costSum += result.cost;

    qCDebug(plChan) << label.id << "\t" << result.position.x() << "|"
                    << result.position.y() << result.cost;

    auto relativeResult =
        result.position.cast<float>().cwiseQuotient(bufferSize.cast<float>());
    Eigen::Vector2f newNDC2d =
        2.0f * relativeResult - Eigen::Vector2f(1.0f, 1.0f);
    newPositions[label.id] = newNDC2d;
  }

  oldViewProjectionMatrix = frameData.viewProjection;

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

void Labeller::setLabelsArranger(std::shared_ptr<LabelsArranger> labelsArranger)
{
  this->labelsArranger = labelsArranger;
}

std::map<int, Eigen::Vector2f> Labeller::getLastPlacementResult()
{
  return newPositions;
}

float Labeller::getLastSumOfCosts()
{
  return costSum;
}

}  // namespace Placement
