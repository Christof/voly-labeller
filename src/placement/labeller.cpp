#include "./labeller.h"
#include <Eigen/Geometry>
#include <QLoggingCategory>
#include <vector>
#include <map>
#include "../utils/cuda_array_provider.h"
#include "./summed_area_table.h"
#include "./apollonius.h"
#include "./occupancy_updater.h"
#include "./constraint_updater.h"
#include "../utils/memory.h"

namespace Placement
{

QLoggingCategory plChan("Placement.Labeller");

Labeller::Labeller(std::shared_ptr<Labels> labels) : labels(labels)
{
}

void Labeller::initialize(
    std::shared_ptr<CudaArrayProvider> occupancyTextureMapper,
    std::shared_ptr<CudaArrayProvider> distanceTransformTextureMapper,
    std::shared_ptr<CudaArrayProvider> apolloniusTextureMapper,
    std::shared_ptr<CudaArrayProvider> constraintTextureMapper,
    std::shared_ptr<ConstraintUpdater> constraintUpdater)
{
  qCInfo(plChan) << "Initialize";
  if (!occupancySummedAreaTable.get())
    occupancySummedAreaTable =
        std::make_shared<SummedAreaTable>(occupancyTextureMapper);
  occupancyUpdater = std::make_shared<OccupancyUpdater>(occupancyTextureMapper);

  this->distanceTransformTextureMapper = distanceTransformTextureMapper;
  this->apolloniusTextureMapper = apolloniusTextureMapper;
  this->constraintUpdater = constraintUpdater;

  costFunctionCalculator =
      std::make_unique<CostFunctionCalculator>(constraintTextureMapper);
  costFunctionCalculator->resize(width, height);
  costFunctionCalculator->setTextureSize(occupancyTextureMapper->getWidth(),
                                         occupancyTextureMapper->getHeight());
}

void Labeller::cleanup()
{
  occupancySummedAreaTable.reset();
  apolloniusTextureMapper.reset();
  occupancyUpdater.reset();
  costFunctionCalculator.reset();
}

void Labeller::setInsertionOrder(std::vector<int> ids)
{
  insertionOrder = ids;
}

std::map<int, Eigen::Vector3f>
Labeller::update(const LabellerFrameData &frameData)
{
  newPositions.clear();
  if (!occupancySummedAreaTable.get())
    return newPositions;

  Eigen::Vector2i bufferSize(distanceTransformTextureMapper->getWidth(),
                             distanceTransformTextureMapper->getHeight());
  std::vector<Eigen::Vector4f> labelsSeed =
      createLabelSeeds(bufferSize, frameData.viewProjection);

  Apollonius apollonius(distanceTransformTextureMapper, apolloniusTextureMapper,
                        labelsSeed, labels->count());
  apollonius.run();
  setInsertionOrder(apollonius.calculateOrdering());

  Eigen::Matrix4f inverseViewProjection = frameData.viewProjection.inverse();

  std::map<int, Eigen::Vector2i> labelSizesForBuffer;
  std::map<int, Eigen::Vector2i> anchors2DForBuffer;
  std::map<int, Eigen::Vector2i> labelPositionsForBuffer;

  occupancySummedAreaTable->runKernel();

  for (size_t i = 0; i < insertionOrder.size(); ++i)
  {
    int id = insertionOrder[i];

    auto label = labels->getById(id);
    auto anchor2D = frameData.project(label.anchorPosition);
    float x = (anchor2D.x() * 0.5f + 0.5f) * width;
    float y = (anchor2D.y() * 0.5f + 0.5f) * height;

    std::cout << "x " << int(x) << " y " << int(y) << std::endl;
    Eigen::Vector2i labelSizeForBuffer =
        label.size.cast<int>().cwiseProduct(bufferSize).cwiseQuotient(
            Eigen::Vector2i(width, height));
    labelSizesForBuffer[id] = labelSizeForBuffer;

    Eigen::Vector2i anchor2DForBuffer =
        toPixel(anchor2D, bufferSize).cast<int>();
    anchors2DForBuffer[id] = anchor2DForBuffer;

    constraintUpdater->clear();
    for (size_t insertedLabelIndex = 0; insertedLabelIndex < i;
         ++insertedLabelIndex)
    {
      int oldId = insertionOrder[insertedLabelIndex];
      constraintUpdater->addLabel(
          anchor2DForBuffer, labelSizeForBuffer, anchors2DForBuffer[oldId],
          labelPositionsForBuffer[oldId], labelSizesForBuffer[oldId]);
    }

    auto newPosition = costFunctionCalculator->calculateForLabel(
        occupancySummedAreaTable->getResults(), label.id, x, y, label.size.x(),
        label.size.y());

    float newXPosition = std::get<0>(newPosition);
    float newYPosition = std::get<1>(newPosition);

    labelPositionsForBuffer[id] = Eigen::Vector2i(newXPosition, newYPosition);

    // occupancyUpdater->addLabel(newXPosition, newYPosition,
    //                            labelSizeForBuffer.x(),
    //                            labelSizeForBuffer.y());

    float newXNDC = (newXPosition / bufferSize.x() - 0.5f) * 2.0f;
    float newYNDC = (newYPosition / bufferSize.y() - 0.5f) * 2.0f;
    Eigen::Vector4f reprojected =
        inverseViewProjection *
        Eigen::Vector4f(newXNDC, newYNDC, anchor2D.z(), 1);
    reprojected /= reprojected.w();

    newPositions[label.id] = toVector3f(reprojected);
  }

  return newPositions;
}

void Labeller::resize(int width, int height)
{
  this->width = width;
  this->height = height;

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

Eigen::Vector2f Labeller::toPixel(Eigen::Vector3f ndc, Eigen::Vector2i size)
{
  const Eigen::Vector2f half(0.5f, 0.5f);

  auto zeroToOne = ndc.head<2>().cwiseProduct(half) + half;

  return zeroToOne.cwiseProduct(size.cast<float>());
}

}  // namespace Placement
