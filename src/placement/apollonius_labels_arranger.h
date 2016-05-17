#ifndef SRC_PLACEMENT_APOLLONIUS_LABELS_ARRANGER_H_

#define SRC_PLACEMENT_APOLLONIUS_LABELS_ARRANGER_H_

#include <Eigen/Core>
#include <memory>
#include <vector>
#include "./labels_arranger.h"

class CudaArrayProvider;

namespace Placement
{

/**
 * \brief Returns the labels in an order defined by the apollonius graph
 *
 */
class ApolloniusLabelsArranger : public LabelsArranger
{
 public:
  ApolloniusLabelsArranger() = default;

  void
  initialize(std::shared_ptr<CudaArrayProvider> distanceTransformTextureMapper,
             std::shared_ptr<CudaArrayProvider> apolloniusTextureMapper);

  virtual std::vector<Label>
  getArrangement(const LabellerFrameData &frameData,
                 std::shared_ptr<LabelsContainer> labels);

 private:
  std::shared_ptr<CudaArrayProvider> distanceTransformTextureMapper;
  std::shared_ptr<CudaArrayProvider> apolloniusTextureMapper;

  std::vector<Eigen::Vector4f>
  createLabelSeeds(Eigen::Vector2i size, Eigen::Matrix4f viewProjection,
                   std::shared_ptr<LabelsContainer> labels);
};

}  // namespace Placement

#endif  // SRC_PLACEMENT_APOLLONIUS_LABELS_ARRANGER_H_
