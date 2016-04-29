#ifndef SRC_PLACEMENT_APOLLONIUS_LABELS_ARRANGER_H_

#define SRC_PLACEMENT_APOLLONIUS_LABELS_ARRANGER_H_

#include <Eigen/Core>
#include <memory>
#include <vector>
#include "../labelling/labels_container.h"
#include "../labelling/labeller_frame_data.h"

class CudaArrayProvider;

namespace Placement
{

/**
 * \brief
 *
 *
 */
class ApolloniusLabelsArranger
{
 public:
  ApolloniusLabelsArranger() = default;
  virtual ~ApolloniusLabelsArranger();

  void
  initialize(std::shared_ptr<CudaArrayProvider> distanceTransformTextureMapper,
             std::shared_ptr<CudaArrayProvider> apolloniusTextureMapper);

  std::vector<Label> calculateOrder(const LabellerFrameData &frameData,
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
