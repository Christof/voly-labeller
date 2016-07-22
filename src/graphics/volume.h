#ifndef SRC_GRAPHICS_VOLUME_H_

#define SRC_GRAPHICS_VOLUME_H_

#include <Eigen/Core>
#include <memory>
#include "./volume_data.h"
#include "./volume_manager.h"

namespace Graphics
{
/**
 * \brief Represents an interface for volumes to get data
 * and met information
 *
 */
class Volume
{
 public:
  ~Volume()
  {
    if (volumeId > 0 && volumeManager)
      volumeManager->removeVolume(volumeId);
  };

  void initialize(int volumeId, Graphics::VolumeManager *volumeManager)
  {
    this->volumeId = volumeId;
    this->volumeManager = volumeManager;
  };

  virtual VolumeData getVolumeData(const RenderData &renderData) = 0;
  virtual float *getData() = 0;
  virtual Eigen::Vector3i getDataSize() = 0;

 protected:
  int volumeId = -1;
  Graphics::VolumeManager *volumeManager = nullptr;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_VOLUME_H_
