#ifndef SRC_GRAPHICS_VOLUME_H_

#define SRC_GRAPHICS_VOLUME_H_

#include <Eigen/Core>
#include <functional>
#include "./volume_data.h"

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
    if (removeFromVolumes)
      removeFromVolumes();
  }
  virtual VolumeData getVolumeData() = 0;
  virtual float *getData() = 0;
  virtual Eigen::Vector3i getDataSize() = 0;

  void setRemoveFromVolumesFunction(std::function<void()> removeFromVolumes)
  {
    this->removeFromVolumes = removeFromVolumes;
  }

 protected:
  std::function<void()> removeFromVolumes;
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_VOLUME_H_
