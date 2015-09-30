#ifndef SRC_GRAPHICS_VOLUME_H_

#define SRC_GRAPHICS_VOLUME_H_

#include <Eigen/Core>
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
  virtual VolumeData getVolumeData() = 0;
  virtual float* getData() = 0;
  virtual Eigen::Vector3i getDataSize() = 0;
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_VOLUME_H_
