#ifndef SRC_GRAPHICS_VOLUME_H_

#define SRC_GRAPHICS_VOLUME_H_

#include "./volume_data.h"

namespace Graphics
{
/**
 * \brief
 *
 *
 */
class Volume
{
 public:
  virtual VolumeData getVolumeData() = 0;
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_VOLUME_H_
