#ifndef SRC_GRAPHICS_VOLUME_DATA_H_

#define SRC_GRAPHICS_VOLUME_DATA_H_

#include <Eigen/Core>
#include "./texture_address.h"

namespace Graphics
{

/**
 * \brief Data for one volume necessary for volume rendering
 */
struct VolumeData
{
  TextureAddress textureAddress;
  Eigen::Matrix4f textureMatrix;
  Eigen::Matrix4f gradientMatrix;
  Eigen::Matrix4f objectToDatasetMatrix;
  int volumeId;
  int transferFunctionRow;
  // int transferFunctionRowCount;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_VOLUME_DATA_H_
