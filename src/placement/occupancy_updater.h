#ifndef SRC_PLACEMENT_OCCUPANCY_UPDATER_H_

#define SRC_PLACEMENT_OCCUPANCY_UPDATER_H_

#include <cuda_runtime.h>
#include <memory>
#include "../utils/cuda_array_provider.h"

/**
 * \brief
 *
 *
 */
class OccupancyUpdater
{
 public:
  OccupancyUpdater(std::shared_ptr<CudaArrayProvider> occupancy);
  virtual ~OccupancyUpdater();

  void addLabel(int x, int y, int width, int height);

 private:
  std::shared_ptr<CudaArrayProvider> occupancy;
  cudaSurfaceObject_t surface = 0;

  void createSurface();
};

#endif  // SRC_PLACEMENT_OCCUPANCY_UPDATER_H_
