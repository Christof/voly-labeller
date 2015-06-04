#ifndef SRC_FRUSTUM_OPTIMIZER_H_

#define SRC_FRUSTUM_OPTIMIZER_H_

#include <memory>
#include "./math/eigen.h"

class Nodes;

/**
 * \brief Calculates near and far plane distances from the bounding boxes of all
 * nodes
 *
 */
class FrustumOptimizer
{
 public:
  explicit FrustumOptimizer(std::shared_ptr<Nodes> nodes);

  void update(Eigen::Matrix4f viewMatrix);
  float getNear();
  float getFar();

 private:
  std::shared_ptr<Nodes> nodes;
  float near;
  float far;
};

#endif  // SRC_FRUSTUM_OPTIMIZER_H_
