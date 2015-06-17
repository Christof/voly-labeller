#include "./frustum_optimizer.h"
#include <limits>
#include <algorithm>
#include "./nodes.h"

FrustumOptimizer::FrustumOptimizer(std::shared_ptr<Nodes> nodes) : nodes(nodes)
{
}

void FrustumOptimizer::update(Eigen::Matrix4f viewMatrix)
{
  float min = std::numeric_limits<float>::max();
  float max = -min;
  for (auto node : nodes->getNodes())
  {
    auto obb = node->getObb();
    if (!obb.get())
      continue;

    for (int i = 0; i < 8; ++i)
    {
      auto point = obb->corners[i];

      Eigen::Vector4f transformed = mul(viewMatrix, point);

      if (transformed.z() < min)
        min = transformed.z();

      if (transformed.z() > max)
        max = transformed.z();
    }
  }

  near = std::max(0.1f, -max - 0.1f);
  far = -min + 0.1f;
}

float FrustumOptimizer::getNear()
{
  return near;
}

float FrustumOptimizer::getFar()
{
  return far;
}

