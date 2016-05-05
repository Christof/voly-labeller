#include "./frustum_optimizer.h"
#include <limits>
#include <algorithm>
#include "./nodes.h"

FrustumOptimizer::FrustumOptimizer(std::shared_ptr<Nodes> nodes) : nodes(nodes)
{
}

void FrustumOptimizer::update(Eigen::Matrix4f viewMatrix)
{
  float fmin = std::numeric_limits<float>::max();
  float fmax = -fmin;
  for (auto node : nodes->getNodes())
  {
    auto obb = node->getObb();
    if (!obb.isInitialized())
      continue;

    for (int i = 0; i < 8; ++i)
    {
      auto point = obb.corners[i];

      Eigen::Vector4f transformed = mul(viewMatrix, point);

      if (transformed.z() < fmin)
        fmin = transformed.z();

      if (transformed.z() > fmax)
        fmax = transformed.z();
    }
  }

  nearPlane = std::max(0.001f, -fmax - 0.1f);
  farPlane = -fmin + 0.1f;
}

float FrustumOptimizer::getNear()
{
  return nearPlane;
}

float FrustumOptimizer::getFar()
{
  return farPlane;
}

