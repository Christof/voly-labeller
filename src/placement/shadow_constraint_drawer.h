#ifndef SRC_PLACEMENT_SHADOW_CONSTRAINT_DRAWER_H_

#define SRC_PLACEMENT_SHADOW_CONSTRAINT_DRAWER_H_

#include <memory>
#include <vector>
#include "../graphics/render_data.h"

class ConstraintDrawer;

namespace Graphics
{
class VertexArray;
class Gl;
class ShaderManager;
}

/**
 * \brief
 *
 *
 */
class ShadowConstraintDrawer
{
 public:
  ShadowConstraintDrawer(
      int width, int height, Graphics::Gl *gl,
      std::shared_ptr<Graphics::ShaderManager> shaderManager);
  ~ShadowConstraintDrawer();

  void update(const std::vector<float> &sources,
              const std::vector<float> &starts, const std::vector<float> &ends);
  void draw(float color, Eigen::Vector2f halfSize);
  void clear();

 private:
  int width;
  int height;
  Graphics::Gl *gl;
  std::unique_ptr<Graphics::VertexArray> vertexArray;
  std::unique_ptr<ConstraintDrawer> constraintDrawer;
  RenderData renderData;
};

#endif  // SRC_PLACEMENT_SHADOW_CONSTRAINT_DRAWER_H_
