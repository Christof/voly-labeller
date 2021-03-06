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
 * \brief Draws shadow constraints using a Graphics::VertexArray and a
 * ConstraintDrawer
 *
 * Before using an instance, it must be initialized by calling #initialize.
 *
 * Certain methods are marked as virtual, which is just used for testing
 * purposes.
 *
 *
 */
class ShadowConstraintDrawer
{
 public:
  ShadowConstraintDrawer(int width, int height);
  virtual ~ShadowConstraintDrawer();

  void initialize(Graphics::Gl *gl,
                  std::shared_ptr<Graphics::ShaderManager> shaderManager);

  virtual void update(const std::vector<float> &sources,
                      const std::vector<float> &starts,
                      const std::vector<float> &ends);
  virtual void draw(float color, Eigen::Vector2f halfSize);
  virtual void clear();

 private:
  int width;
  int height;
  Graphics::Gl *gl;
  std::unique_ptr<Graphics::VertexArray> vertexArray;
  std::unique_ptr<ConstraintDrawer> constraintDrawer;
  RenderData renderData;
};

#endif  // SRC_PLACEMENT_SHADOW_CONSTRAINT_DRAWER_H_
