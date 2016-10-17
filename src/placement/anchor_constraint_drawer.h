#ifndef SRC_PLACEMENT_ANCHOR_CONSTRAINT_DRAWER_H_

#define SRC_PLACEMENT_ANCHOR_CONSTRAINT_DRAWER_H_

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
class AnchorConstraintDrawer
{
 public:
  AnchorConstraintDrawer(int width, int height);
  virtual ~AnchorConstraintDrawer();

  void initialize(Graphics::Gl *gl,
                  std::shared_ptr<Graphics::ShaderManager> shaderManager);

  virtual void update(const std::vector<float> &anchors);
  virtual void draw(float color, Eigen::Vector2f halfSize);
  virtual void clear();

  void saveBufferTo(std::string filename);

 private:
  int width;
  int height;
  Graphics::Gl *gl;
  std::unique_ptr<Graphics::VertexArray> vertexArray;
  std::unique_ptr<ConstraintDrawer> constraintDrawer;
  RenderData renderData;
};

#endif  // SRC_PLACEMENT_ANCHOR_CONSTRAINT_DRAWER_H_
