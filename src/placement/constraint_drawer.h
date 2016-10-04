#ifndef SRC_PLACEMENT_CONSTRAINT_DRAWER_H_

#define SRC_PLACEMENT_CONSTRAINT_DRAWER_H_

#include <Eigen/Core>
#include <memory>
#include <string>

class RenderData;

namespace Graphics
{
class Gl;
class ShaderManager;
class VertexArray;
}

/**
 * \brief
 *
 *
 */
class ConstraintDrawer
{
 public:
  ConstraintDrawer(Graphics::Gl *gl,
                   std::shared_ptr<Graphics::ShaderManager> shaderManager,
                   std::string vertexShader, std::string geometryShader);

  void draw(const Graphics::VertexArray *vertexArray,
            const RenderData &renderData, float color,
            Eigen::Vector2f halfSize);

 private:
  Graphics::Gl *gl;
  std::shared_ptr<Graphics::ShaderManager> shaderManager;
  int shaderId;
};

#endif  // SRC_PLACEMENT_CONSTRAINT_DRAWER_H_
