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
 * \brief Helper class to draw constraints provided by a Graphics::VertexArray
 *
 * The given vertexShader and geometryShader is combined with
 * `colorImmediate.frag` into a Graphics::ShaderProgram. Before the
 * Graphics::VertexArray is drawn, a logical or operation for drawing is
 * configured and afterwards reset to the initial value.
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
