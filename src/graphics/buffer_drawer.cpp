#include "./buffer_drawer.h"
#include <Eigen/Geometry>
#include <vector>
#include "./gl.h"
#include "./shader_manager.h"
#include "./vertex_array.h"
#include "./render_data.h"
#include "./shader_program.h"

namespace Graphics
{

BufferDrawer::BufferDrawer(int width, int height, Gl *gl,
                           std::shared_ptr<ShaderManager> shaderManager)
  : width(width), height(height), gl(gl), shaderManager(shaderManager)
{
  shaderId = shaderManager->addShader(":/shader/constraint.vert",
                                      ":/shader/colorImmediate.frag");

  Eigen::Affine3f pixelToNDCTransform(
      Eigen::Translation3f(Eigen::Vector3f(-1, 1, 0)) *
      Eigen::Scaling(Eigen::Vector3f(2.0f / width, -2.0f / height, 1)));
  pixelToNDC = pixelToNDCTransform.matrix();
}

void BufferDrawer::drawElementVector(std::vector<float> positions, float weight)
{
  Graphics::VertexArray *vertexArray =
      new Graphics::VertexArray(gl, GL_TRIANGLE_FAN, 2);
  vertexArray->addStream(positions, 2);

  RenderData renderData;
  renderData.viewMatrix = pixelToNDC;
  renderData.viewProjectionMatrix = pixelToNDC;
  shaderManager->getShader(shaderId)->setUniform("weight", weight);
  shaderManager->bind(shaderId, renderData);
  vertexArray->draw();

  delete vertexArray;
}
void BufferDrawer::clear()
{
  gl->glViewport(0, 0, width, height);
  gl->glClearColor(0, 0, 0, 0);
  gl->glClear(GL_COLOR_BUFFER_BIT);
}

}  // namespace Graphics
