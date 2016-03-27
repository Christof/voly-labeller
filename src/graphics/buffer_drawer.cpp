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

void BufferDrawer::drawElementVector(std::vector<float> positions, char bitIndex)
{
  Graphics::VertexArray *vertexArray =
      new Graphics::VertexArray(gl, GL_TRIANGLE_FAN, 2);
  vertexArray->addStream(positions, 2);

  GLboolean isBlendingEnabled = gl->glIsEnabled(GL_BLEND);
  gl->glEnable(GL_BLEND);
  GLint logicOperationMode;
  gl->glGetIntegerv(GL_LOGIC_OP_MODE, &logicOperationMode);
  gl->glEnable(GL_COLOR_LOGIC_OP);
  gl->glLogicOp(GL_OR);

  RenderData renderData;
  renderData.viewMatrix = pixelToNDC;
  renderData.viewProjectionMatrix = pixelToNDC;
  float color = (1 << (7 - bitIndex)) / 255.0f;
  shaderManager->getShader(shaderId)->setUniform("color", color);
  shaderManager->bind(shaderId, renderData);
  vertexArray->draw();

  if (isBlendingEnabled)
    gl->glEnable(GL_BLEND);

  gl->glLogicOp(logicOperationMode);
  gl->glDisable(GL_COLOR_LOGIC_OP);

  delete vertexArray;
}
void BufferDrawer::clear()
{
  gl->glViewport(0, 0, width, height);
  gl->glClearColor(0, 0, 0, 0);
  gl->glClear(GL_COLOR_BUFFER_BIT);
}

}  // namespace Graphics
