#include "./constraint_drawer.h"
#include "../graphics/gl.h"
#include "../graphics/shader_manager.h"
#include "../graphics/shader_program.h"
#include "../graphics/render_data.h"
#include "../graphics/vertex_array.h"

ConstraintDrawer::ConstraintDrawer(
    Graphics::Gl *gl, std::shared_ptr<Graphics::ShaderManager> shaderManager,
    std::string vertexShader, std::string geometryShader)
  : gl(gl), shaderManager(shaderManager)
{
  shaderId = shaderManager->addShader(vertexShader, geometryShader,
                                      ":/shader/colorImmediate.frag");
}

void ConstraintDrawer::draw(const Graphics::VertexArray *vertexArray,
                            const RenderData &renderData, float color,
                            Eigen::Vector2f halfSize)
{
  GLboolean isBlendingEnabled = gl->glIsEnabled(GL_BLEND);
  gl->glEnable(GL_BLEND);
  GLint logicOperationMode;
  gl->glGetIntegerv(GL_LOGIC_OP_MODE, &logicOperationMode);
  gl->glEnable(GL_COLOR_LOGIC_OP);
  gl->glLogicOp(GL_OR);

  auto shader = shaderManager->getShader(shaderId);
  shader->setUniform("color", color);
  shader->setUniform("halfSize", halfSize);
  shaderManager->bind(shaderId, renderData);

  vertexArray->draw();

  if (isBlendingEnabled)
    gl->glEnable(GL_BLEND);

  gl->glLogicOp(logicOperationMode);
  gl->glDisable(GL_COLOR_LOGIC_OP);
}
