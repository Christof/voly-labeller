#include "./constraint_updater.h"
#include "../graphics/vertex_array.h"
#include "../graphics/render_data.h"

ConstraintUpdater::ConstraintUpdater(
    Graphics::Gl *gl, std::shared_ptr<Graphics::ShaderManager> shaderManager,
    int width, int height)
  : gl(gl), shaderManager(shaderManager), width(width), height(height)
{
  shaderId = shaderManager->addShader(":/shader/pass.vert",
                                      ":/shader/colorImmediate.frag");
}

void ConstraintUpdater::addLabel(Eigen::Vector2i anchorPosition,
                                 Eigen::Vector2i lastAnchorPosition,
                                 Eigen::Vector2i lastLabelPosition,
                                 Eigen::Vector2i lastLabelSize)
{
  std::vector<float> positions = {
    -0.5f, 0.5f, 0, 0.5f, 0.5f, 0, 0.5f, -0.5f, 0
  };
  Graphics::VertexArray *vertexArray =
      new Graphics::VertexArray(gl, GL_TRIANGLES);
  vertexArray->addStream(positions);

  RenderData renderData;
  renderData.modelMatrix = Eigen::Matrix4f::Identity();
  renderData.viewMatrix = Eigen::Matrix4f::Identity();
  renderData.projectionMatrix = Eigen::Matrix4f::Identity();
  shaderManager->bind(shaderId, renderData);
  vertexArray->draw();
}
