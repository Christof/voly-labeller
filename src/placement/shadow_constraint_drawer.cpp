#include "./shadow_constraint_drawer.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include "../utils/memory.h"
#include "../graphics/vertex_array.h"
#include "./constraint_drawer.h"

ShadowConstraintDrawer::ShadowConstraintDrawer(
    int width, int height, Graphics::Gl *gl,
    std::shared_ptr<Graphics::ShaderManager> shaderManager)
{
  Eigen::Affine3f pixelToNDCTransform(
      Eigen::Translation3f(Eigen::Vector3f(-1, -1, 0)) *
      Eigen::Scaling(Eigen::Vector3f(2.0f / width, 2.0f / height, 1)));
  Eigen::Matrix4f pixelToNDC = pixelToNDCTransform.matrix();

  renderData.viewMatrix = pixelToNDC;
  renderData.viewProjectionMatrix = pixelToNDC;

  constraintDrawer = std::make_unique<ConstraintDrawer>(
      gl, shaderManager, ":/shader/line_constraint.vert",
      ":/shader/constraint.geom");

  const int maxLabelCount = 100;
  vertexArray = std::make_unique<Graphics::VertexArray>(gl, GL_POINTS, 2);
  vertexArray->addStream(maxLabelCount, 2);
  vertexArray->addStream(maxLabelCount, 2);
  vertexArray->addStream(maxLabelCount, 2);
}

void ShadowConstraintDrawer::update(const std::vector<float> &sources,
                                    const std::vector<float> &starts,
                                    const std::vector<float> &ends)
{
  vertexArray->updateStream(0, sources);
  vertexArray->updateStream(1, starts);
  vertexArray->updateStream(2, ends);
}

void ShadowConstraintDrawer::draw(const RenderData &renderData, float color,
                                  Eigen::Vector2f halfSize)
{
  constraintDrawer->draw(vertexArray.get(), renderData, color, halfSize);
}
