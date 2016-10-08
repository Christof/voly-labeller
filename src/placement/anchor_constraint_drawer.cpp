#include "./anchor_constraint_drawer.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include "../utils/memory.h"
#include "../graphics/vertex_array.h"
#include "./constraint_drawer.h"

AnchorConstraintDrawer::AnchorConstraintDrawer(
    int width, int height, Graphics::Gl *gl,
    std::shared_ptr<Graphics::ShaderManager> shaderManager)
  : width(width), height(height), gl(gl)
{
  Eigen::Affine3f pixelToNDCTransform(
      Eigen::Translation3f(Eigen::Vector3f(-1, -1, 0)) *
      Eigen::Scaling(Eigen::Vector3f(2.0f / width, 2.0f / height, 1)));
  Eigen::Matrix4f pixelToNDC = pixelToNDCTransform.matrix();

  renderData.viewMatrix = pixelToNDC;
  renderData.viewProjectionMatrix = pixelToNDC;

  constraintDrawer = std::make_unique<ConstraintDrawer>(
      gl, shaderManager, ":/shader/constraint.vert",
      ":/shader/quad.geom");

  const int maxLabelCount = 100;
  vertexArray = std::make_unique<Graphics::VertexArray>(gl, GL_POINTS, 2);
  vertexArray->addStream(maxLabelCount, 2);
}

AnchorConstraintDrawer::~AnchorConstraintDrawer()
{
}

void AnchorConstraintDrawer::update(const std::vector<float> &anchors)
{
  vertexArray->updateStream(0, anchors);
}

void AnchorConstraintDrawer::draw(float color, Eigen::Vector2f halfSize)
{
  constraintDrawer->draw(vertexArray.get(), renderData, color, halfSize);
}

void AnchorConstraintDrawer::clear()
{
  gl->glViewport(0, 0, width, height);
  gl->glClearColor(0, 0, 0, 0);
  gl->glClear(GL_COLOR_BUFFER_BIT);
}
