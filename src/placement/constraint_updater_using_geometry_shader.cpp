#include "./constraint_updater_using_geometry_shader.h"
#include "../graphics/gl.h"
#include "../graphics/shader_manager.h"
#include "../graphics/shader_program.h"
#include "../graphics/vertex_array.h"
#include <Eigen/Geometry>

ConstraintUpdaterUsingGeometryShader::ConstraintUpdaterUsingGeometryShader(
    int width, int height, Graphics::Gl *gl,
    std::shared_ptr<Graphics::ShaderManager> shaderManager)
  : width(width), height(height), gl(gl), shaderManager(shaderManager)
{
  shaderId = shaderManager->addShader(":/shader/constraint2.vert",
                                      ":/shader/constraint.geom",
                                      ":/shader/colorImmediate.frag");

  Eigen::Affine3f pixelToNDCTransform(
      Eigen::Translation3f(Eigen::Vector3f(-1, 1, 0)) *
      Eigen::Scaling(Eigen::Vector3f(2.0f / width, -2.0f / height, 1)));
  pixelToNDC = pixelToNDCTransform.matrix();
}

ConstraintUpdaterUsingGeometryShader::~ConstraintUpdaterUsingGeometryShader()
{
}

void ConstraintUpdaterUsingGeometryShader::drawConstraintRegionFor(
    Eigen::Vector2i anchorPosition, Eigen::Vector2i labelSize,
    Eigen::Vector2i lastAnchorPosition, Eigen::Vector2i lastLabelPosition,
    Eigen::Vector2i lastLabelSize)
{
  std::vector<float> anchors = { static_cast<float>(anchorPosition.x()),
                                 static_cast<float>(anchorPosition.y()) };
  std::vector<float> connectorStart = {
    static_cast<float>(lastAnchorPosition.x()),
    static_cast<float>(lastAnchorPosition.y())
  };
  std::vector<float> connectorEnd = { static_cast<float>(lastLabelPosition.x()),
                                      static_cast<float>(
                                          lastLabelPosition.y()) };

  Graphics::VertexArray *vertexArray =
      new Graphics::VertexArray(gl, GL_POINTS, 2);
  vertexArray->addStream(anchors, 2);
  vertexArray->addStream(connectorStart, 2);
  vertexArray->addStream(connectorEnd, 2);

  GLboolean isBlendingEnabled = gl->glIsEnabled(GL_BLEND);
  gl->glEnable(GL_BLEND);
  GLint logicOperationMode;
  gl->glGetIntegerv(GL_LOGIC_OP_MODE, &logicOperationMode);
  gl->glEnable(GL_COLOR_LOGIC_OP);
  gl->glLogicOp(GL_OR);

  RenderData renderData;
  renderData.viewMatrix = pixelToNDC;
  renderData.viewProjectionMatrix = pixelToNDC;
  // move to ConstraintUpdater
  shaderManager->getShader(shaderId)->setUniform("color", 1.0f);
  shaderManager->bind(shaderId, renderData);
  vertexArray->draw();

  if (isBlendingEnabled)
    gl->glEnable(GL_BLEND);

  gl->glLogicOp(logicOperationMode);
  gl->glDisable(GL_COLOR_LOGIC_OP);

  delete vertexArray;
}

void ConstraintUpdaterUsingGeometryShader::drawRegionsForAnchors(
    std::vector<Eigen::Vector2i> anchorPositions, Eigen::Vector2i labelSize)
{
}

void ConstraintUpdaterUsingGeometryShader::clear()
{
  gl->glViewport(0, 0, width, height);
  gl->glClearColor(0, 0, 0, 0);
  gl->glClear(GL_COLOR_BUFFER_BIT);
}

void ConstraintUpdaterUsingGeometryShader::setIsConnectorShadowEnabled(
    bool enabled)
{
}
