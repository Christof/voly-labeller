#include "./constraint_updater_using_geometry_shader.h"

ConstraintUpdaterUsingGeometryShader::ConstraintUpdaterUsingGeometryShader()
{
}

ConstraintUpdaterUsingGeometryShader::~ConstraintUpdaterUsingGeometryShader()
{
}

void ConstraintUpdaterUsingGeometryShader::drawConstraintRegionFor(
    Eigen::Vector2i anchorPosition, Eigen::Vector2i labelSize,
    Eigen::Vector2i lastAnchorPosition, Eigen::Vector2i lastLabelPosition,
    Eigen::Vector2i lastLabelSize)
{
  /*
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
  // move to ConstraintUpdater
  shaderManager->getShader(shaderId)->setUniform("color", color);
  shaderManager->bind(shaderId, renderData);
  vertexArray->draw();

  if (isBlendingEnabled)
    gl->glEnable(GL_BLEND);

  gl->glLogicOp(logicOperationMode);
  gl->glDisable(GL_COLOR_LOGIC_OP);

  delete vertexArray;
  */
}

void ConstraintUpdaterUsingGeometryShader::drawRegionsForAnchors(
    std::vector<Eigen::Vector2i> anchorPositions, Eigen::Vector2i labelSize)
{
}

void ConstraintUpdaterUsingGeometryShader::clear()
{
}

void ConstraintUpdaterUsingGeometryShader::setIsConnectorShadowEnabled(
    bool enabled)
{
}
