#include "./constraint_updater_using_geometry_shader.h"
#include <Eigen/Geometry>
#include <vector>
#include "../graphics/gl.h"
#include "../graphics/shader_manager.h"
#include "../graphics/shader_program.h"
#include "../graphics/vertex_array.h"
#include "./placement.h"
#include "./constraint_drawer.h"
#include "../utils/memory.h"

ConstraintUpdaterUsingGeometryShader::ConstraintUpdaterUsingGeometryShader(
    int width, int height, Graphics::Gl *gl,
    std::shared_ptr<Graphics::ShaderManager> shaderManager)
  : width(width), height(height), gl(gl), shaderManager(shaderManager)
{
  dilatingDrawer = std::make_unique<ConstraintDrawer>(
      gl, shaderManager, ":/shader/constraint2.vert",
      ":/shader/constraint.geom");
  quadDrawer = std::make_unique<ConstraintDrawer>(
      gl, shaderManager, ":/shader/constraint.vert", ":/shader/quad.geom");

  Eigen::Affine3f pixelToNDCTransform(
      Eigen::Translation3f(Eigen::Vector3f(-1, -1, 0)) *
      Eigen::Scaling(Eigen::Vector3f(2.0f / width, 2.0f / height, 1)));
  pixelToNDC = pixelToNDCTransform.matrix();

  labelShadowColor = Placement::labelShadowValue / 255.0f;
  connectorShadowColor = Placement::connectorShadowValue / 255.0f;
  anchorConstraintColor = Placement::anchorConstraintValue / 255.0f;

  vertexArray = std::make_unique<Graphics::VertexArray>(gl, GL_POINTS, 2);
  vertexArray->addStream(100, 2);
  vertexArray->addStream(100, 2);
  vertexArray->addStream(100, 2);

  vertexArrayForConnectors =
      std::make_unique<Graphics::VertexArray>(gl, GL_POINTS, 2);
  vertexArrayForConnectors->addStream(100, 2);
  vertexArrayForConnectors->addStream(100, 2);
  vertexArrayForConnectors->addStream(100, 2);

  vertexArrayForAnchors =
      std::make_unique<Graphics::VertexArray>(gl, GL_POINTS, 2);
  vertexArrayForAnchors->addStream(100, 2);
}

ConstraintUpdaterUsingGeometryShader::~ConstraintUpdaterUsingGeometryShader()
{
}

void ConstraintUpdaterUsingGeometryShader::drawConstraintRegionFor(
    Eigen::Vector2i anchorPosition, Eigen::Vector2i labelSize,
    Eigen::Vector2i lastAnchorPosition, Eigen::Vector2i lastLabelPosition,
    Eigen::Vector2i lastLabelSize)
{
  fillForConnectorShadowRegion(anchorPosition, lastAnchorPosition,
                               lastLabelPosition);

  Eigen::Vector2f lastHalfSize = 0.5f * lastLabelSize.cast<float>();
  std::vector<Eigen::Vector2f> corners = {
    Eigen::Vector2f(lastLabelPosition.x() + lastHalfSize.x(),
                    lastLabelPosition.y() + lastHalfSize.y()),
    Eigen::Vector2f(lastLabelPosition.x() - lastHalfSize.x(),
                    lastLabelPosition.y() + lastHalfSize.y()),
    Eigen::Vector2f(lastLabelPosition.x() - lastHalfSize.x(),
                    lastLabelPosition.y() - lastHalfSize.y()),
    Eigen::Vector2f(lastLabelPosition.x() + lastHalfSize.x(),
                    lastLabelPosition.y() - lastHalfSize.y()),
  };
  Eigen::Vector2f anchor = anchorPosition.cast<float>();
  std::vector<float> cornerAnchorDistances;
  for (auto corner : corners)
    cornerAnchorDistances.push_back((corner - anchor).squaredNorm());

  int maxIndex = std::distance(cornerAnchorDistances.begin(),
                               std::max_element(cornerAnchorDistances.begin(),
                                                cornerAnchorDistances.end()));

  std::vector<float> anchors = { anchor.x(), anchor.y(), anchor.x(),
                                 anchor.y() };
  std::vector<float> connectorStart;
  std::vector<float> connectorEnd;
  if (maxIndex == 2 || maxIndex == 3)
  {
    connectorStart.push_back(corners[0].x());
    connectorStart.push_back(corners[0].y());

    connectorEnd.push_back(corners[1].x());
    connectorEnd.push_back(corners[1].y());
  }

  if (maxIndex == 0 || maxIndex == 3)
  {
    connectorStart.push_back(corners[1].x());
    connectorStart.push_back(corners[1].y());

    connectorEnd.push_back(corners[2].x());
    connectorEnd.push_back(corners[2].y());
  }

  if (maxIndex == 0 || maxIndex == 1)
  {
    connectorStart.push_back(corners[2].x());
    connectorStart.push_back(corners[2].y());

    connectorEnd.push_back(corners[3].x());
    connectorEnd.push_back(corners[3].y());
  }

  if (maxIndex == 1 || maxIndex == 2)
  {
    connectorStart.push_back(corners[3].x());
    connectorStart.push_back(corners[3].y());

    connectorEnd.push_back(corners[0].x());
    connectorEnd.push_back(corners[0].y());
  }

  vertexArray->updateStream(0, anchors);
  vertexArray->updateStream(1, connectorStart);
  vertexArray->updateStream(2, connectorEnd);

  RenderData renderData;
  renderData.viewMatrix = pixelToNDC;
  renderData.viewProjectionMatrix = pixelToNDC;

  Eigen::Vector2f borderPixel(2.0f, 2.0f);
  Eigen::Vector2f sizeWithBorder = labelSize.cast<float>() + borderPixel;
  Eigen::Vector2f halfSize =
      sizeWithBorder.cwiseQuotient(Eigen::Vector2f(width, height));

  dilatingDrawer->draw(vertexArray.get(), renderData, labelShadowColor,
                       halfSize);

  dilatingDrawer->draw(vertexArrayForConnectors.get(), renderData,
                       connectorShadowColor, halfSize);
}

void ConstraintUpdaterUsingGeometryShader::fillForConnectorShadowRegion(
    Eigen::Vector2i anchorPosition, Eigen::Vector2i lastAnchorPosition,
    Eigen::Vector2i lastLabelPosition)
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

  vertexArrayForConnectors->updateStream(0, anchors);
  vertexArrayForConnectors->updateStream(1, connectorStart);
  vertexArrayForConnectors->updateStream(2, connectorEnd);
}

void ConstraintUpdaterUsingGeometryShader::drawRegionsForAnchors(
    std::vector<Eigen::Vector2i> anchorPositions, Eigen::Vector2i labelSize)
{
  std::vector<float> positions(anchorPositions.size() * 2);
  size_t index = 0;
  for (auto &anchorPosition : anchorPositions)
  {
    positions[index++] = anchorPosition.x();
    positions[index++] = anchorPosition.y();
  }
  assert(positions.size() == index);

  vertexArrayForAnchors->updateStream(0, positions);

  RenderData renderData;
  renderData.viewMatrix = pixelToNDC;
  renderData.viewProjectionMatrix = pixelToNDC;

  Eigen::Vector2f constraintSize = 2.0f * labelSize.cast<float>();
  Eigen::Vector2f halfSize =
      constraintSize.cwiseQuotient(0.5 * Eigen::Vector2f(width, height));

  quadDrawer->draw(vertexArrayForAnchors.get(), renderData,
                   anchorConstraintColor, halfSize);
  vertexArrayForAnchors->draw();
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
