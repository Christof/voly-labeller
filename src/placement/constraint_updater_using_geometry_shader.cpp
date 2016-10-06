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
      gl, shaderManager, ":/shader/line_constraint.vert",
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

  const int maxLabelCount = 100;
  vertexArray = std::make_unique<Graphics::VertexArray>(gl, GL_POINTS, 2);
  vertexArray->addStream(maxLabelCount, 2);
  vertexArray->addStream(maxLabelCount, 2);
  vertexArray->addStream(maxLabelCount, 2);

  vertexArrayForConnectors =
      std::make_unique<Graphics::VertexArray>(gl, GL_POINTS, 2);
  vertexArrayForConnectors->addStream(2 * maxLabelCount, 2);
  vertexArrayForConnectors->addStream(2 * maxLabelCount, 2);
  vertexArrayForConnectors->addStream(2 * maxLabelCount, 2);

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
  this->labelSize = labelSize.cast<float>();

  addConnectorShadow(anchorPosition, lastAnchorPosition, lastLabelPosition);

  Eigen::Vector2f anchor = anchorPosition.cast<float>();
  Eigen::Vector2f lastHalfSize = 0.5f * lastLabelSize.cast<float>();
  addLabelShadow(anchor, lastLabelPosition, lastHalfSize);
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

  sources.clear();
  starts.clear();
  ends.clear();

  anchors.clear();
  connectorStart.clear();
  connectorEnd.clear();
}

void ConstraintUpdaterUsingGeometryShader::finish()
{
  vertexArray->updateStream(0, sources);
  vertexArray->updateStream(1, starts);
  vertexArray->updateStream(2, ends);

  vertexArrayForConnectors->updateStream(0, anchors);
  vertexArrayForConnectors->updateStream(1, connectorStart);
  vertexArrayForConnectors->updateStream(2, connectorEnd);

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

void ConstraintUpdaterUsingGeometryShader::setIsConnectorShadowEnabled(
    bool enabled)
{
}

void ConstraintUpdaterUsingGeometryShader::addConnectorShadow(
    Eigen::Vector2i anchor, Eigen::Vector2i start, Eigen::Vector2i end)
{
  anchors.push_back(anchor.x());
  anchors.push_back(anchor.y());

  connectorStart.push_back(start.x());
  connectorStart.push_back(start.y());

  connectorEnd.push_back(end.x());
  connectorEnd.push_back(end.y());
}

void ConstraintUpdaterUsingGeometryShader::addLineShadow(Eigen::Vector2f source,
                                                         Eigen::Vector2f start,
                                                         Eigen::Vector2f end)
{
  sources.push_back(source.x());
  sources.push_back(source.y());

  starts.push_back(start.x());
  starts.push_back(start.y());

  ends.push_back(end.x());
  ends.push_back(end.y());
}

void ConstraintUpdaterUsingGeometryShader::addLabelShadow(
    Eigen::Vector2f anchor, Eigen::Vector2i lastLabelPosition,
    Eigen::Vector2f lastHalfSize)
{
  std::vector<Eigen::Vector2f> corners =
      getCornersFor(lastLabelPosition, lastHalfSize);
  std::vector<float> cornerAnchorDistances;
  for (auto corner : corners)
    cornerAnchorDistances.push_back((corner - anchor).squaredNorm());

  int maxIndex = std::distance(cornerAnchorDistances.begin(),
                               std::max_element(cornerAnchorDistances.begin(),
                                                cornerAnchorDistances.end()));

  // Just add a line shadow for two edges of the label.
  // The other two, which are connected to the farthest corner
  // would produce an area which is included in the first one.
  // The following table illustrates which lines must be drawn,
  // between which corners (given by index)
  //
  // maxIndex  line 1  line 2
  // 0         1 to 2  2 to 3
  // 1         2 to 3  3 to 0
  // 2         0 to 1  3 to 0
  // 3         0 to 1  1 to 2
  if (maxIndex == 2 || maxIndex == 3)
    addLineShadow(anchor, corners[0], corners[1]);

  if (maxIndex == 0 || maxIndex == 3)
    addLineShadow(anchor, corners[1], corners[2]);

  if (maxIndex == 0 || maxIndex == 1)
    addLineShadow(anchor, corners[2], corners[3]);

  if (maxIndex == 1 || maxIndex == 2)
    addLineShadow(anchor, corners[3], corners[0]);
}

std::vector<Eigen::Vector2f>
ConstraintUpdaterUsingGeometryShader::getCornersFor(Eigen::Vector2i position,
                                                    Eigen::Vector2f halfSize)
{
  std::vector<Eigen::Vector2f> corners = {
    Eigen::Vector2f(position.x() + halfSize.x(), position.y() + halfSize.y()),
    Eigen::Vector2f(position.x() - halfSize.x(), position.y() + halfSize.y()),
    Eigen::Vector2f(position.x() - halfSize.x(), position.y() - halfSize.y()),
    Eigen::Vector2f(position.x() + halfSize.x(), position.y() - halfSize.y()),
  };

  return corners;
}
