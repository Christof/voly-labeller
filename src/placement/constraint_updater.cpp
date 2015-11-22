#include "./constraint_updater.h"
#include <Eigen/Geometry>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <vector>
#include "../graphics/vertex_array.h"
#include "../graphics/render_data.h"

namespace bg = boost::geometry;

BOOST_GEOMETRY_REGISTER_POINT_2D(Eigen::Vector2i, int, cs::cartesian, x(), y())

// ccw, open polygon
typedef bg::model::polygon<Eigen::Vector2i, false, false> polygon;

ConstraintUpdater::ConstraintUpdater(
    Graphics::Gl *gl, std::shared_ptr<Graphics::ShaderManager> shaderManager,
    int width, int height)
  : gl(gl), shaderManager(shaderManager), width(width), height(height)
{
  shaderId = shaderManager->addShader(":/shader/constraint.vert",
                                      ":/shader/colorImmediate.frag");

  Eigen::Affine3f pixelToNDCTransform(
      Eigen::Translation3f(Eigen::Vector3f(-1, 1, 0)) *
      Eigen::Scaling(Eigen::Vector3f(2.0f / width, -2.0f / height, 1)));
  pixelToNDC = pixelToNDCTransform.matrix();
}

void ConstraintUpdater::addLabel(Eigen::Vector2i anchorPosition,
                                 Eigen::Vector2i labelSize,
                                 Eigen::Vector2i lastAnchorPosition,
                                 Eigen::Vector2i lastLabelPosition,
                                 Eigen::Vector2i lastLabelSize)
{
  Eigen::Vector2i halfSize = lastLabelSize / 2;

  polygon oldLabel;
  oldLabel.outer().push_back(
      Eigen::Vector2i(lastLabelPosition.x() - halfSize.x(),
                      lastLabelPosition.y() - halfSize.y()));
  oldLabel.outer().push_back(
      Eigen::Vector2i(lastLabelPosition.x() + halfSize.x(),
                      lastLabelPosition.y() - halfSize.y()));
  oldLabel.outer().push_back(
      Eigen::Vector2i(lastLabelPosition.x() + halfSize.x(),
                      lastLabelPosition.y() + halfSize.y()));
  oldLabel.outer().push_back(
      Eigen::Vector2i(lastLabelPosition.x() - halfSize.x(),
                      lastLabelPosition.y() + halfSize.y()));

  int border = 2;
  polygon newLabelDilation;
  newLabelDilation.outer().push_back(
      Eigen::Vector2i(-labelSize.x() - border, -labelSize.y() - border));
  newLabelDilation.outer().push_back(
      Eigen::Vector2i(0.0f + border, -labelSize.y() - border));
  newLabelDilation.outer().push_back(
      Eigen::Vector2i(0.0f + border, 0.0f + border));
  newLabelDilation.outer().push_back(
      Eigen::Vector2i(-labelSize.x() - border, 0.0f + border));


  drawPolygon(oldLabel.outer());
}

void ConstraintUpdater::drawPolygon(std::vector<Eigen::Vector2i> polygon)
{
  std::vector<float> positions;
  if (polygon.size() > 0)
  {
    auto point = polygon[0];
    positions.push_back(point.x());
    positions.push_back(height - point.y());
    positions.push_back(0.0f);
  }
  for (auto point : polygon)
  {
    positions.push_back(point.x());
    positions.push_back(height - point.y());
    positions.push_back(0.0f);
  }

  Graphics::VertexArray *vertexArray =
      new Graphics::VertexArray(gl, GL_TRIANGLE_FAN);
  vertexArray->addStream(positions);

  RenderData renderData;
  renderData.modelMatrix = Eigen::Matrix4f::Identity();
  renderData.viewMatrix = pixelToNDC;
  renderData.projectionMatrix = Eigen::Matrix4f::Identity();
  shaderManager->bind(shaderId, renderData);
  vertexArray->draw();
}

