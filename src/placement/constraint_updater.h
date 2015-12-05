#ifndef SRC_PLACEMENT_CONSTRAINT_UPDATER_H_

#define SRC_PLACEMENT_CONSTRAINT_UPDATER_H_

#include <Eigen/Core>
#include <boost/polygon/polygon.hpp>
#include <boost/geometry.hpp>
#include <vector>
#include <utility>
#include "../graphics/gl.h"
#include "../graphics/shader_manager.h"

// ccw, closed polygon
typedef boost::geometry::model::polygon<Eigen::Vector2i, false, true> polygon;
typedef boost::polygon::point_data<int> point;
typedef boost::polygon::polygon_set_data<int> polygon_set;
typedef boost::polygon::polygon_data<int> ppolygon;
typedef std::pair<point, point> edge;

/**
 * \brief
 *
 *
 */
class ConstraintUpdater
{
 public:
  ConstraintUpdater(Graphics::Gl *gl,
                    std::shared_ptr<Graphics::ShaderManager> shaderManager,
                    int width, int height);

  void drawConstraintRegionFor(Eigen::Vector2i anchorPosition,
                               Eigen::Vector2i labelSize,
                               Eigen::Vector2i lastAnchorPosition,
                               Eigen::Vector2i lastLabelPosition,
                               Eigen::Vector2i lastLabelSize);

  void clear();

 private:
  Graphics::Gl *gl;
  std::shared_ptr<Graphics::ShaderManager> shaderManager;
  int width;
  int height;

  int shaderId;
  Eigen::Matrix4f pixelToNDC;

  std::vector<float> positions;

  template <typename edge>
  void convolveTwoSegements(polygon &polygon, const edge &a, const edge &b);
  template <typename itrT1, typename itrT2>
  void convolveTwoPointSequences(itrT1 ab, itrT1 ae, itrT2 bb, itrT2 be);
  template <typename Polygon>
  void convolveTwoPolygons(const Polygon &a, const Polygon &b);
  template <typename Polygon>
  void addPolygonToPositions(const Polygon &polygon);

  template <class T> void drawPolygon(std::vector<T> polygon);

  void drawElementVector(std::vector<float> positions);
};

#endif  // SRC_PLACEMENT_CONSTRAINT_UPDATER_H_
