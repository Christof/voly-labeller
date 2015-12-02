#ifndef SRC_PLACEMENT_CONSTRAINT_UPDATER_H_

#define SRC_PLACEMENT_CONSTRAINT_UPDATER_H_

#include <Eigen/Core>
#include <vector>
#include "../graphics/gl.h"
#include "../graphics/shader_manager.h"

#include <boost/polygon/polygon.hpp>
#include <boost/geometry.hpp>
namespace bg = boost::geometry;

// ccw, open polygon
typedef bg::model::polygon<Eigen::Vector2i, false, false> polygon;
typedef boost::polygon::point_data<int> point;
typedef boost::polygon::polygon_set_data<int> polygon_set;
typedef boost::polygon::polygon_with_holes_data<int> ppolygon;
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

  void addLabel(Eigen::Vector2i anchorPosition, Eigen::Vector2i labelSize,
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

  void convolveTwoSegements(const edge &a, const edge &b);
  template <typename itrT1, typename itrT2>
  void convolveTwoPointSequences(itrT1 ab, itrT1 ae, itrT2 bb, itrT2 be);
  template <typename polygon_set, typename ppolygon>
  void convolveTwoPolygons(polygon_set &result, const ppolygon &a,
                           const ppolygon &b);
  std::vector<boost::polygon::polygon_with_holes_data<int>>
  minkowskiSum(const polygon &a, const polygon &b);

  template <class T> void drawPolygon(std::vector<T> polygon);

  void drawElementVector(std::vector<float> positions);
};

#endif  // SRC_PLACEMENT_CONSTRAINT_UPDATER_H_
