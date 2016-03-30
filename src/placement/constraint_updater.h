#ifndef SRC_PLACEMENT_CONSTRAINT_UPDATER_H_

#define SRC_PLACEMENT_CONSTRAINT_UPDATER_H_

#include <Eigen/Core>
#include <boost/polygon/polygon.hpp>
#include <boost/geometry.hpp>
#include <vector>
#include <utility>
#include <polyclipping/clipper.hpp>
#include "../graphics/drawer.h"

// ccw, closed polygon
typedef boost::geometry::model::polygon<Eigen::Vector2i, false, true> polygon;
typedef boost::polygon::point_data<int> point;
typedef boost::polygon::polygon_set_data<int> polygon_set;
typedef boost::polygon::polygon_data<int> ppolygon;
typedef std::pair<point, point> edge;

/**
 * \brief Updates the constraint buffer by drawing occupied regions for already
 * placed labels
 *
 * Each placed label makes two regions unusable for future labels:
 * - A dilated shadow region which is created by using a virtual light position
 *   at the anchor point of the next placed label and using already placed
 *   label boxes as shadow casters.
 * - A dilated shadow region which is also created by using a virtual light
 *   position at the anchor point and using the line between an already placed
 *   label's anchor and the label box center.
 * The dilation size is determined by the new label size and a border.
 *
 *
 * For each newly placed label the already bound ConstraintBufferObject must
 * be cleared by calling #clear.
 * Afterwards #drawConstraintRegionFor must be called for each already placed
 * label.
 */
class ConstraintUpdater
{
 public:
  ConstraintUpdater(std::shared_ptr<Graphics::Drawer> drawer, int width,
                    int height);

  void drawConstraintRegionFor(Eigen::Vector2i anchorPosition,
                               Eigen::Vector2i labelSize,
                               Eigen::Vector2i lastAnchorPosition,
                               Eigen::Vector2i lastLabelPosition,
                               Eigen::Vector2i lastLabelSize);

  void clear();
  void setIsConnectorShadowEnabled(bool enabled);

 private:
  std::shared_ptr<Graphics::Drawer> drawer;
  int width;
  int height;
  bool isConnectorShadowEnabled = true;
  float labelShadowColor;
  float connectorShadowColor;

  std::vector<float> positions;

  void drawConnectorShadowRegion(Eigen::Vector2i anchorPosition,
                                 Eigen::Vector2i lastAnchorPosition,
                                 Eigen::Vector2i lastLabelPosition,
                                 const ClipperLib::Path &newLabel);
  void drawLabelShadowRegion(Eigen::Vector2i anchorPosition,
                             Eigen::Vector2i lastLabelPosition,
                             Eigen::Vector2i lastLabelSize,
                             const ClipperLib::Path &newLabel);

  template <typename edge>
  void convolveTwoSegements(polygon &polygon, const edge &a, const edge &b);
  template <typename itrT1, typename itrT2>
  void convolveTwoPointSequences(itrT1 ab, itrT1 ae, itrT2 bb, itrT2 be);
  template <typename Polygon>
  void convolveTwoPolygons(const Polygon &a, const Polygon &b);
  template <typename Polygon>
  void addPolygonToPositions(const Polygon &polygon);

  template <class T> void drawPolygon(std::vector<T> polygon);
  void drawPolygon(ClipperLib::Path polygon);
};

#endif  // SRC_PLACEMENT_CONSTRAINT_UPDATER_H_
