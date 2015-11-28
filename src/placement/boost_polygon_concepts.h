#ifndef SRC_PLACEMENT_BOOST_POLYGON_CONCEPTS_H_

#define SRC_PLACEMENT_BOOST_POLYGON_CONCEPTS_H_

#include <Eigen/Core>
#include <boost/polygon/polygon.hpp>
#include <boost/geometry.hpp>

typedef boost::geometry::model::polygon<Eigen::Vector2i, false, false> poly;

namespace boost
{
namespace polygon
{
template <> struct geometry_concept<Eigen::Vector2i>
{
  typedef point_concept type;
};

// Then we specialize the gtl point traits for our point type
template <> struct point_traits<Eigen::Vector2i>
{
  typedef int coordinate_type;

  static inline coordinate_type get(const Eigen::Vector2i &point,
                                    orientation_2d orient)
  {
    if (orient == HORIZONTAL)
      return point.x();
    return point.y();
  }
};

template <> struct point_mutable_traits<Eigen::Vector2i>
{
  typedef int coordinate_type;

  static inline void set(Eigen::Vector2i &point, orientation_2d orient,
                         int value)
  {
    if (orient == HORIZONTAL)
      point.x() = value;
    else
      point.y() = value;
  }
  static inline Eigen::Vector2i construct(int x_value, int y_value)
  {
    return Eigen::Vector2i(x_value, y_value);
  }
};
}
}
namespace boost
{
namespace polygon
{
// first register CPolygon as a polygon_concept type
template <> struct geometry_concept<poly>
{
  typedef polygon_concept type;
};

template <> struct polygon_traits<poly>
{
  typedef int coordinate_type;
  typedef std::vector<Eigen::Vector2i>::const_iterator iterator_type;
  typedef Eigen::Vector2i point_type;

  // Get the begin iterator
  static inline iterator_type begin_points(const poly &t)
  {
    return t.outer().begin();
  }

  // Get the end iterator
  static inline iterator_type end_points(const poly &t)
  {
    return t.outer().end();
  }

  // Get the number of sides of the polygon
  static inline std::size_t size(const poly &t)
  {
    return t.outer().size();
  }

  // Get the winding direction of the polygon
  static inline winding_direction winding(const poly &t)
  {
    return unknown_winding;
  }
};

template <> struct polygon_mutable_traits<poly>
{
  // expects stl style iterators
  template <typename iT>
  static inline poly &set_points(poly &t, iT input_begin, iT input_end)
  {
    t.outer().clear();
    t.outer().insert(t.outer().end(), input_begin, input_end);
    return t;
  }
};
}
}

#endif  // SRC_PLACEMENT_BOOST_POLYGON_CONCEPTS_H_
