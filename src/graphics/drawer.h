#ifndef SRC_GRAPHICS_DRAWER_H_

#define SRC_GRAPHICS_DRAWER_H_

#include <vector>

namespace Graphics
{

/**
 * \brief Interface for drawing 2d polygons
 *
 */
class Drawer
{
 public:
  virtual void drawElementVector(std::vector<float> positions, float weight) = 0;
  virtual void clear() = 0;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_DRAWER_H_
