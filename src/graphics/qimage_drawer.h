#ifndef SRC_GRAPHICS_QIMAGE_DRAWER_H_

#define SRC_GRAPHICS_QIMAGE_DRAWER_H_

#include <QImage>
#include <memory>
#include <vector>
#include "./drawer.h"

namespace Graphics
{

/**
 * \brief Drawer implementation which draws into a QImage
 *
 * This is a good fit for testing because it doesn't require an OpenGL context
 * to be created.
 */
class QImageDrawer : public Drawer
{
 public:
  std::shared_ptr<QImage> image;

  QImageDrawer(int width, int height);

  void drawElementVector(std::vector<float> positions, float weight);

  void clear();
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_QIMAGE_DRAWER_H_
