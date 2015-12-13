#ifndef SRC_GRAPHICS_QIMAGE_DRAWER_H_

#define SRC_GRAPHICS_QIMAGE_DRAWER_H_

#include <QImage>
#include <memory>
#include <vector>
#include "./drawer.h"

namespace Graphics
{

/**
 * \brief
 *
 *
 */
class QImageDrawer : public Drawer
{
 public:
  std::shared_ptr<QImage> image;

  QImageDrawer(int width, int height);

  void drawElementVector(std::vector<float> positions);

  void clear();
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_QIMAGE_DRAWER_H_
