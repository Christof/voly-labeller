#ifndef TEST_QIMAGE_DRAWER_WITH_UPDATING_H_

#define TEST_QIMAGE_DRAWER_WITH_UPDATING_H_

#include "../src/graphics/qimage_drawer.h"
#include "./cuda_array_mapper.h"

/**
 * \brief Specialization of Graphics::QImageDrawer which updates an underlying
 * CudaArrayMapper
 *
 * Currently only a type of unsigend char is supported for the CudaArrayMapper.
 */
class QImageDrawerWithUpdating : public Graphics::QImageDrawer
{
 public:
  QImageDrawerWithUpdating(
      int width, int height,
      std::shared_ptr<CudaArrayMapper<unsigned char>> texture)
    : QImageDrawer(width, height), texture(texture)
  {
  }

  virtual void drawElementVector(std::vector<float> positions)
  {
    Graphics::QImageDrawer::drawElementVector(positions);

    const unsigned char *data = image->constBits();
    std::vector<unsigned char> newData(image->width() * image->height(), 0.0f);
    for (int i = 0; i < image->byteCount(); ++i)
    {
      newData[i] = data[i] > 0 || texture->getDataAt(i) > 0 ? 255 : 0;
    }

    texture->updateData(newData);
  }

  virtual void clear()
  {
    Graphics::QImageDrawer::clear();

    std::vector<unsigned char> newData(image->width() * image->height(), 0);
    texture->updateData(newData);
  }

 private:
  std::shared_ptr<CudaArrayMapper<unsigned char>> texture;
};

#endif  // TEST_QIMAGE_DRAWER_WITH_UPDATING_H_
