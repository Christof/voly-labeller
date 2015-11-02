#ifndef SRC_UTILS_IMAGE_PERSISTER_H_

#define SRC_UTILS_IMAGE_PERSISTER_H_

#include <string>
#include <Magick++.h>

/**
 * \brief Provides static functions to load and save images
 *
 */
class ImagePersister
{
 public:
  template <class T>
  static void saveRGBA32F(T *data, int width, int height, std::string filename)
  {
    Magick::InitializeMagick("");
    Magick::Image image(width, height, "RGBA", Magick::StorageType::FloatPixel,
                        data);
    image.write(filename);
  }
};

#endif  // SRC_UTILS_IMAGE_PERSISTER_H_
