#ifndef SRC_UTILS_IMAGE_PERSISTER_H_

#define SRC_UTILS_IMAGE_PERSISTER_H_

#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include <string>
#include <vector>
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

  static void saveR32F(float *data, int width, int height, std::string filename)
  {
    Magick::InitializeMagick("");
    Magick::Image image(width, height, "R", Magick::StorageType::FloatPixel,
                        data);
    image.write(filename);
  }

  static std::vector<Eigen::Vector4f> loadRGBA32F(std::string filename)
  {
    if (!boost::filesystem::exists(filename))
      throw std::invalid_argument("Image '" + filename + "' doesn't exist.");

    Magick::Image image(filename);
    std::vector<Eigen::Vector4f> result(image.columns() * image.rows());

    image.write(0, 0, image.columns(), image.rows(), "RGBA",
                Magick::StorageType::FloatPixel, result.data());

    return result;
  }

  static std::vector<float> loadR32F(std::string filename)
  {
    if (!boost::filesystem::exists(filename))
      throw std::invalid_argument("Image '" + filename + "' doesn't exist.");

    Magick::Image image(filename);
    std::vector<float> result(image.columns() * image.rows());

    image.write(0, 0, image.columns(), image.rows(), "R",
                Magick::StorageType::FloatPixel, result.data());

    return result;
  }
};

#endif  // SRC_UTILS_IMAGE_PERSISTER_H_
