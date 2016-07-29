#ifndef SRC_UTILS_IMAGE_PERSISTER_H_

#define SRC_UTILS_IMAGE_PERSISTER_H_

#include <Eigen/Core>
#include <QImage>
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

  static void saveR8I(unsigned char *data, int width, int height,
                      std::string filename)
  {
    QImage image(data, width, height, QImage::Format_Grayscale8);
    image.save(QString(filename.c_str()));
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

  static std::vector<unsigned char> loadR8I(std::string filename)
  {
    QImage image(filename.c_str());

    std::vector<unsigned char> result(image.byteCount());

    std::memcpy(result.data(), image.bits(), image.byteCount());

    return result;
  }
};

#endif  // SRC_UTILS_IMAGE_PERSISTER_H_
