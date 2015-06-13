#ifndef SRC_VOLUME_READER_H_

#define SRC_VOLUME_READER_H_

#include <Eigen/Core>
#include <string>
#include <itkImage.h>
#include <itkImageRegionIterator.h>

typedef itk::Image<float, 3> ImageType;

/**
 * \brief Read volume data and meta information from given filename
 *
 */
class VolumeReader
{
 public:
  explicit VolumeReader(std::string filename);
  virtual ~VolumeReader();

  float *getDataPointer();
  Eigen::Vector3i getSize();
  Eigen::Matrix4f getTransformationMatrix();
  Eigen::Vector3f getSpacing();
  Eigen::Vector3f getPhysicalSize();
  bool isCT();

 private:
  ImageType::Pointer image;
  float min;
  float max;

  void normalizeToCT(itk::ImageRegionIterator<ImageType> imageIterator);
  void normalizeTo01(itk::ImageRegionIterator<ImageType> imageIterator);
};

#endif  // SRC_VOLUME_READER_H_
