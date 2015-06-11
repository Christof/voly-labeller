#ifndef SRC_VOLUME_READER_H_

#define SRC_VOLUME_READER_H_

#include <string>
#include <itkImage.h>
#include <itkImageRegionIterator.h>
#include <Eigen/Core>

typedef itk::Image<float, 3> ImageType;

/**
 * \brief
 *
 *
 */
class VolumeReader
{
 public:
  VolumeReader(std::string filename);
  virtual ~VolumeReader();

  float* getDataPointer();
  Eigen::Vector3i getSize();
 private:
  ImageType::Pointer image;
  float min;
  float max;

  void normalizeToCT(itk::ImageRegionIterator<ImageType> imageIterator);
  void normalizeTo01(itk::ImageRegionIterator<ImageType> imageIterator);
};

#endif  // SRC_VOLUME_READER_H_
