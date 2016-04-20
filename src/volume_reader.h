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
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit VolumeReader(std::string filename);
  virtual ~VolumeReader();

  float *getDataPointer();
  Eigen::Vector3i getSize();
  Eigen::Matrix4f getTransformationMatrix();
  Eigen::Vector3f getSpacing();
  Eigen::Vector3f getPhysicalSize();

 private:
  ImageType::Pointer image;
  float min;
  float max;
  Eigen::Matrix4f transformation;
  Eigen::Vector3i size;
  Eigen::Vector3f spacing;
  Eigen::Vector3f physicalSize;

  void normalizeToCT(itk::ImageRegionIterator<ImageType> imageIterator);
  void normalizeTo01(itk::ImageRegionIterator<ImageType> imageIterator);
  void calculateTransformationMatrix();
  void calculateSizes();
};

#endif  // SRC_VOLUME_READER_H_
