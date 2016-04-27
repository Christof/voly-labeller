#if _WIN32
#pragma warning (disable: 4996)
#endif

#include "./volume_reader.h"
#include <string>
#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>
#include <itkMinimumMaximumImageCalculator.h>
#include "./utils/path_helper.h"

typedef itk::ImageFileReader<ImageType> VolumeReaderType;
typedef itk::MinimumMaximumImageCalculator<ImageType> MinMaxCalculator;

VolumeReader::VolumeReader(std::string filename)
{
  VolumeReaderType::Pointer reader = VolumeReaderType::New();
  reader->SetFileName(absolutePathOfProjectRelativePath(filename).c_str());

  reader->Update();
  image = reader->GetOutput();

  itk::ImageRegionIterator<ImageType> imageIterator(
      image, image->GetRequestedRegion());
  MinMaxCalculator::Pointer calculator = MinMaxCalculator::New();

  calculator->SetImage(image);
  calculator->Compute();

  max = calculator->GetMaximum();
  min = calculator->GetMinimum();

  bool isCT = min < 0;
  if (isCT)
    normalizeToCT(imageIterator);
  else
    normalizeTo01(imageIterator);

  calculateSizes();
  calculateTransformationMatrix();
}

VolumeReader::~VolumeReader()
{
}

float *VolumeReader::getDataPointer()
{
  return image->GetBufferPointer();
}

Eigen::Matrix4f VolumeReader::getTransformationMatrix()
{
  return transformation;
}

Eigen::Vector3i VolumeReader::getSize()
{
  return size;
}

Eigen::Vector3f VolumeReader::getSpacing()
{
  return spacing;
}

Eigen::Vector3f VolumeReader::getPhysicalSize()
{
  return physicalSize;
}

void
VolumeReader::normalizeToCT(itk::ImageRegionIterator<ImageType> imageIterator)
{
  const float maxAllowedValue = 3071;
  const float minAllowedValue = -1024;

  for (imageIterator.GoToBegin(); !imageIterator.IsAtEnd(); ++imageIterator)
  {
    float value = (imageIterator.Get() > maxAllowedValue) ? maxAllowedValue
                                                          : imageIterator.Get();
    value = (value < minAllowedValue) ? minAllowedValue : value;

    auto normalizedValue =
        (value - minAllowedValue) / (maxAllowedValue - minAllowedValue);
    imageIterator.Set(normalizedValue);
  }

  if (min < minAllowedValue)
  {
    std::cout << "VolumeReader adjusting min value!" << std::endl;
    min = minAllowedValue;
  }

  if (max > maxAllowedValue)
  {
    std::cout << "VolumeReader adjusting max value!" << std::endl;
    max = maxAllowedValue;
  }
}

void
VolumeReader::normalizeTo01(itk::ImageRegionIterator<ImageType> imageIterator)
{
  for (imageIterator.GoToBegin(); !imageIterator.IsAtEnd(); ++imageIterator)
  {
    auto normalizedValue = (imageIterator.Get() - min) / (max - min);
    imageIterator.Set(normalizedValue);
  }
}

void VolumeReader::calculateTransformationMatrix()
{
  itk::Point<float, 3> originItk = image->GetOrigin();
  Eigen::Vector3f offset = 0.5f * getPhysicalSize().cast<float>();
  Eigen::Vector3f origin =
      0.001f * Eigen::Vector3f(originItk[0], originItk[1], originItk[2]) +
      offset;

  ImageType::DirectionType directionItk = image->GetDirection();

  Eigen::Matrix3f rotation;
  for (int rowIndex = 0; rowIndex < 3; ++rowIndex)
    for (int columnIndex = 0; columnIndex < 3; ++columnIndex)
      rotation(rowIndex, columnIndex) = directionItk(rowIndex, columnIndex);

  transformation = Eigen::Matrix4f::Identity();
  transformation.block<3, 3>(0, 0) = rotation;
  transformation.col(3).head<3>() = origin;
}

void VolumeReader::calculateSizes()
{
  ImageType::RegionType region = image->GetLargestPossibleRegion();
  ImageType::SizeType sizeItk = region.GetSize();
  size = Eigen::Vector3i(sizeItk[0], sizeItk[1], sizeItk[2]);

  ImageType::SpacingType spacingItk = image->GetSpacing();
  spacing =
      Eigen::Vector3f(spacingItk[0], spacingItk[1], spacingItk[2]) * 0.001f;

  physicalSize = spacing.cwiseProduct(size.cast<float>());
}

