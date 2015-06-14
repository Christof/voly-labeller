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

  if (isCT())
    normalizeToCT(imageIterator);
  else
    normalizeTo01(imageIterator);
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

  Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
  transformation.block<3, 3>(0, 0) = rotation;
  transformation.col(3).head<3>() = origin;

  return transformation;
}

Eigen::Vector3i VolumeReader::getSize()
{
  ImageType::RegionType region = image->GetLargestPossibleRegion();
  ImageType::SizeType size = region.GetSize();

  return Eigen::Vector3i(size[0], size[1], size[2]);
}

Eigen::Vector3f VolumeReader::getSpacing()
{
  ImageType::SpacingType spacingItk = image->GetSpacing();
  return Eigen::Vector3f(spacingItk[0], spacingItk[1], spacingItk[2]) * 0.001f;
}

Eigen::Vector3f VolumeReader::getPhysicalSize()
{
  return getSpacing().cwiseProduct(getSize().cast<float>());
}

bool VolumeReader::isCT()
{
  auto size = getSize();
  return size.x() == 512 && size.y() == 512;
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

