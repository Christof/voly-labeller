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
  ImageType::Pointer image = reader->GetOutput();

  /*
  itk::Point<float,3> origin = image->GetOrigin();
  ImageType::DirectionType direction = image->GetDirection();
  ImageType::RegionType region = image->GetLargestPossibleRegion();
  ImageType::SizeType size = region.GetSize();
  */

  itk::ImageRegionIterator<ImageType> imageIterator(image, image->GetRequestedRegion());
  MinMaxCalculator::Pointer calculator = MinMaxCalculator::New();

  calculator->SetImage(image);
  calculator->Compute();

  max = calculator->GetMaximum();
  min = calculator->GetMinimum();

  // normalizeToCT(imageIterator);
  normalizeTo01(imageIterator);

  std::cout << "min" << min << std::endl;
  std::cout << "max" << max << std::endl;
}

VolumeReader::~VolumeReader()
{
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

