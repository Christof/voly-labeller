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

  itk::ImageRegionIterator<ImageType> it(image, image->GetRequestedRegion());
  MinMaxCalculator::Pointer calculator = MinMaxCalculator::New();

  calculator->SetImage(image);
  calculator->Compute();

  max = calculator->GetMaximum();
  min = calculator->GetMinimum();

  normalizeToCT(it);

  std::cout << "min" << min << std::endl;
  std::cout << "max" << max << std::endl;
}

VolumeReader::~VolumeReader()
{
}

void VolumeReader::normalizeToCT(itk::ImageRegionIterator<ImageType> it)
{
  const float maxAllowedValue = 3071;
  const float minAllowedValue = -1024;

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    float value = (it.Get() > maxAllowedValue) ? maxAllowedValue : it.Get();
    value = (value < minAllowedValue) ? minAllowedValue : value;

    auto normalizedValue =
        (value - minAllowedValue) / (maxAllowedValue - minAllowedValue);
    it.Set(normalizedValue);
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

