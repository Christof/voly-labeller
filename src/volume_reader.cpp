#include "./volume_reader.h"
#include <string>
#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>
#include <itkMinimumMaximumImageCalculator.h>
#include "./utils/path_helper.h"

typedef itk::Image< float, 3 > ImageType;
typedef itk::ImageFileReader<ImageType> VolumeReaderType;
typedef itk::MinimumMaximumImageCalculator< ImageType > MinMaxCalculator;

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
  MinMaxCalculator::Pointer calculator= MinMaxCalculator::New();

  calculator->SetImage(image);
  calculator->Compute();

  float min = calculator->GetMaximum();
  float max = calculator->GetMinimum();

  std::cout << "min" << min << std::endl;
  std::cout << "max" << max << std::endl;
}

VolumeReader::~VolumeReader()
{
}
