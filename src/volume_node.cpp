#if _WIN32
#pragma warning(disable : 4522)
#endif

#include "./volume_node.h"
#include <Eigen/Geometry>
#include <string>
#include <vector>
#include "./graphics/render_data.h"
#include "./volume_reader.h"
#include "./graphics/object_manager.h"
#include "./graphics/shader_manager.h"
#include "./graphics/volume_manager.h"
#include "./graphics/transfer_function_manager.h"
#include "./utils/path_helper.h"
#include "./eigen_qdebug.h"

VolumeNode::VolumeNode(std::string volumePath, std::string transferFunctionPath,
                       Eigen::Matrix4f transformation,
                       bool ignoreTransformationFromFile)
  : volumePath(volumePath), transferFunctionPath(transferFunctionPath),
    transformation(transformation)
{
  volumeReader = std::make_unique<VolumeReader>(volumePath);

  Eigen::Matrix4f transformationFromFile =
      ignoreTransformationFromFile ? Eigen::Matrix4f::Identity()
                                   : volumeReader->getTransformationMatrix();
  Eigen::Vector3f halfWidths = 0.5f * volumeReader->getPhysicalSize();
  Eigen::Vector3f center = transformationFromFile.col(3).head<3>();
  obb =
      Math::Obb(center, halfWidths, transformationFromFile.block<3, 3>(0, 0)) *
      transformation;
  overallTransformation = transformation * transformationFromFile;
}

VolumeNode::~VolumeNode()
{
}

void VolumeNode::render(Graphics::Gl *gl,
                        std::shared_ptr<Graphics::Managers> managers,
                        RenderData renderData)
{
  if (cube.get() == nullptr)
    initialize(gl, managers);

  glAssert(gl->glActiveTexture(GL_TEXTURE0));

  managers->getObjectManager()->renderLater(cubeData);
}

Graphics::VolumeData VolumeNode::getVolumeData()
{
  Graphics::VolumeData data;
  data.textureAddress = transferFunctionAddress;
  Eigen::Affine3f positionToTexture(Eigen::Translation3f(0.5f, 0.5f, 0.5f));
  Eigen::Affine3f scale(Eigen::Scaling(volumeReader->getPhysicalSize()));
  data.textureMatrix = positionToTexture.matrix() *
                       (cubeData.modelMatrix * scale.matrix()).inverse();
  data.volumeId = volumeId;
  data.objectToDatasetMatrix = Eigen::Matrix4f::Identity();
  data.transferFunctionRow = transferFunctionRow;

  return data;
}

float *VolumeNode::getData()
{
  return volumeReader->getDataPointer();
}

Eigen::Vector3i VolumeNode::getDataSize()
{
  return volumeReader->getSize();
}

void VolumeNode::initialize(Graphics::Gl *gl,
                            std::shared_ptr<Graphics::Managers> managers)
{
  managers->getVolumeManager()->addVolume(this, gl);

  cube = std::unique_ptr<Graphics::Cube>(new Graphics::Cube());
  cube->initialize(gl, managers);
  int shaderProgramId = managers->getShaderManager()->addShader(
      ":/shader/cube.vert", ":/shader/cube.geom", ":/shader/volume_cube.frag");

  auto colors = std::vector<float>{ 1, 0, 0, 0.5f };
  auto pos = std::vector<float>{ 0, 0, 0 };
  cubeData = managers->getObjectManager()->addObject(
      pos, pos, colors, std::vector<float>{ 0, 0 }, std::vector<uint>{ 0 },
      shaderProgramId, GL_POINTS);
  cubeData.modelMatrix = overallTransformation;
  float volumeIdAsFloat = *reinterpret_cast<float *>(&volumeId);
  physicalSize = Eigen::Vector4f(
      volumeReader->getPhysicalSize().x(), volumeReader->getPhysicalSize().y(),
      volumeReader->getPhysicalSize().z(), volumeIdAsFloat);

  cubeData.setCustomBufferFor(1, &physicalSize);

  auto transferFunctionManager = managers->getTransferFunctionManager();
  transferFunctionRow = transferFunctionManager->add(
      absolutePathOfProjectRelativePath(transferFunctionPath));
  transferFunctionAddress = transferFunctionManager->getTextureAddress();
}

