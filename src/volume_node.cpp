#include "./volume_node.h"
#include <string>
#include <vector>
#include <Eigen/Geometry>
#include "./graphics/render_data.h"
#include "./volume_reader.h"
#include "./graphics/object_manager.h"
#include "./graphics/shader_manager.h"
#include "./graphics/volume_manager.h"
#include "./utils/path_helper.h"

std::unique_ptr<Graphics::TransferFunctionManager>
    VolumeNode::transferFunctionManager;

VolumeNode::VolumeNode(std::string filename) : filename(filename)
{
  volumeReader = std::unique_ptr<VolumeReader>(new VolumeReader(filename));

  auto transformation = volumeReader->getTransformationMatrix();
  Eigen::Vector3f halfWidths = 0.5f * volumeReader->getPhysicalSize();
  Eigen::Vector3f center = transformation.col(3).head<3>();
  obb = Math::Obb(center, halfWidths, transformation.block<3, 3>(0, 0));

  volumeId = Graphics::VolumeManager::instance->addVolume(this);
}

VolumeNode::~VolumeNode()
{
}

void VolumeNode::render(Graphics::Gl *gl, RenderData renderData)
{
  if (cube.get() == nullptr)
  {
    initialize(gl);
    if (!transferFunctionManager.get())
      transferFunctionManager =
          std::unique_ptr<Graphics::TransferFunctionManager>(
              new Graphics::TransferFunctionManager(textureManager));

    transferFunctionRow =
        transferFunctionManager->add(absolutePathOfProjectRelativePath(
            std::string("assets/transferfunctions/scapula4.gra")));
  }

  glAssert(gl->glActiveTexture(GL_TEXTURE0));

  objectManager->renderLater(cubeData);
}

Graphics::VolumeData VolumeNode::getVolumeData()
{
  Graphics::VolumeData data;
  data.textureAddress = transferFunctionManager->getTextureAddress();
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

void VolumeNode::initialize(Graphics::Gl *gl)
{
  cube = std::unique_ptr<Graphics::Cube>(new Graphics::Cube());
  cube->initialize(gl, objectManager, textureManager, shaderManager);
  int shaderProgramId = shaderManager->addShader(
      ":/shader/cube.vert", ":/shader/cube.geom", ":/shader/volume_cube.frag");

  auto colors = std::vector<float>{ 1, 0, 0, 0.5f };
  auto pos = std::vector<float>{ 0, 0, 0 };
  cubeData = objectManager->addObject(
      pos, pos, colors, std::vector<float>{ 0, 0 }, std::vector<uint>{ 0 },
      shaderProgramId, GL_POINTS);
  auto transformation = volumeReader->getTransformationMatrix();
  cubeData.modelMatrix = transformation;
  float volumeIdAsFloat = *reinterpret_cast<float *>(&volumeId);
  Eigen::Vector4f physicalSize(
      volumeReader->getPhysicalSize().x(), volumeReader->getPhysicalSize().y(),
      volumeReader->getPhysicalSize().z(), volumeIdAsFloat);

  cubeData.setCustomBuffer(sizeof(Eigen::Vector4f),
                           [physicalSize](void *insertionPoint)
                           {
    memcpy(insertionPoint, physicalSize.data(), sizeof(Eigen::Vector4f));
  });

  glAssert(gl->glPointSize(40));
}

