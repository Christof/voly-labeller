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
  cube = std::unique_ptr<Graphics::Cube>(new Graphics::Cube());

  auto transformation = volumeReader->getTransformationMatrix();
  Eigen::Vector3f halfWidths = 0.5f * volumeReader->getPhysicalSize();
  Eigen::Vector3f center = transformation.col(3).head<3>();
  obb = std::make_shared<Math::Obb>(center, halfWidths,
                                    transformation.block<3, 3>(0, 0));

  Graphics::VolumeManager::instance->addVolume(this);
}

VolumeNode::~VolumeNode()
{
}

void VolumeNode::render(Graphics::Gl *gl, RenderData renderData)
{
  if (texture == 0)
  {
    initializeTexture(gl);
    if (!transferFunctionManager.get())
      transferFunctionManager =
          std::unique_ptr<Graphics::TransferFunctionManager>(
              new Graphics::TransferFunctionManager(textureManager));

    transferFunctionRow =
        transferFunctionManager->add(absolutePathOfProjectRelativePath(
            std::string("assets/transferfunctions/scapula4.gra")));
  }

  glAssert(gl->glActiveTexture(GL_TEXTURE0));
  glAssert(gl->glBindTexture(GL_TEXTURE_3D, texture));

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

std::shared_ptr<Math::Obb> VolumeNode::getObb()
{
  return obb;
}

void VolumeNode::initializeTexture(Graphics::Gl *gl)
{
  cube->initialize(gl, objectManager, textureManager, shaderManager);
  int shaderProgramId = shaderManager->addShader(
      ":/shader/cube.vert", ":/shader/cube.geom", ":/shader/test.frag");

  auto colors = std::vector<float>{ 1, 0, 0, 0.5f };
  auto pos = std::vector<float>{ 0, 0, 0 };
  cubeData = objectManager->addObject(
      pos, pos, colors, std::vector<float>{ 0, 0 }, std::vector<uint>{ 0 },
      shaderProgramId, GL_POINTS);
  auto transformation = volumeReader->getTransformationMatrix();
  cubeData.modelMatrix = transformation;
  auto physicalSize = volumeReader->getPhysicalSize();
  cubeData.setCustomBuffer(sizeof(Eigen::Vector3f),
                           [physicalSize](void *insertionPoint)
                           {
    memcpy(insertionPoint, physicalSize.data(), sizeof(Eigen::Vector3f));
  });

  glAssert(gl->glPointSize(40));

  texture = textureManager->add3dTexture(volumeReader->getSize(),
                                         volumeReader->getDataPointer());
}

