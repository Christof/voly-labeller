#include "./volume_node.h"
#include <string>
#include <vector>
#include "./graphics/render_data.h"
#include "./volume_reader.h"
#include "./graphics/object_manager.h"
#include "./graphics/shader_manager.h"

VolumeNode::VolumeNode(std::string filename) : filename(filename)
{
  volumeReader = std::unique_ptr<VolumeReader>(new VolumeReader(filename));
  quad = std::unique_ptr<Graphics::Quad>(
      new Graphics::Quad(":shader/label.vert", ":shader/slice.frag"));
  cube = std::unique_ptr<Graphics::Cube>(new Graphics::Cube());

  auto transformation = volumeReader->getTransformationMatrix();
  Eigen::Vector3f halfWidths = 0.5f * volumeReader->getPhysicalSize();
  Eigen::Vector3f center = transformation.col(3).head<3>();
  obb = std::make_shared<Math::Obb>(center, halfWidths,
                                    transformation.block<3, 3>(0, 0));
}

VolumeNode::~VolumeNode()
{
}

void VolumeNode::render(Graphics::Gl *gl, RenderData renderData)
{
  if (texture == 0)
    initializeTexture(gl);

  glAssert(gl->glActiveTexture(GL_TEXTURE0));
  glAssert(gl->glBindTexture(GL_TEXTURE_3D, texture));
  // quad->render(gl, objectManager, textureManager, shaderManager, renderData);

  auto transformation = volumeReader->getTransformationMatrix();
  auto size = volumeReader->getPhysicalSize();
  Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
  scale.diagonal().head<3>() = size;
  cubeData.transform = transformation * scale;
  objectManager->renderLater(cubeData);
}

std::shared_ptr<Math::Obb> VolumeNode::getObb()
{
  return obb;
}

void VolumeNode::initializeTexture(Graphics::Gl *gl)
{
  cube->initialize(gl, objectManager, textureManager, shaderManager);
  int shaderProgramId = shaderManager->addShader(
      ":/shader/simple.vert", ":/shader/cube.geom", ":/shader/test.frag");

  auto colors = std::vector<float>{ 1, 0, 0, 0.5f };
  auto pos = std::vector<float>{ 0, 0, 0 };
  cubeData = objectManager->addObject(
      pos, pos, colors, std::vector<float>{ 0, 0 }, std::vector<uint>{ 0 },
      shaderProgramId, GL_POINTS);
  cubeData.transform = Eigen::Matrix4f::Identity();
  glAssert(gl->glPointSize(40));

  texture = textureManager->add3dTexture(volumeReader->getSize(),
                                         volumeReader->getDataPointer());
}

