#include "./volume_node.h"
#include <string>
#include "./graphics/render_data.h"
#include "./volume_reader.h"

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
  quad->render(gl, objectManager, textureManager, shaderManager, renderData);

  auto transformation = volumeReader->getTransformationMatrix();
  auto size = volumeReader->getPhysicalSize();
  Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
  scale.diagonal().head<3>() = size;
  renderData.modelMatrix = transformation * scale;
  cube->render(gl, objectManager, textureManager, shaderManager, renderData);
}

std::shared_ptr<Math::Obb> VolumeNode::getObb()
{
  return obb;
}

void VolumeNode::initializeTexture(Graphics::Gl *gl)
{
  auto size = volumeReader->getSize();
  auto textureTarget = GL_TEXTURE_3D;

  glAssert(gl->glGenTextures(1, &texture));
  glAssert(gl->glBindTexture(textureTarget, texture));
  glAssert(gl->glTexImage3D(textureTarget, 0, GL_R32F, size.x(), size.y(),
                            size.z(), 0, GL_RED, GL_FLOAT,
                            volumeReader->getDataPointer()));

  glTexParameterf(textureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameterf(textureTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(textureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(textureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexParameteri(textureTarget, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
  glAssert(gl->glBindTexture(textureTarget, 0));
}

