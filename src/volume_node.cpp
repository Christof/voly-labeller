#include "./volume_node.h"
#include "./render_data.h"
#include "./volume_reader.h"
#include "./quad.h"

VolumeNode::VolumeNode(std::string filename) : filename(filename)
{
  volumeReader = std::unique_ptr<VolumeReader>(new VolumeReader(filename));
  quad = std::unique_ptr<Quad>(new Quad(":shader/label.vert", ":shader/slice.frag"));
}

VolumeNode::~VolumeNode()
{
}

void VolumeNode::render(Gl *gl, RenderData renderData)
{
  if (texture == 0)
    initializeTexture(gl);

  glAssert(gl->glActiveTexture(GL_TEXTURE0));
  glAssert(gl->glBindTexture(GL_TEXTURE_3D, texture));
  quad->render(gl, renderData);
}

void VolumeNode::initializeTexture(Gl *gl)
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

