#ifndef SRC_GRAPHICS_BUFFER_MANAGER_H_

#define SRC_GRAPHICS_BUFFER_MANAGER_H_

#include <Eigen/Core>
#include "./gl.h"
#include "./buffer_hole_manager.h"
#include "./attribute_buffer.h"

namespace Graphics
{

struct TexAddress
{
  GLuint64 ContainerHandle;
  GLfloat TexPage;
  GLint Reserved;
  GLfloat Texscale[2];
};

struct ObjectData
{
  int VertexOffset;
  int IndexOffset;
  int VertexSize;
  int IndexSize;

  TexAddress TextureAddress;

  Eigen::Matrix4f transform;
};

/**
 * \brief
 *
 *
 */
class BufferManager
{
 public:
  BufferManager();
  virtual ~BufferManager();
  void initialize(Gl *gl, uint maxobjects, uint buffersize);

  int addObject(const std::vector<float> &vertices,
                const std::vector<float> &normals,
                const std::vector<float> &colors,
                const std::vector<float> &texCoords,
                const std::vector<uint> &indices);
  bool removeObject(int objID);

 private:
  int objectCount = 0;
  GLuint vertexArrayId;
  Gl *gl;
  std::map<int, ObjectData> objects;

  AttributeBuffer positionBuffer;
  AttributeBuffer normalBuffer;
  AttributeBuffer colorBuffer;
  AttributeBuffer texCoordBuffer;
  AttributeBuffer drawIdBuffer;
  AttributeBuffer indexBuffer;

  BufferHoleManager vertexBufferManager;
  BufferHoleManager indexBufferManager;
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_BUFFER_MANAGER_H_
