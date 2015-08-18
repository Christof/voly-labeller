#ifndef SRC_GRAPHICS_OBJECT_DATA_H_

#define SRC_GRAPHICS_OBJECT_DATA_H_

#include <Eigen/Core>
#include <functional>

namespace Graphics
{

/**
 * \brief Stores informations to render an object
 *
 * It stores the vertex and index information, the primitive type as well es the
 * used shader program id. The transform matrix is exposed to position the
 * object in world space.
 *
 * Additionally provide a way to set the custom buffer via a callback.
 */
struct ObjectData
{
  ObjectData(int id, int vertexOffset, int indexOffset, int indexSize,
             int shaderProgramId, int primitiveType);

  ObjectData();

  int getId() const;
  int getPrimitiveType() const;
  int getVertexOffset() const;
  int getIndexOffset() const;
  int getIndexSize() const;

  int getShaderProgramId() const;
  int getCustomBufferSize() const;

  bool hasCustomBuffer() const;

  Eigen::Matrix4f modelMatrix;

  bool isInitialized();

  void setCustomBuffer(int size, std::function<void(void *)> setFunction);

  void fillBufferElement(void *bufferStart, int index);

 private:
  int id;
  int primitiveType;
  int vertexOffset;
  int indexOffset;
  int indexSize;
  int shaderProgramId;

  int customBufferSize;
  std::function<void(void *)> setBuffer;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_OBJECT_DATA_H_
