#include "./attribute_buffer.h"

namespace Graphics
{

AttributeBuffer::AttributeBuffer(int componentCount, int componentSize,
                                 GLenum componentType)
  : bufferType(DEFAULT_BUFFER_TYPE), componentCount(componentCount),
    componentSize(componentSize), componentType(componentType)
{
}

AttributeBuffer::~AttributeBuffer()
{
  if (id == 0)
    return;

  glAssert(gl->glBindBuffer(bufferType, 0));

  glAssert(gl->glDeleteBuffers(1, &id));

  id = 0;
  count = 0;
}

int AttributeBuffer::elementSize() const
{
  return componentCount * componentSize;
}

bool AttributeBuffer::isInitialized() const
{
  return (id > 0);
};

GLuint AttributeBuffer::getId() const
{
  return id;
}

int AttributeBuffer::getComponentCount() const
{
  return componentCount;
}

uint AttributeBuffer::sizeBytes() const
{
  return (count * elementSize());
}

void AttributeBuffer::initialize(Gl *gl, uint size, GLuint type)
{
  this->gl = gl;
  assert(id == 0);

  count = size;
  bufferType = type;

  glAssert(gl->glGenBuffers(1, &id));
  glAssert(gl->glBindBuffer(bufferType, id));
  glAssert(gl->glBufferData(bufferType, sizeBytes(), NULL, GL_DYNAMIC_DRAW));

  glAssert(gl->glBindBuffer(bufferType, 0));
}

void AttributeBuffer::bindAttrib(int attribnum)
{
  assert(id > 0);

  glAssert(gl->glBindBuffer(bufferType, id));
  glAssert(gl->glVertexAttribPointer(attribnum, componentCount, componentType,
                                     GL_FALSE, 0, NULL));
  glAssert(gl->glEnableVertexAttribArray(attribnum));
}

void AttributeBuffer::bindAttribDivisor(int attribnum, int divisor)
{
  assert(id > 0);

  glAssert(gl->glBindBuffer(bufferType, id));
  glAssert(gl->glVertexAttribIPointer(attribnum, componentCount,
                                      componentType, componentSize, nullptr));
  glAssert(gl->glVertexAttribDivisor(attribnum, divisor));
  glAssert(gl->glEnableVertexAttribArray(attribnum));
}

void AttributeBuffer::bind()
{
  assert(id > 0);

  glAssert(gl->glBindBuffer(bufferType, id));
}

void AttributeBuffer::unbind()
{
  glAssert(gl->glBindBuffer(bufferType, 0));
}

}  // namespace Graphics
