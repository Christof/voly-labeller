#include "./buffer.h"
#include <cassert>

namespace Graphics
{

Buffer::Buffer() : id(0), gpuPointer(0), size(0)
{
}

Buffer::Buffer(Gl *gl, uint size) : id(0), gpuPointer(0), size(0)
{
  initialize(gl, size);
}

Buffer::Buffer(Buffer const &buffer)
{
  initialize(buffer.gl, buffer.getSize());
  copy(buffer);
}

GLuint Buffer::getId() const
{
  return id;
}

GLuint Buffer::getType() const
{
  return bufferType;
}

uint Buffer::getSize() const
{
  return size;
}

GLuint64 Buffer::getGpuPointer() const
{
  assert(id != 0);
  return gpuPointer;
}


void Buffer::adopt(Buffer const &buffer)
{
  id = buffer.id;
  gpuPointer = buffer.gpuPointer;
  size = buffer.size;
}

void Buffer::forget()
{
  id = 0;
  gpuPointer = 0;
  size = 0;
}

void Buffer::copy(Buffer const &buffer)
{
  if (buffer.getSize() != getSize())
  {
    terminate();
    initialize(buffer.gl, buffer.getSize());
  }

  if (getSize() != 0)
  {
    glAssert(gl->getDirectStateAccess()->glNamedCopyBufferSubDataEXT(
        buffer.id, id, 0, 0, buffer.getSize()));
  }
}

void Buffer::initialize(Gl *gl, uint size)
{
  this->gl = gl;

  assert(id == 0);

  this->size = size;
  glAssert(gl->glGenBuffers(1, &id));
  glAssert(gl->glBindBuffer(bufferType, id));
  glAssert(gl->glBufferData(bufferType, size, NULL, GL_DYNAMIC_DRAW));
  auto extension = gl->getShaderBufferLoad();
  glAssert(extension->glGetBufferParameterui64vNV(
      bufferType, GL_BUFFER_GPU_ADDRESS_NV, &gpuPointer));
  glAssert(extension->glMakeBufferResidentNV(bufferType, GL_READ_WRITE));

  glAssert(gl->glBindBuffer(bufferType, 0));
}

void Buffer::resize(uint size)
{
  this->size = size;

  glAssert(gl->glBindBuffer(bufferType, id));
  glAssert(gl->glBufferData(bufferType, size, NULL, GL_DYNAMIC_DRAW));

  glAssert(gl->getShaderBufferLoad()->glGetBufferParameterui64vNV(
      bufferType, GL_BUFFER_GPU_ADDRESS_NV, &gpuPointer));
  glAssert(gl->getShaderBufferLoad()->glMakeBufferResidentNV(bufferType,
                                                             GL_READ_WRITE));

  glAssert(gl->glBindBuffer(bufferType, 0));
}

void Buffer::terminate()
{
  if (id == 0)
    return;

  glAssert(gl->glBindBuffer(bufferType, 0));
  glAssert(gl->getShaderBufferLoad()->glMakeBufferNonResidentNV(bufferType));
  glAssert(gl->glDeleteBuffers(1, &id));

  id = 0;
  gpuPointer = 0;
  size = 0;
}

void Buffer::clear(uint value)
{
  glAssert(gl->getDirectStateAccess()->glClearNamedBufferDataEXT(
      id, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &value));
}

void Buffer::clear(int value)
{
  glAssert(gl->getDirectStateAccess()->glClearNamedBufferDataEXT(
      id, GL_R32I, GL_RED_INTEGER, GL_INT, &value));
}

void Buffer::clear(float value)
{
  glAssert(gl->getDirectStateAccess()->glClearNamedBufferDataEXT(
      id, GL_R32F, GL_RED, GL_FLOAT, &value));
}

void Buffer::setData(const void *raw, uint byteCount, uint offset)
{
  assert(id != 0);
  assert((offset + byteCount) == size);

  glAssert(gl->getDirectStateAccess()->glNamedBufferSubDataEXT(id, offset,
                                                               byteCount, raw));
}

void Buffer::getData(void *raw, uint byteCount, uint offset)
{
  assert(id != 0);
  assert(offset + byteCount <= size);

  glAssert(gl->getDirectStateAccess()->glGetNamedBufferSubDataEXT(id, offset,
                                                                  byteCount, raw));
}

Buffer::~Buffer()
{
  terminate();
}
}  // namespace Graphics
