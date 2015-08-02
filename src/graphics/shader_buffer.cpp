#include "./shader_buffer.h"
#include <cassert>

namespace Graphics
{

ShaderBuffer::ShaderBuffer(GLenum target, bool runUpdatesOnCPU)
  : lockManager(runUpdatesOnCPU), target(target)
{
}

bool ShaderBuffer::initialize(Gl *gl, GLuint count, GLbitfield createFlags,
                              GLbitfield mapFlags)
{
  this->gl = gl;
  this->count = count;
  this->lockManager.initialize(gl);
  this->head = 0;

  if (bufferContent)
    terminate();

  glAssert(gl->glGenBuffers(1, &id));
  glAssert(gl->glBindBuffer(target, id));

  // reserve GPU memory
  glAssert(gl->glBufferStorage(target, count, nullptr, createFlags));
  bufferContent = gl->glMapBufferRange(target, 0, count, mapFlags);
  glCheckError();

  if (!bufferContent)
  {
    qWarning() << "glMapBufferRange failed, probable bug.";
    return false;
  }

  return true;
}

void ShaderBuffer::terminate()
{
  if (id != 0)
  {
    glAssert(gl->glBindBuffer(target, id));
    glAssert(gl->glUnmapBuffer(target));
    glAssert(gl->glDeleteBuffers(1, &id));

    bufferContent = nullptr;
    count = 0;
    id = 0;
  }
}

void *ShaderBuffer::reserve(GLsizeiptr count)
{
  qWarning() << "Reserve" << count << "bytes" << "head" << head;
  assert(count <= this->size());

  if (head + count > this->size())
    head = 0;

  GLsizeiptr lockStart = head;

  this->waitForLockedRange(lockStart, count);
  return static_cast<char*>(bufferContent) + head;
}

void ShaderBuffer::onUsageComplete(GLsizeiptr count)
{
  this->lockRange(head, count);
  head += count;
}

void ShaderBuffer::bindBufferRange(GLuint index, GLsizeiptr count)
{
  assert(count <= this->count);
  assert(head + count <= this->count);

  glAssert(
      gl->glBindBufferRange(this->getTarget(), index, this->id, head, count));
}

void ShaderBuffer::waitForLockedRange(size_t lockBegin, size_t lockLength)
{
  lockManager.waitForLockedRange(lockBegin, lockLength);
}

void *ShaderBuffer::contents()
{
  return bufferContent;
}

void ShaderBuffer::lockRange(size_t lockBegin, size_t lockLength)
{
  lockManager.lockRange(lockBegin, lockLength);
}

void ShaderBuffer::bindBuffer()
{
  glAssert(gl->glBindBuffer(target, id));
}

void ShaderBuffer::bindBufferBase(GLuint index)
{
  glAssert(gl->glBindBufferBase(target, index, id));
}

void ShaderBuffer::bindBufferRange(GLuint index, GLsizeiptr head,
                                   GLsizeiptr count)
{
  glAssert(gl->glBindBufferRange(target, index, id, head, count));
}

GLsizeiptr ShaderBuffer::size() const
{
  return count;
}

GLenum ShaderBuffer::getTarget() const
{
  return target;
}

}  // namespace Graphics
