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

  gl->glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &offsetAlignment);
  qDebug() << "Offset alignment" << offsetAlignment;

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
  qDebug() << "Reserve" << count << "bytes"
           << "head" << head;
  assert(count <= this->count);

  if (head + count > this->count)
    head = 0;

  GLsizeiptr lockStart = head;

  lockManager.waitForLockedRange(lockStart, count);
  return static_cast<char *>(bufferContent) + head;
}

void ShaderBuffer::onUsageComplete(GLsizeiptr count)
{
  int alignedCount =
      std::ceil(count / static_cast<float>(offsetAlignment)) * offsetAlignment;

  lockManager.lockRange(head, alignedCount);
  head += alignedCount;
}

void ShaderBuffer::bindBufferRange(GLuint index, GLsizeiptr count)
{
  assert(count <= this->count);
  assert(head + count <= this->count);

  glAssert(gl->glBindBufferRange(target, index, this->id, head, count));
}

}  // namespace Graphics
