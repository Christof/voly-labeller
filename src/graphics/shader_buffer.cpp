#include "./shader_buffer.h"
#include <QLoggingCategory>
#include <cassert>

namespace Graphics
{

QLoggingCategory sbChan("Graphics.ShaderBuffer");

ShaderBuffer::ShaderBuffer(GLenum target, bool runUpdatesOnCPU)
  : lockManager(runUpdatesOnCPU), target(target)
{
}

ShaderBuffer::~ShaderBuffer()
{
  qCInfo(sbChan) << "Destructor of ShaderBuffer";
  terminate();
}

bool ShaderBuffer::initialize(Gl *gl, GLuint count, GLbitfield createFlags,
                              GLbitfield mapFlags)
{
  this->gl = gl;
  this->count = count;
  this->lockManager.initialize(gl);
  this->head = 0;

  gl->glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &offsetAlignment);
  qCDebug(sbChan) << "Offset alignment" << offsetAlignment;

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
    qCWarning(sbChan) << "glMapBufferRange failed, probable bug.";
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
  qCDebug(sbChan) << this << "Reserve" << count << "bytes"
           << "head" << head << "this->count" << this->count;
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
  qCDebug(sbChan) << this << "onUsageComplete: count" << count << "aligendCount"
    << alignedCount;


  lockManager.lockRange(head, alignedCount);
  head += alignedCount;
}

void ShaderBuffer::bindBufferRange(GLuint index, GLsizeiptr count)
{
  qCDebug(sbChan) << this << "bindBufferRange: count" << count << "head"
    << head << "this->count" << this->count;
  assert(count <= this->count);
  assert(head + count <= this->count);

  glAssert(gl->glBindBufferRange(target, index, this->id, head, count));
}

}  // namespace Graphics
