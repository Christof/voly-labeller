#ifndef SRC_GRAPHICS_MAPPED_BUFFER_H_

#define SRC_GRAPHICS_MAPPED_BUFFER_H_

#include "./gl.h"
#include "./buffer_lock_manager.h"

namespace Graphics
{

/**
 * \brief Shader buffer mapped to a given number of elements of type \T
 */
template <typename T> class MappedBuffer
{
 public:
  MappedBuffer(GLenum target, bool runUpdatesOnCPU)
    : lockManager(runUpdatesOnCPU), target(target)
  {
  }

  virtual bool initialize(Gl *gl, GLuint count, GLbitfield createFlags,
                          GLbitfield mapFlags)
  {
    this->gl = gl;
    this->count = count;
    this->lockManager.initialize(gl);

    if (bufferContent)
      terminate();

    glAssert(gl->glGenBuffers(1, &id));
    glAssert(gl->glBindBuffer(target, id));

    // reserve GPU memory
    glAssert(
        gl->glBufferStorage(target, sizeof(T) * count, nullptr, createFlags));
    bufferContent = reinterpret_cast<T *>(
        gl->glMapBufferRange(target, 0, sizeof(T) * count, mapFlags));
    glCheckError();

    if (!bufferContent)
    {
      qWarning() << "glMapBufferRange failed, probable bug.";
      return false;
    }

    return true;
  }

  virtual void terminate()
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

  void waitForLockedRange(size_t lockBegin, size_t lockLength)
  {
    lockManager.waitForLockedRange(lockBegin, lockLength);
  }

  T *contents()
  {
    return bufferContent;
  }

  void lockRange(size_t lockBegin, size_t lockLength)
  {
    lockManager.lockRange(lockBegin, lockLength);
  }

  void bindBuffer()
  {
    glAssert(gl->glBindBuffer(target, id));
  }

  void bindBufferBase(GLuint index)
  {
    glAssert(gl->glBindBufferBase(target, index, id));
  }

  void bindBufferRange(GLuint index, GLsizeiptr head, GLsizeiptr count)
  {
    glAssert(gl->glBindBufferRange(target, index, id, head * sizeof(T),
                                   count * sizeof(T)));
  }

  GLsizeiptr size() const
  {
    return count;
  }

  GLenum getTarget() const
  {
    return target;
  }

 protected:
  BufferLockManager lockManager;
  GLsizeiptr count = 0;
  Gl *gl;
  T *bufferContent;
  GLuint id = 0;
  GLenum target;
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_MAPPED_BUFFER_H_
