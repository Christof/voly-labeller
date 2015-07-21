#ifndef SRC_GRAPHICS_CIRCULAR_BUFFER_H_

#define SRC_GRAPHICS_CIRCULAR_BUFFER_H_

#include "./gl.h"
#include "./mapped_buffer.h"

namespace Graphics
{

template <typename T> class CircularBuffer : public MappedBuffer<T>
{
 public:
  explicit CircularBuffer(GLenum target, bool runUpdatesOnCPU = true)
    : MappedBuffer<T>(target, runUpdatesOnCPU)
  {
  }

  bool initialize(Gl *gl, GLuint count, GLbitfield createFlags,
                  GLbitfield mapFlags)
  {
    this->gl = gl;
    head = 0;
    return MappedBuffer<T>::initialize(gl, count, createFlags, mapFlags);
  }

  T *reserve(GLsizeiptr count)
  {
    assert(count <= this->size());

    GLsizeiptr lockStart = head;

    if (lockStart + count > this->size())
    {
      // wrap
      lockStart = 0;
    }

    this->waitForLockedRange(lockStart, count);
    return &this->contents()[lockStart];
  }

  void onUsageComplete(GLsizeiptr count)
  {
    this->lockRange(head, count);
    head = (head + count) % this->size();
  }

  void bindBufferRange(GLuint index, GLsizeiptr count)
  {
    assert(count <= this->count);
    assert(head + count <= this->count);

    glAssert(gl->glBindBufferRange(this->getTarget(), index, this->id,
                                   head * sizeof(T), count * sizeof(T)));
  }

  GLsizeiptr getHead() const
  {
    return head;
  }

  void *headOffset() const
  {
    return reinterpret_cast<void *>(head * sizeof(T));
  }

 private:
  Gl *gl;
  GLsizeiptr head;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_CIRCULAR_BUFFER_H_
