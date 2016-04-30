#ifndef SRC_GRAPHICS_CIRCULAR_BUFFER_H_

#define SRC_GRAPHICS_CIRCULAR_BUFFER_H_

#include "./gl.h"
#include "./mapped_buffer.h"

#include <cassert>

namespace Graphics
{

/**
 * \brief MappedBuffer with a circular locking/usage scheme
 *
 * The usage pattern is to first reserve a certain amount
 * of elements using #reserve() then writing the elements to
 * the given pointer. Afterwards #bindBufferRange() must be
 * called so that the buffer can be used in a shader. Finally
 * after using drawing something the #onUsageComplete must be
 * called.
 */
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
    gl->glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &offsetAlignment);

    return MappedBuffer<T>::initialize(gl, count, createFlags, mapFlags);
  }

  T *reserve(GLsizeiptr count)
  {
    assert(count <= this->size());

    if (head + count > this->size())
      head = 0;

    GLsizeiptr lockStart = head;

    this->waitForLockedRange(lockStart, count);
    return &this->contents()[lockStart];
  }

  void onUsageComplete(GLsizeiptr count)
  {
    int alignedCount = std::ceil(count / static_cast<float>(offsetAlignment)) *
                       offsetAlignment;
    this->lockRange(head, alignedCount);
    head += alignedCount;
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
  int offsetAlignment;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_CIRCULAR_BUFFER_H_
