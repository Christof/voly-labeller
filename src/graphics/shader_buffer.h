#ifndef SRC_GRAPHICS_SHADER_BUFFER_H_

#define SRC_GRAPHICS_SHADER_BUFFER_H_

#include "./gl.h"
#include "./buffer_lock_manager.h"

namespace Graphics
{

class ShaderBuffer
{
 public:
  explicit ShaderBuffer(GLenum target, bool runUpdatesOnCPU = true);

  virtual bool initialize(Gl *gl, GLuint count, GLbitfield createFlags,
                          GLbitfield mapFlags);

  virtual void terminate();

  void *reserve(GLsizeiptr count);

  void onUsageComplete(GLsizeiptr count);

  void bindBufferRange(GLuint index, GLsizeiptr count);

  void waitForLockedRange(size_t lockBegin, size_t lockLength);

  void *contents();

  void lockRange(size_t lockBegin, size_t lockLength);

  void bindBuffer();

  void bindBufferBase(GLuint index);

  void bindBufferRange(GLuint index, GLsizeiptr head, GLsizeiptr count);

  GLsizeiptr size() const;

  GLenum getTarget() const;

 protected:
  BufferLockManager lockManager;
  GLsizeiptr count = 0;
  Gl *gl;
  void *bufferContent;
  GLuint id = 0;
  GLenum target;
  GLsizeiptr head;
  int offsetAlignment;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_SHADER_BUFFER_H_
