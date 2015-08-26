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

  bool initialize(Gl *gl, GLuint count, GLbitfield createFlags,
                          GLbitfield mapFlags);

  void *reserve(GLsizeiptr count);

  void onUsageComplete(GLsizeiptr count);

  void bindBufferRange(GLuint index, GLsizeiptr count);

 private:
  BufferLockManager lockManager;
  GLsizeiptr count = 0;
  Gl *gl;
  void *bufferContent;
  GLuint id = 0;
  GLenum target;
  GLsizeiptr head;
  int offsetAlignment;

  void terminate();
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_SHADER_BUFFER_H_
