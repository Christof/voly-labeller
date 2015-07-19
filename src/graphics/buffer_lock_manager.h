#ifndef SRC_GRAPHICS_BUFFER_LOCK_MANAGER_H_

#define SRC_GRAPHICS_BUFFER_LOCK_MANAGER_H_

#include <vector>
#include "./gl.h"

namespace Graphics
{

struct BufferRange
{
  size_t startOffset;
  size_t length;

  size_t endOffset() const
  {
    return startOffset + length;
  }

  bool overlaps(const BufferRange &other) const
  {
    return startOffset < other.endOffset() && other.startOffset < endOffset();
  }
};

struct BufferLock
{
  BufferRange range;
  GLsync syncObject;
};

/**
 * \brief
 *
 *
 */
class BufferLockManager
{
 public:
  explicit BufferLockManager(bool runUpdatesOnCPU);
  ~BufferLockManager();

  void initialize(Gl *gl);

  void waitForLockedRange(size_t lockBeginBytes, size_t lockLength);
  void lockRange(size_t lockBeginBytes, size_t lockLength);

 private:
  void wait(GLsync *syncObject);
  void cleanup(BufferLock *bufferLock);

  std::vector<BufferLock> mBufferLocks;

  bool runUpdatesOnCPU;

  Gl *gl;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_BUFFER_LOCK_MANAGER_H_
