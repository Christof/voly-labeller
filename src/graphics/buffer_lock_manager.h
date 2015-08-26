#ifndef SRC_GRAPHICS_BUFFER_LOCK_MANAGER_H_

#define SRC_GRAPHICS_BUFFER_LOCK_MANAGER_H_

#include <vector>
#include "./gl.h"

namespace Graphics
{

/**
 * \brief Encapsulates a locked buffer range
 *
 * It stores the start offset in the buffer and the length of the range.
 * The end offset can be retrieved with #endOffset().
 *
 * The method #overlaps() checks if the range overlaps with the
 * given range.
 */
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

/**
 * \brief Lock on a buffer determined by \link BufferRange range \endlink and
 * a sync object
 */
struct BufferLock
{
  BufferRange range;
  GLsync syncObject;
};

/**
 * \brief Manages locks for a buffer
 *
 * Locks can be acquired with #lockRange() and #waitForLockedRange() waits until
 * the range is free again.
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

  std::vector<BufferLock> bufferLocks;
  bool runUpdatesOnCPU;
  Gl *gl;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_BUFFER_LOCK_MANAGER_H_
