#include "./buffer_lock_manager.h"
#include <cassert>
#include <vector>

namespace Graphics
{

BufferLockManager::BufferLockManager(bool runUpdatesOnCPU)
  : runUpdatesOnCPU(runUpdatesOnCPU)
{
}

BufferLockManager::~BufferLockManager()
{
  for (auto it = mBufferLocks.begin(); it != mBufferLocks.end(); ++it)
  {
    cleanup(&*it);
  }

  mBufferLocks.clear();
}

void BufferLockManager::initialize(Gl *gl)
{
  this->gl = gl;
}

void BufferLockManager::waitForLockedRange(size_t lockBeginBytes,
                                           size_t lockLength)
{
  BufferRange testRange = { lockBeginBytes, lockLength };
  std::vector<BufferLock> swapLocks;
  for (auto it = mBufferLocks.begin(); it != mBufferLocks.end(); ++it)
  {
    if (testRange.overlaps(it->range))
    {
      wait(&it->syncObject);
      cleanup(&*it);
    }
    else
    {
      swapLocks.push_back(*it);
    }
  }

  mBufferLocks.swap(swapLocks);
}

void BufferLockManager::lockRange(size_t lockBeginBytes, size_t lockLength)
{
  BufferRange newRange = { lockBeginBytes, lockLength };
  GLsync syncName = gl->glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  BufferLock newLock = { newRange, syncName };

  mBufferLocks.push_back(newLock);
}

void BufferLockManager::wait(GLsync *syncObj)
{
  if (runUpdatesOnCPU)
  {
    GLbitfield waitFlags = 0;
    GLuint64 waitDuration = 0;
    while (1)
    {
      GLenum waitRet = gl->glClientWaitSync(*syncObj, waitFlags, waitDuration);
      if (waitRet == GL_ALREADY_SIGNALED || waitRet == GL_CONDITION_SATISFIED)
      {
        return;
      }

      if (waitRet == GL_WAIT_FAILED)
      {
        assert(!"Not sure what to do here. Probably raise an exception or "
                "something.");
        return;
      }

      // After the first time, need to start flushing, and wait for a looong
      // time.
      waitFlags = GL_SYNC_FLUSH_COMMANDS_BIT;

      const GLuint64 oneSecondInNanoSeconds = 1000000000;
      waitDuration = oneSecondInNanoSeconds;
    }
  }
  else
  {
    glAssert(gl->glWaitSync(*syncObj, 0, GL_TIMEOUT_IGNORED));
  }
}

void BufferLockManager::cleanup(BufferLock *bufferLock)
{
  glAssert(gl->glDeleteSync(bufferLock->syncObject));
}

}  // namespace Graphics
