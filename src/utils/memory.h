#ifndef SRC_UTILS_MEMORY_H_

#define SRC_UTILS_MEMORY_H_

#include <memory>

namespace std
{

template <typename T, typename... Args>
unique_ptr<T> make_unique(Args &&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace std


#endif  // SRC_UTILS_MEMORY_H_
