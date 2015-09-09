#ifndef SRC_GRAPHICS_TRANSFER_FUNCTION_MANAGER_H_

#define SRC_GRAPHICS_TRANSFER_FUNCTION_MANAGER_H_

#include <string>
#include <memory>

namespace Graphics
{

class TextureManager;

/**
 * \brief Provides access to a large lookup table wherein each row represents a
 * transfer function
 *
 * New transfer functions can be added from a gradient file (.gra) using #add().
 */
class TransferFunctionManager
{
 public:
  TransferFunctionManager(std::shared_ptr<TextureManager> textureManager);

  /**
   * \brief Generates a transfer function lookup row from the given gradient file
   *
   * \return The row in the transfer function lookup table.
   */
  int add(std::string path);
  virtual ~TransferFunctionManager();

 private:
  const int width = 4096;
  std::shared_ptr<TextureManager> textureManager;
  int textureId = -1;
  int usedRows = 0;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TRANSFER_FUNCTION_MANAGER_H_
