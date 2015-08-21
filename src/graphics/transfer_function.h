#ifndef SRC_GRAPHICS_TRANSFER_FUNCTION_H_

#define SRC_GRAPHICS_TRANSFER_FUNCTION_H_

#include <string>
#include <memory>

namespace Graphics
{

class TextureManager;

/**
 * \brief
 *
 *
 */
class TransferFunction
{
 public:
  TransferFunction(std::shared_ptr<TextureManager> textureManager,
                   std::string path);
  virtual ~TransferFunction();

 private:
  std::shared_ptr<TextureManager> textureManager;
  int texture;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TRANSFER_FUNCTION_H_
