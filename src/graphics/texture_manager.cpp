#include "./texture_manager.h"
#include <vector>
#include <cassert>
#include <QImage>
#include <QLoggingCategory>
#include "./texture_container.h"
#include "./texture2d.h"
#include "./gl.h"

namespace Graphics
{

QLoggingCategory tmChan("Graphics.TextureManager");

TextureManager::~TextureManager()
{
  for (auto texture : textures)
    delete texture;

  textures.clear();

  shutdown();
}

int TextureManager::addTexture(std::string path)
{
  int id = textures.size();

  textures.push_back(newTexture2d(path));

  return id;
}

Texture2d *TextureManager::newTexture2d(int levels, int internalFormat,
                                        int width, int height)
{
  Texture2d *texture = allocateTexture2d(levels, internalFormat, width, height);
  texture->commit();

  return texture;
}

Texture2d *TextureManager::newTexture2d(std::string path)
{
  try
  {
    auto image = new QImage(path.c_str());

    auto internalformat = GL_RGBA8;
    auto format = GL_BGRA;
    auto type = GL_UNSIGNED_BYTE;

    Texture2d *texture =
        allocateTexture2d(1, internalformat, image->width(), image->height());
    texture->commit();

    texture->texSubImage2D(0, 0, 0, image->width(), image->height(), format,
                           type, image->bits());

    return texture;
  }
  catch (std::exception &error)
  {
    qCCritical(tmChan) << "Error loading texture '" << path.c_str()
                       << "': " << error.what();
    throw;
  }
}

bool TextureManager::initialize(Gl *gl, bool sparse, int maxTextureArrayLevels)
{
  this->gl = gl;
  this->maxTextureArrayLevels = maxTextureArrayLevels;
  this->isSparse = sparse;

  if (maxTextureArrayLevels > 0)
    return true;

  if (sparse)
  {
    glGetIntegerv(GL_MAX_SPARSE_ARRAY_TEXTURE_LAYERS_ARB,
                  &maxTextureArrayLevels);
  }
  else
  {
    glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &maxTextureArrayLevels);
  }

  return true;
}

void TextureManager::shutdown()
{
  for (auto containIt = textureContainers.begin();
       containIt != textureContainers.end(); ++containIt)
  {
    for (auto ptrIt = containIt->second.begin();
         ptrIt != containIt->second.end(); ++ptrIt)
    {
      {
        delete *ptrIt;
        *ptrIt = nullptr;
      }
    }
  }

  textureContainers.clear();
}

TextureAddress TextureManager::getAddressFor(int textureId)
{
  return textures[textureId]->address();
}

Texture2d *TextureManager::allocateTexture2d(int levels, int internalformat,
                                             int width, int height)
{
  TextureContainer *memArray = nullptr;

  // compute power of two texture size
  int twidth = 1;
  int theight = 1;
  while (twidth < width)
    twidth <<= 1;
  while (theight < height)
    theight <<= 1;

  qCDebug(tmChan) << "width/height:" << width << "/" << height << " -> "
                  << twidth << " " << theight;

  auto texType = std::make_tuple(levels, internalformat, twidth, theight);
  auto arrayIt = textureContainers.find(texType);
  if (arrayIt == textureContainers.end())
  {
    textureContainers[texType] = std::vector<TextureContainer *>();
    arrayIt = textureContainers.find(texType);
    assert(arrayIt != textureContainers.end());
  }

  for (auto it = arrayIt->second.begin(); it != arrayIt->second.end(); ++it)
  {
    if ((*it)->hasRoom())
    {
      memArray = (*it);
      break;
    }
  }

  if (memArray == nullptr)
  {
    memArray = new TextureContainer(gl, isSparse, levels, internalformat,
                                    twidth, theight, maxTextureArrayLevels);
    arrayIt->second.push_back(memArray);
  }

  assert(memArray);
  return new Texture2d(memArray, memArray->virtualAlloc());
}

}  // namespace Graphics
