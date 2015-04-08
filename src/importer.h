#ifndef SRC_IMPORTER_H_

#define SRC_IMPORTER_H_

#include <memory>
#include <map>
#include <assimp/Importer.hpp>
#include "./mesh.h"
#include "./gl.h"

/**
 * \brief
 *
 *
 */
class Importer
{
 public:
  Importer(Gl *gl);
  virtual ~Importer();

  std::shared_ptr<Mesh> import(std::string filename, int meshIndex);

 private:
  Gl *gl;
  Assimp::Importer importer;
  std::map<std::string, const aiScene *> scenes;

  const aiScene *readScene(std::string filename);
};

#endif  // SRC_IMPORTER_H_
