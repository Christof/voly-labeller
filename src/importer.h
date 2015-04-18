#ifndef SRC_IMPORTER_H_

#define SRC_IMPORTER_H_

#include <assimp/Importer.hpp>
#include <memory>
#include <map>
#include <vector>
#include <string>

class Mesh;

/**
 * \brief Provides functions to import Mesh%es from an asset file
 */
class Importer
{
 public:
  Importer();
  virtual ~Importer();

  std::shared_ptr<Mesh> import(std::string filename, int meshIndex);
  std::vector<std::shared_ptr<Mesh>> importAll(std::string filename);

 private:
  Assimp::Importer importer;
  std::map<std::string, const aiScene *> scenes;

  const aiScene *readScene(std::string filename);
};

#endif  // SRC_IMPORTER_H_
