#ifndef SRC_IMPORTER_H_

#define SRC_IMPORTER_H_

#include <memory>
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
};

#endif  // SRC_IMPORTER_H_
