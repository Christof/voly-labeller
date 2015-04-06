#ifndef SRC_UTILS_PERSISTER_H_

#define SRC_UTILS_PERSISTER_H_

#include <fstream>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

/**
 * \brief
 *
 *
 */
class Persister
{
 public:
  Persister();
  virtual ~Persister();

  template <typename T> static void save(T &data, std::string filename)
  {
    std::ofstream ofs(filename);
    boost::archive::xml_oarchive oa(ofs);

    oa <<  BOOST_SERIALIZATION_NVP(data);
  }

  template <typename T> static T load(std::string filename)
  {
    std::ifstream ifs(filename);
    boost::archive::xml_iarchive ia(ifs);
    T result;
    ia >> BOOST_SERIALIZATION_NVP(result);

    return result;
  }

 private:
  /* data */
};

#endif  // SRC_UTILS_PERSISTER_H_
