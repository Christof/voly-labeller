#ifndef SRC_UTILS_PERSISTER_H_

#define SRC_UTILS_PERSISTER_H_

#include <fstream>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <Eigen/Core>

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

    oa << BOOST_SERIALIZATION_NVP(data);
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

namespace boost
{
namespace serialization
{
template <class Archive>
void serialize(Archive &ar, Eigen::Vector3f &vector, const unsigned int version)
{
  ar & boost::serialization::make_nvp("x", vector.x());
  ar & boost::serialization::make_nvp("y", vector.y());
  ar & boost::serialization::make_nvp("z", vector.z());
}
}  // namespace serialization
}  // namespace boost

#endif  // SRC_UTILS_PERSISTER_H_
