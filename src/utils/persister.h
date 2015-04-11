#ifndef SRC_UTILS_PERSISTER_H_

#define SRC_UTILS_PERSISTER_H_

#include <fstream>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
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

  /*
  template <typename T> static std::shared_ptr<T> loadSpecial(std::string filename)
  {
    std::ifstream ifs(filename);
    boost::archive::xml_iarchive ia(ifs);
    T *result = static_cast<T*>(malloc(sizeof(T)));
    ia >> BOOST_SERIALIZATION_NVP(result);//boost::serialization::make_nvp<T>("result", (T&)*result);

    return std::shared_ptr<T>(result);
  }
  */

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

namespace boost
{
namespace serialization
{
template <class Archive>
void serialize(Archive &ar, Eigen::Matrix4f &matrix, const unsigned int version)
{
  ar & boost::serialization::make_nvp("r0c0", matrix(0, 0));
  ar & boost::serialization::make_nvp("r0c1", matrix(0, 1));
  ar & boost::serialization::make_nvp("r0c2", matrix(0, 2));
  ar & boost::serialization::make_nvp("r0c3", matrix(0, 3));

  ar & boost::serialization::make_nvp("r1c0", matrix(1, 0));
  ar & boost::serialization::make_nvp("r1c1", matrix(1, 1));
  ar & boost::serialization::make_nvp("r1c2", matrix(1, 2));
  ar & boost::serialization::make_nvp("r1c3", matrix(1, 3));

  ar & boost::serialization::make_nvp("r2c0", matrix(2, 0));
  ar & boost::serialization::make_nvp("r2c1", matrix(2, 1));
  ar & boost::serialization::make_nvp("r2c2", matrix(2, 2));
  ar & boost::serialization::make_nvp("r2c3", matrix(2, 3));

  ar & boost::serialization::make_nvp("r3c0", matrix(3, 0));
  ar & boost::serialization::make_nvp("r3c1", matrix(3, 1));
  ar & boost::serialization::make_nvp("r3c2", matrix(3, 2));
  ar & boost::serialization::make_nvp("r3c3", matrix(3, 3));
}
}  // namespace serialization
}  // namespace boost
#endif  // SRC_UTILS_PERSISTER_H_
