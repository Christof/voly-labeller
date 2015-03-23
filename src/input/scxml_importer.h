#ifndef SRC_INPUT_SCXML_IMPORTER_H_

#define SRC_INPUT_SCXML_IMPORTER_H_

#include <QUrl>
#include <QState>
#include <memory>

/**
 * \brief
 *
 *
 */
class State : public QState
{
 public:
  State(std::string name) : name(name)
  {
  }
  virtual ~State()
  {
  }

  std::string name;

 private:
  /* data */
};

/**
 * \brief
 *
 *
 */
class ScxmlImporter
{
 public:
  ScxmlImporter(QUrl url);
  virtual ~ScxmlImporter();

 private:
  std::shared_ptr<QStateMachine> stateMachine;
};

#endif  // SRC_INPUT_SCXML_IMPORTER_H_
