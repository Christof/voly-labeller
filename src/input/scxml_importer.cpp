#include "./scxml_importer.h"
#include <QXmlStreamReader>
#include <QFile>
#include <QStateMachine>
#include <QAbstractTransition>
#include <QState>
#include <QFinalState>
#include <QEvent>
#include <QKeySequence>
#include <QKeyEventTransition>
#include <iostream>
#include <cassert>
#include "./invoke.h"

// from
// http://stackoverflow.com/questions/14034209/convert-string-representation-of-keycode-to-qtkey-or-any-int-and-back
uint toKey(QString const &str)
{
  QKeySequence seq(str);
  uint keyCode;

  // We should only working with a single key here
  if (seq.count() == 1)
    keyCode = seq[0];
  else
  {
    // Should be here only if a modifier key (e.g. Ctrl, Alt) is pressed.
    assert(seq.count() == 0);

    // Add a non-modifier key "A" to the picture because QKeySequence
    // seems to need that to acknowledge the modifier. We know that A has
    // a keyCode of 65 (or 0x41 in hex)
    seq = QKeySequence(str + "+A");
    assert(seq.count() == 1);
    assert(seq[0] > 65);
    keyCode = seq[0] - 65;
  }

  return keyCode;
}

ScxmlImporter::ScxmlImporter(QUrl url, QObject *keyboardEventReceiver,
    std::shared_ptr<InvokeManager> invokeManager)
  : keyboardEventReceiver(keyboardEventReceiver), invokeManager(invokeManager)
{
  QFile file(url.toLocalFile());
  file.open(QFile::OpenModeFlag::ReadOnly);

  stateMachine = std::shared_ptr<QStateMachine>(new QStateMachine());
  connect(stateMachine.get(), &QStateMachine::finished, []
          {
    std::cout << "finished" << std::endl;
  });

  reader =
      std::unique_ptr<QXmlStreamReader>(new QXmlStreamReader(file.readAll()));

  while (!reader->atEnd() && !reader->hasError())
  {
    QXmlStreamReader::TokenType token = reader->readNext();
    if (token == QXmlStreamReader::StartDocument)
      continue;

    if (token == QXmlStreamReader::StartElement)
      readElement();

    if (token == QXmlStreamReader::EndElement)
      finishElement();
  }

  qRegisterMetaType<QAbstractState *>("QAbstractState*");
  for (auto &transitionTuple : transitions)
  {
    auto transition = std::get<0>(transitionTuple);
    auto targetStateName = std::get<1>(transitionTuple);
    std::cout
        << "from "
        << transition->sourceState()->property("name").toString().toStdString()
        << " to " << targetStateName.toStdString() << std::endl;
    transition->setTargetState(states[targetStateName]);
  }

  if (reader->hasError())
    std::cerr << "Error while parsing: " << reader->errorString().toStdString()
              << std::endl;
}

ScxmlImporter::~ScxmlImporter()
{
}

void ScxmlImporter::readElement()
{
  auto elementName = reader->name();
  activeElement = elementName.toString();
  if (elementName == "scxml")
    initialState = attributeAsString("initialstate");

  if (elementName == "state")
    readState();

  if (elementName == "transition")
    readTransition();

  if (elementName == "final")
    readFinalState();

  if (elementName == "invoke")
  {
    std::cout << "targettype: " << attributeAsString("targettype").toStdString()
              << std::endl;
    std::cout << "src: " << attributeAsString("src").toStdString() << std::endl;
    invokeManager->addFor(currentTransition, attributeAsString("targettype"),
                          attributeAsString("src"));
  }
}

void ScxmlImporter::finishElement()
{
  if (reader->name() != activeElement)
    std::cout << "not finishing active element" << std::endl;
}

void ScxmlImporter::readState()
{
  auto stateName = attributeAsString("id");
  state = new QState();
  state->setProperty("name", stateName);
  stateMachine->addState(state);

  if (stateName == initialState)
    stateMachine->setInitialState(state);

  auto currentState = state;
  connect(state, &QState::entered, [currentState]()
          {
    std::cout << "entered: "
              << currentState->property("name").toString().toStdString()
              << std::endl;
  });

  std::cout << "state: " << state->property("name").toString().toStdString()
            << std::endl;

  states[stateName] = state;
}

void ScxmlImporter::readFinalState()
{
  auto stateName = attributeAsString("id");
  auto finalState = new QFinalState();
  finalState->setProperty("name", stateName);
  stateMachine->addState(finalState);
  states[stateName] = finalState;

  auto currentState = finalState;
  connect(finalState, &QState::entered, [currentState]()
          {
    std::cout << "entered: "
              << currentState->property("name").toString().toStdString()
              << std::endl;
  });

  std::cout << "state (final): "
            << finalState->property("name").toString().toStdString()
            << std::endl;
}

void ScxmlImporter::readTransition()
{
  auto event = attributeAsString("event");
  auto target = attributeAsString("target");
  QString keyboardEvent = "KeyboardEvent";
  QAbstractTransition *transition = nullptr;
  if (event.startsWith(keyboardEvent))
  {
    QEvent::Type eventType = QEvent::KeyPress;
    auto keyAsString = event.mid(event.lastIndexOf(".") + 1);
    auto keyCode = toKey(keyAsString);
    transition = new QKeyEventTransition(keyboardEventReceiver, eventType,
                                         keyCode, state);
    std::cout << keyAsString.toStdString() << ": " << keyCode << "|"
              << " target: " << target.toStdString() << std::endl;

    transitions.push_back(std::make_tuple(transition, target));
  }
  std::cout << "transition: " << event.toStdString() << " | "
            << target.toStdString() << std::endl;

  currentTransition = transition;
}

QString ScxmlImporter::attributeAsString(const char *name)
{
  return reader->attributes().value(name).toString();
}

std::shared_ptr<QStateMachine> ScxmlImporter::getStateMachine()
{
  return stateMachine;
}

