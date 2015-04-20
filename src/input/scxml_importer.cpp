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
#include <QMouseEventTransition>
#include <QSignalTransition>
#include <QLoggingCategory>
#include <cassert>
#include "./invoke.h"

QLoggingCategory channel("Input::ScxmlImporter");

bool equalIgnoreCase(QString const &a, QString const &b)
{
  return QString::compare(a, b, Qt::CaseSensitivity::CaseInsensitive) == 0;
}

// from
// http://stackoverflow.com/questions/14034209/convert-string-representation-of-keycode-to-qtkey-or-any-int-and-back
Qt::Key toKey(QString const &str)
{
  QKeySequence seq(str);

  if (equalIgnoreCase(str, "ctrl"))
    return Qt::Key_Control;
  if (equalIgnoreCase(str, "space"))
    return Qt::Key_Space;
  if (equalIgnoreCase(str, "alt"))
    return Qt::Key_Alt;
  if (equalIgnoreCase(str, "up_arrow"))
    return Qt::Key_Up;
  if (equalIgnoreCase(str, "down_arrow"))
    return Qt::Key_Down;
  if (equalIgnoreCase(str, "left_arrow"))
    return Qt::Key_Left;
  if (equalIgnoreCase(str, "right_arrow"))
    return Qt::Key_Right;
  if (equalIgnoreCase(str, "shift"))
    return Qt::Key_Shift;
  if (equalIgnoreCase(str, "esc"))
    return Qt::Key_Escape;
  if (equalIgnoreCase(str, "delete"))
    return Qt::Key_Delete;
  if (equalIgnoreCase(str, "backspace"))
    return Qt::Key_Backspace;

  // We should only working with a single key here
  assert(seq.count() == 1);
  return static_cast<Qt::Key>(seq[0]);
}

Qt::MouseButton toButton(const QString &str)
{
  if (equalIgnoreCase(str, "left"))
    return Qt::MouseButton::LeftButton;
  if (equalIgnoreCase(str, "right"))
    return Qt::MouseButton::RightButton;
  if (equalIgnoreCase(str, "middle"))
    return Qt::MouseButton::MiddleButton;
  if (equalIgnoreCase(str, "button4"))
    return Qt::MouseButton::ExtraButton4;
  if (equalIgnoreCase(str, "button5"))
    return Qt::MouseButton::ExtraButton5;

  return Qt::MouseButton::NoButton;
}

ScxmlImporter::ScxmlImporter(QUrl url,
                             std::shared_ptr<InvokeManager> invokeManager,
                             std::shared_ptr<SignalManager> signalManager)
  : url(url), invokeManager(invokeManager), signalManager(signalManager)
{
  const QMetaObject &mo = ScxmlImporter::staticMetaObject;
  int index = mo.indexOfEnumerator("ScxmlElement");
  metaScxmlElement = mo.enumerator(index);
}

ScxmlImporter::~ScxmlImporter()
{
}

void ScxmlImporter::readElement()
{
  auto elementName = reader->name().toString();
  elementStack.push(elementFromString(elementName));
  if (elementName == "scxml")
    initialState = attributeAsString("initialstate");

  if (elementName == "state")
    readState();

  if (elementName == "transition")
    readTransition();

  if (elementName == "final")
    readFinalState();

  if (elementName == "invoke")
    readInvoke();
}

void ScxmlImporter::finishElement()
{
  auto elementName = reader->name().toString();
  auto element = elementFromString(elementName);
  auto expectedElement = elementStack.pop();

  if (element != expectedElement)
    qCWarning(channel) << "not finishing active element. Expected:"
                       << metaScxmlElement.valueToKey(expectedElement)
                       << "but was:" << metaScxmlElement.valueToKey(element);

  if (elementName == "state")
    stateStack.pop();
}

void ScxmlImporter::readState()
{
  auto stateName = attributeAsString("id");
  auto state = new QState(stateStack.empty() ? 0 : stateStack.top());
  state->setProperty("name", stateName);
  if (stateStack.empty())
    stateMachine->addState(state);

  if (stateName == initialState)
    stateMachine->setInitialState(state);

  connect(state, &QState::entered, [state]()
          {
    qCDebug(channel) << "entered:" << state->property("name").toString();
  });

  stateStack.push(state);
  states[stateName] = state;
}

void ScxmlImporter::readFinalState()
{
  auto stateName = attributeAsString("id");
  auto finalState = new QFinalState();
  finalState->setProperty("name", stateName);
  stateMachine->addState(finalState);
  states[stateName] = finalState;
}

void ScxmlImporter::readTransition()
{
  auto event = attributeAsString("event");
  auto target = attributeAsString("target");
  if (event.startsWith("KeyboardEvent"))
  {
    currentTransition = createKeyEventTransition(event);
    transitions.push_back(std::make_tuple(currentTransition, target));
  }
  else if (event.startsWith("MouseButtonEvent"))
  {
    qCDebug(channel) << "in MouseButtonEvent" << event;
    currentTransition = createMouseButtonEventTransition(event);
    transitions.push_back(std::make_tuple(currentTransition, target));
  }
  else if (event.startsWith("MouseMoveEvent"))
  {
    currentTransition = createMouseMoveEventTransition();
    transitions.push_back(std::make_tuple(currentTransition, target));
  }
  else if (event.isEmpty() &&
           elementStack[elementStack.size() - 2] == ScxmlElement::initial)
  {
    currentTransition = new QSignalTransition(
        stateStack.top(), SIGNAL(entered()), stateStack.top());
    // just add target as initial state and don't set it for the transition,
    // because otherwise it would result in an infinite loop.
    initialStateTransitions[stateStack.top()] = target;
  }
  else
  {
    currentTransition = createSignalTransition(event);
    transitions.push_back(std::make_tuple(currentTransition, target));
  }
}

void ScxmlImporter::readInvoke()
{
  auto targetType = attributeAsString("targettype");
  auto source = attributeAsString("src");

  if (elementStack[elementStack.size() - 2] == ScxmlElement::onentry)
  {
    invokeManager->addForSignal(stateStack.top(), &QState::entered, targetType,
                                source);
    return;
  }

  if (elementStack[elementStack.size() - 2] == ScxmlElement::onexit)
  {
    invokeManager->addForSignal(stateStack.top(), &QState::exited, targetType,
                                source);
    return;
  }

  invokeManager->addFor(currentTransition, targetType, source);
}

std::shared_ptr<QStateMachine> ScxmlImporter::import()
{
  stateMachine = std::shared_ptr<QStateMachine>(new QStateMachine());

  parse();
  addTransitions();
  setInitialStates();

  return stateMachine;
}

void ScxmlImporter::parse()
{
  QFile file(url.toLocalFile());
  file.open(QFile::OpenModeFlag::ReadOnly);
  qCDebug(channel) << "Import scxml" << url;

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

  if (reader->hasError())
    qCCritical(channel) << "Error while parsing:" << reader->errorString();
}

void ScxmlImporter::addTransitions()
{
  qRegisterMetaType<QAbstractState *>("QAbstractState*");
  for (auto &transitionTuple : transitions)
  {
    auto transition = std::get<0>(transitionTuple);
    auto targetStateName = std::get<1>(transitionTuple);
    transition->setTargetState(states[targetStateName]);
  }
}

void ScxmlImporter::setInitialStates()
{
  for (auto &initialPair : initialStateTransitions)
    initialPair.first->setInitialState(states[initialPair.second]);
}

QAbstractTransition *
ScxmlImporter::createKeyEventTransition(const QString &event)
{
  auto lastDotIndex = event.lastIndexOf(".");
  auto secondLastDotIndex = event.lastIndexOf(".", lastDotIndex - 1);
  auto typeString =
      event.mid(secondLastDotIndex + 1, lastDotIndex - secondLastDotIndex - 1);
  QEvent::Type eventType =
      typeString == "DOWN" ? QEvent::KeyPress : QEvent::KeyRelease;

  auto keyAsString = event.mid(lastDotIndex + 1);
  auto keyCode = toKey(keyAsString);
  return new QKeyEventTransition(signalManager->getFor("KeyboardEventSender"),
                                 eventType, keyCode, stateStack.top());
}

QAbstractTransition *
ScxmlImporter::createMouseButtonEventTransition(const QString &event)
{
  auto lastDotIndex = event.lastIndexOf(".");
  auto secondLastDotIndex = event.lastIndexOf(".", lastDotIndex - 1);
  auto typeString =
      event.mid(secondLastDotIndex + 1, lastDotIndex - secondLastDotIndex - 1);
  QEvent::Type eventType = typeString == "DOWN" ? QEvent::MouseButtonPress
                                                : QEvent::MouseButtonRelease;

  auto keyAsString = event.mid(lastDotIndex + 1);
  auto buttonCode = toButton(keyAsString);
  return new QMouseEventTransition(signalManager->getFor("KeyboardEventSender"),
                                   eventType, buttonCode, stateStack.top());
}

QAbstractTransition *
ScxmlImporter::createMouseMoveEventTransition()
{
  QEvent::Type eventType = QEvent::MouseMove;

  auto buttonCode = Qt::MouseButton::NoButton;
  return new QMouseEventTransition(signalManager->getFor("KeyboardEventSender"),
                                   eventType, buttonCode, stateStack.top());
}

QAbstractTransition *ScxmlImporter::createSignalTransition(const QString &event)
{
  auto dotPosition = event.indexOf(".");
  auto name = event.left(dotPosition);
  QString signal = "2" + event.mid(dotPosition + 1) + "()";
  return new QSignalTransition(signalManager->getFor(name),
                               signal.toStdString().c_str(), stateStack.top());
}

QString ScxmlImporter::attributeAsString(const char *name)
{
  return reader->attributes().value(name).toString();
}

ScxmlImporter::ScxmlElement ScxmlImporter::elementFromString(QString name)
{
  bool couldConvert = false;
  auto result = static_cast<ScxmlImporter::ScxmlElement>(
      metaScxmlElement.keyToValue(name.toStdString().c_str(), &couldConvert));

  if (!couldConvert)
    throw std::runtime_error("Could not convert " + name.toStdString());

  return result;
}

