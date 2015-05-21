#ifndef SRC_INPUT_KEY_HELPER_H_

#define SRC_INPUT_KEY_HELPER_H_

#include <QString>
#include <QKeySequence>
#include <cassert>

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

#endif  // SRC_INPUT_KEY_HELPER_H_
