import QtQuick 2.7
import QtQuick.Controls 1.4

Item {
  id: root
  signal inputValueChanged(real value)
  property real value
  property color color: "black"
  property alias minTextValue: textInputValidator.bottom
  property alias maxTextValue: textInputValidator.top
  property alias minSliderValue: slider.minimumValue
  property alias maxSliderValue: slider.maximumValue

  TextInput {
    x: 10
    id: textInput
    maximumLength: 6
    width: 50
    validator: DoubleValidator {
      id: textInputValidator
      bottom: 0
      top: 10
    }
    inputMethodHints: Qt.ImhFormattedNumbersOnly
    horizontalAlignment: Qt.AlignRight
    text: root.value
    color: root.color
    onTextChanged: {
      slider.value = text;
      root.inputValueChanged(text);
    }
  }
  Slider {
    id: slider
    width: 150
    x: 66
    activeFocusOnPress: true
    minimumValue: 0
    maximumValue: 10
    value: root.value
    onValueChanged: {
      textInput.text = value;
      root.inputValueChanged(value);
    }
  }
}

