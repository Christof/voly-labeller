import QtQuick 2.1
import QtQuick.Controls 1.2

Item {
  id: root
  signal inputValueChanged(real value)
  property real value
  property color color
  property real minValue: 0
  property real maxValue: 10

  TextInput {
    x: 10
    id: weightInput
    maximumLength: 6
    width: 50
    validator: DoubleValidator {
      bottom: root.minValue
      top: root.maxValue
    }
    inputMethodHints: Qt.ImhFormattedNumbersOnly
    horizontalAlignment: Qt.AlignRight
    text: root.value
    color: root.color
    onTextChanged: {
      weightSlider.value = text;
      root.inputValueChanged(text);
    }
  }
  Slider {
    id: weightSlider
    width: 180
    x: 66
    minimumValue: root.minValue
    maximumValue: root.maxValue
    value: root.value
    onValueChanged: {
      weightInput.text = value;
      root.inputValueChanged(value);
    }
  }
}

