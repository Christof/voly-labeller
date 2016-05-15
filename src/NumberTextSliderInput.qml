import QtQuick 2.1
import QtQuick.Controls 1.2

Item {
  TextInput {
    x: 10
    id: weightInput
    maximumLength: 6
    width: 50
    validator: DoubleValidator {
      bottom: 0
      top: 10
    }
    inputMethodHints: Qt.ImhFormattedNumbersOnly
    horizontalAlignment: Qt.AlignRight
    text: styleData.value
    color: model ? model.forceColor : "black"
    onTextChanged: {
      weightSlider.value = text;
      if (labeller) labeller.changeWeight(styleData.row, text);
    }
  }
  Slider {
    id: weightSlider
    width: 180
    x: 66
    minimumValue: 0
    maximumValue: 10
    value: styleData.value
    onValueChanged: {
      weightInput.text = value;
      if (labeller) labeller.changeWeight(styleData.row, value);
    }
  }
}

