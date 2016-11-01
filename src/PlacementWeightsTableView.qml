import QtQuick 2.7
import QtQuick.Controls 1.4

Item {
  Component {
    id: weightDelegate
    Item {
      width: 400; height: 30
      Row {
        Text {
          text: name
          width: 200
          focus: true
        }
        TextEdit { text: weight }
      }
    }
  }

  Component {
    id: costFunctionWeightDelegate
    TextInput {
      x: 10
      id: textInput
      width: 30
      rightPadding: 10
      validator: DoubleValidator {
        id: textInputValidator
        bottom: 0
        decimals: 5
        top: Number.MAX_VALUE
        notation: DoubleValidator.ScientificNotation
      }
      horizontalAlignment: TextInput.AlignRight
      text: styleData.value
      onTextChanged: {
        if (placement) placement.changeWeight(styleData.row, text);
      }
    }
  }

  TableView {
    visible: placement.isVisible
    TableViewColumn {
      role: "name"
      title: "Name"
      width: 200
      delegate:
        Item {
          Text {
            text: styleData.value
          }
        }
    }
    TableViewColumn {
      role: "weight"
      title: "Weight"
      width: 220
      delegate: costFunctionWeightDelegate
    }
    x: 570; y: 51
    width: 430
    height: 200
    model: placement
    focus: true
    clip: true
  }
}
