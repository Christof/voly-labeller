import QtQuick 2.1
import QtQuick.Controls 1.2

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
    Item {
      Loader {
        id: numberTextDelegate
        source: "NumberTextSliderInput.qml"
      }

      Connections {
        target: numberTextDelegate.item
        onInputValueChanged: {
          if (placement) placement.changeWeight(styleData.row, value);
        }
      }

      Binding {
        target: numberTextDelegate.item
        property: "value"
        value: styleData.value
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
    height: 170
    model: placement
    focus: true
    clip: true
  }
}
