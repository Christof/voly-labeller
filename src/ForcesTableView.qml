import QtQuick 2.1
import QtQuick.Controls 1.2

Item {
  Component {
    id: forceDelegate
    Item {
      width: 400; height: 30
      Row {
        CheckBox {
          checked: enabled
        }
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
    id: checkBoxDelegate

    Item {
      CheckBox {
        anchors.fill: parent
        checked: styleData.value
        onCheckedChanged: {
          if (labeller) labeller.changeEnabled(styleData.row, checked);
        }
      }
    }
  }

  Component {
    id: textDelegate
    Item {
      Loader {
        id: numberTextDelegate
        source: "NumberTextSliderInput.qml"
      }

      Connections {
        target: numberTextDelegate.item
        onInputValueChanged: {
          if (labeller) labeller.changeWeight(styleData.row, value);
        }
      }

      Binding {
        target: numberTextDelegate.item
        property: "value"
        value: styleData.value
      }

      Binding {
        target: numberTextDelegate.item
        property: "color"
        value: model ? model.forceColor : "black"
      }
    }
  }
  Component {
    id: nameDelegate

    Item {
      Text {
        text: styleData.value
        color: model ? model.forceColor : "black"
      }
    }
  }

  TableView {
    visible: labeller.isVisible
    TableViewColumn {
      role: "enabled"
      title: "Enabled"
      width: 100
      delegate: checkBoxDelegate
    }
    TableViewColumn {
      role: "name"
      title: "Name"
      width: 200
      delegate: nameDelegate
    }
    TableViewColumn {
      role: "weight"
      title: "Weight"
      width: 220
      delegate: textDelegate
    }
    x: 10; y: 36
    width: 530
    height: 150
    model: labeller
    focus: true
    clip: true
  }
}
