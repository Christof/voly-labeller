import QtQuick 2.1
import QtQuick.Controls 1.2

Item {
  Component {
    id: labelNameDelegate

    Item {
      TextInput {
        anchors.fill: parent
        text: styleData.value
        onTextChanged: {
          if (labels) labels.changeText(styleData.row, text);
        }
      }
    }
  }
  Component {
    id: sizeYDelegate

    Item {
      TextInput {
        anchors.fill: parent
        maximumLength: 5
        text: styleData.value
        onTextChanged: {
          if (labels) labels.changeSizeY(styleData.row, text);
        }
      }
    }
  }
  Component {
    id: sizeXDelegate

    Item {
      TextInput {
        anchors.fill: parent
        maximumLength: 5
        text: styleData.value
        onTextChanged: {
          if (labels) labels.changeSizeX(styleData.row, text);
        }
      }
    }
  }
  Component {
    id: anchorDelegate

    Item {
      Button {
        anchors.fill: parent
        text: "Pick"
        onClicked: {
          if (labels) labels.pick(styleData.row);
        }
      }
    }
  }

  Column {
    id: labelsItem
    spacing: 4
    x: 10
    y: 192
    width: 441
    height: 263
    visible: labels.isVisible
    Row {
      id: row1
      spacing: 10

      Label {
        text: "Anchor size"
      }

      Item {
        x: 0
        width: 250
        height: 27
        Loader {
          id: anchorSize
          source: "NumberTextSliderInput.qml"
        }

        Connections {
          target: anchorSize.item
          onInputValueChanged: {
            if (nodes) nodes.changeAnchorSize(value);
          }
        }

        Binding {
          target: anchorSize.item
          property: "value"
          value: nodes.anchorSize
        }

        Binding {
          target: anchorSize.item
          property: "minSliderValue"
          value: 0.001
        }
        Binding {
          target: anchorSize.item
          property: "maxSliderValue"
          value: 0.1
        }
      }
    }
    Row {
      spacing: 10
      Button {
        text: "Add Label"
        onClicked: {
          if (labels) labels.addLabel();
        }
      }
      Button {
        text: "Delete Label"
        onClicked: {
          if (labels) labels.deleteLabel(labelsView.currentRow);
        }
      }
    }

    TableView {
      id: labelsView
      TableViewColumn {
        role: "name"
        title: "Name"
        width: 150
        delegate: labelNameDelegate
      }
      TableViewColumn {
        role: "sizeX"
        title: "Width"
        width: 100
        delegate: sizeXDelegate
      }
      TableViewColumn {
        role: "sizeY"
        title: "Height"
        width: 100
        delegate: sizeYDelegate
      }
      TableViewColumn {
        title: "Anchor"
        width: 46
        delegate: anchorDelegate
      }
      width: 417
      height: 190
      model: labels
      focus: true
      clip: true
    }
  }
}
