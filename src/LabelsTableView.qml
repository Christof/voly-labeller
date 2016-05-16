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
    x: 10
    y: 192
    width: 441
    height: 197
    visible: labels.isVisible
    Row {
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
