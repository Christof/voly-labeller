import QtQuick 2.1
import QtQuick.Controls 1.2

Item {
  Component {
    id: cameraPositionNameDelegate

    Item {
      TextInput {
        anchors.fill: parent
        text: styleData.value
        onTextChanged: {
          if (labels) cameraPositions.changeName(styleData.row, text);
        }
      }
    }
  }
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

  Column {
    x: 70; y: 20
    width: 430
    height: 170
    Row {
      id: row1
      spacing: 10

      Label {
        text: "Someone"
      }
    }

    TableView {
      id: cameraPositionsTableView
      TableViewColumn {
        role: "name"
        title: "Name"
        width: 150
        delegate: cameraPositionNameDelegate
      }
      width: 417
      height: 190
      model: cameraPositions
      focus: true
      clip: true
    }

    Row {
      spacing: 10
      Button {
        text: "Save camera"
        onClicked: {
          if (cameraPositions) cameraPositions.save();
        }
      }
      Button {
        text: "Delete Label"
        onClicked: {
          if (labels) labels.deleteLabel(labelsView.currentRow);
        }
      }
    }
  }
}
