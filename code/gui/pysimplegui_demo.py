# requires: pysimplegui==6.0

import PySimpleGUI as sg

sg.theme("SystemDefault")

layout = [
    [sg.Text("Birdsong Classification")],
    [
        sg.Input(key='-FILE-', visible=True, enable_events=True, readonly=True), 
        sg.FileBrowse(button_text="Select File")
    ],
    [sg.Button("Run")],
    [sg.Pane(
        [sg.Column(
            layout=[
                [sg.Text(text="Press \'Run\' to change me :)", enable_events=True, key="-OUTPUT_PANE-")]
            ]
        )],
        size=(400, 300)
    )]
]

window = sg.Window(
    title="Birdsong Classification - up2178845", 
    layout=layout,
#    size=(400, 300)
)

while True:
    # window loop

    event, values = window.read()

    if event == sg.WIN_CLOSED or event == "Run":
        values["-OUTPUT_PANE-"] = "Hi :)"

window.close()