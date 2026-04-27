# requires: pysimplegui==6.0

import pathlib
import numpy as np
from math import floor
import PySimpleGUI as sg
import torch
import torch.nn as nn
import torchaudio

class TestModel(nn.Module):
    def __init__(self, input_shape: torch.Size, dropout_rate: float = 0, total_output_classes: int = 50):
        super().__init__()

        # input shape should be some list/tuple of length 4
        if len(input_shape) != 4: return Exception("Input shape is not AxBxCxD.")
        
        A = input_shape[0]
        B = input_shape[1]
        C = input_shape[2]
        D = input_shape[3]

        print(input_shape)

        self.relu = nn.ReLU() # relu does not have trainable parameters, thus, can be reused

        self.conv1 = nn.Conv2d(in_channels=B, out_channels=10, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.flat = nn.Flatten()

        conv_layers = 2
        pool_layers = 2
        final_width = C
        final_height = D

        # ASSUMES KERNEL SIZE IS 3 AND 2 FOR CONV AND POOL LAYERS
        while conv_layers > 0 and pool_layers > 0:
            if conv_layers > 0:
                final_width = final_width - 2
                final_height = final_height - 2
                conv_layers -= 1
            if pool_layers > 0:
                final_width = final_width // 2
                final_height = final_height // 2
                pool_layers -= 1

        flatten_nodes = 10 * final_width * final_height
        
        self.norm = nn.BatchNorm1d(num_features=flatten_nodes)
        self.linear1 = nn.Linear(in_features=flatten_nodes, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=512)
        self.linear3 = nn.Linear(in_features=512, out_features=128) 
        self.output = nn.Linear(in_features=128, out_features=total_output_classes)

    def forward(self, x):
        # define calculations here
        x = self.conv1(x)
        x = self.relu(x)

        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.pool2(x)
        x = self.drop2(x)

        x = self.flat(x)
        x = self.norm(x)

        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu(x)

        x = self.linear3(x)
        x = self.relu(x)

        x = self.output(x)
        x = self.relu(x)

        return x

def parse_file(file_path, split_interval_secs):
    # run file
    # convert into mel-spectrogram
    # split into X second sound bytes (depending on model)
    # feed sound bytes into model
    # aggregate and output
    samples = []

    sample_rate = 32000
    split_length = 125
    mean_threshold = 5
    variance_threshold = 0.2

    folder_key = {
        0: 'dowwoo',
        1: 'rerswa1',
        2: 'cobtan1',
        3: 'barswa',
        4: 'indbun',
        5: 'comloo',
        6: 'mitpar',
        7: 'blhpar1',
        8: 'amepip',
        9: 'brnjay',
        10: 'sheowl',
        11: 'peflov',
        12: 'zebdov',
        13: 'yehcar1',
        14: 'cubthr',
        15: 'amtspa',
        16: 'spotow',
        17: 'buwwar',
        18: 'insowl1',
        19: 'annhum',
        20: 'bkskit1',
        21: 'bkbmag1',
        22: 'litegr',
        23: 'verdin',
        24: 'comyel',
        25: 'brubru1',
        26: 'blkfra',
        27: 'combuz1',
        28: 'whiwre1',
        29: 'yebsap',
        30: 'leater1',
        31: 'piebus1',
        32: 'yehbla',
        33: 'royter1',
        34: 'lotduc',
        35: 'scbwre1',
        36: 'banswa',
        37: 'logshr',
        38: 'combul2',
        39: 'brncre',
        40: 'whbman1',
        41: 'compau',
        42: 'pirfly1',
        43: 'ocbfly1',
        44: 'oliwoo1',
        45: 'eucdov',
        46: 'phaino',
        47: 'mawthr1',
        48: 'redcro',
        49: 'grycat',
        50: 'SILENT'
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load file data and resample to sample_rate if necessary
    file_data, file_sample_rate_hz = torchaudio.load(uri=file_path, channels_first=True)
    if file_sample_rate_hz != sample_rate:
        file_data = torchaudio.functional.resample(file_data, orig_freq=file_sample_rate_hz, new_freq=sample_rate)

    # convert all audio into mono (1 channel) if audio is stereo (2 channels)
    if file_data.shape[0] == 2: file_data = file_data.mean(dim=0)
    else: file_data = file_data.flatten()

    # get total number of X second splits
    total_splits = floor(len(file_data) / int(sample_rate * split_interval_secs))
    
    # convert file data into mel-spectrogram fourier transform for feeding into CNN
    n_fft = 1024

    mel_spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate = sample_rate, power=2, n_fft=n_fft)
    amp_to_db_transform = torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=80)
    mel_spec_data_db = amp_to_db_transform(mel_spec_transform(file_data))
    
    # if file is at least X seconds.
    if total_splits >= 1:
        split_length = min(mel_spec_data_db.shape[1] // total_splits, split_length)
        mel_spec_splits = np.arange(0, mel_spec_data_db.shape[1], split_length)
        for i in range(len(mel_spec_splits) - 1):
            start = mel_spec_splits[i]
            end = mel_spec_splits[i + 1]

            mel_spec_split = mel_spec_data_db[:, start:end]

            var = np.var(mel_spec_split.numpy())
            mean = np.mean(mel_spec_split.numpy())

            #if var > variance_threshold:
            # automatically keep anything ABOVE mean db level
            if mean > mean_threshold:
                samples.append(mel_spec_split.reshape(1, 128, -1).numpy())
            else:
                # if mean is LOW, only include those with variance ABOVE threshold
                if var > variance_threshold:
                    samples.append(mel_spec_split.reshape(1, 128, -1).numpy())
            
    else:
        return None

    samples = torch.Tensor(np.array(samples)).to(device)

    model = TestModel(input_shape=samples.shape, dropout_rate=0.5).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        preds = model(samples)
    
    # Ax[classes] sized array, A = number of splits in file
    # sequential aggregation via summation (appears to be the best solution right now, but barely by much)
    seq_aggregation_result = torch.sum(preds, dim=0)
    seq_top_k = torch.topk(seq_aggregation_result, k=len(seq_aggregation_result))
    seq_top_k_softmax = torch.softmax(seq_top_k.values, dim=0)
    prediction_order = [(folder_key[i], seq_top_k_softmax[i].cpu().numpy().item()) for i in seq_top_k.indices.cpu().numpy() if folder_key[i] != "SILENT"]

    return prediction_order

def format_predictions(predictions):
    prediction_sorted = sorted(predictions, key = lambda x: x[1], reverse=True)

    prediction_filtered = [x for x in prediction_sorted if x[1] > 0.001]

    output = ""

    for pred in prediction_filtered:
        output += f"{species_key[pred[0]]}: {round(pred[1] * 100, 2)}%\n"

    output = output[:-1]

    return output

np.set_printoptions(formatter={'float_kind':'{:.20f}'.format})
sg.theme("SystemDefault")

file_format_text = sg.Text(text="", visible=True, colors=("Red", None))
output_text = sg.Text(text="Press \'Run\' to change me :)", enable_events=True, key="-OUTPUT_PANE-")

model_path = str(pathlib.Path(__file__).parent.resolve()) + "/../development/models/best_model_070.pth"

species_key = {row.split(",")[0].replace("\n", ""): row.split(",")[1].replace("\n", "") for row in open(str(pathlib.Path(__file__).parent.resolve()) + "/species_key.csv").readlines()}

layout = [
    [sg.Text("Birdsong Classification")],
    [
        sg.Input(key='-FILE-', visible=True, enable_events=True, readonly=True), 
        sg.FileBrowse(button_text="Select File")
    ],
    [
        sg.Button("Run"),
        file_format_text
    ],
    [sg.Pane(
        [sg.Column(
            layout=[
                [
                    output_text
                ]
            ]
        )],
        size=(400, 300)
    )]
]

window = sg.Window(
    title="Birdsong Classification - up2178845", 
    layout=layout
)

while True:
    # window loop

    event, values = window.read()

    if event == "Run":
        file_format_text.Update(value="")

        # check file submitted is wav or mp3
        if values["-FILE-"][-3:].lower() not in ["wav", "mp3", "ogg"]:
            file_format_text.Update(value="File must be .wav, .ogg or .mp3 only.")
            continue
        
        # if file is all good, attempt to parse and predict
        file_format_text.Update(value="Success!", text_color="Green")

        predictions = parse_file(
            file_path=values["-FILE-"],
            split_interval_secs=2
        )

        predictions_str = format_predictions(predictions)

        output_text.Update(value=predictions_str)
        

    if event == sg.WINDOW_CLOSED:
        break

window.close()