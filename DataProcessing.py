import pandas as pd
import os
import numpy as np
import librosa
import librosa.display
import skimage.io
import torch


class DataProcessing:
    """
            Class for generating data for models
    """

    def __init__(self):
        self.path_data = "../data/DEAM/"
        self.path_dnn_data = self.path_data + "1_dnn_averaged/"
        self.path_spectrograms = self.path_data + "2_cnn/"
        self.path_lstm_data = self.path_data + "3_lstm/"

        self.path_metadata = self.path_data + "metadata/metadata_2013.csv"
        self.path_features = self.path_data + "features/"
        self.path_audio = self.path_data + "MEMD_audio/"

        self.path_annotations = self.path_data + "annotations/annotations averaged per song/"
        self.path_annotations_static = self.path_annotations + "song_level/static_annotations_averaged_songs_1_2000.csv"
        self.path_annotations_dynamic_valence = self.path_annotations + "dynamic (per second annotations)/valence.csv"
        self.path_annotations_dynamic_arousal = self.path_annotations + "dynamic (per second annotations)/arousal.csv"

        self.invalid_data_features = [146, 240, 272, 435, 810]
        self.metadata = None
        self.features = None
        self.labels = None
        self.dataframe = None
        self.arousal_dynamic = None
        self.valence_dynamic = None
        self.lstm_data_tensor = None
        self.lstm_labels_tensor = None
        self._deam_processing()

    def _deam_processing(self) -> None:
        self._dnn_1()
        self._cnn_2()
        self._lstm_3()

    def _dnn_1(self):
        """
                Loads dataframes for deep neural network and other models, if directory exists, otherwise creates and
                saves them.
        """
        if os.path.isdir(self.path_dnn_data):
            self.metadata = pd.read_pickle(self.path_dnn_data + "metadata.pkl")
            self.features = pd.read_pickle(self.path_dnn_data + "features.pkl")
            self.labels = pd.read_pickle(self.path_dnn_data + "labels.pkl")
            self.dataframe = pd.read_pickle(self.path_dnn_data + "dataframe.pkl")

        else:
            self.metadata = self._deam_metadata()
            self.features = self._deam_features()
            self.labels = self._deam_labels()
            self.dataframe = self._deam_dataframe()
            os.mkdir(self.path_dnn_data)
            self.metadata.to_pickle(self.path_dnn_data + "metadata.pkl", protocol=4)
            self.features.to_pickle(self.path_dnn_data + "features.pkl", protocol=4)
            self.labels.to_pickle(self.path_dnn_data + "labels.pkl", protocol=4)
            self.dataframe.to_pickle(self.path_dnn_data + "dataframe.pkl", protocol=4)

    def _cnn_2(self):
        """
                Creates and saves images for convolutional neural network, if they don't exist.
        """
        if not os.path.isdir(self.path_spectrograms):
            os.mkdir(self.path_spectrograms)
            self._melspectrogram_images()

    def _lstm_3(self):
        """
                Creates and saves tensors for LSTM if they don't exist
        """
        if os.path.isdir(self.path_lstm_data):
            self.lstm_data_tensor = torch.load(self.path_lstm_data + "data_tensor.pt")
            self.lstm_labels_tensor = torch.load(self.path_lstm_data + "labels_tensor.pt")

        else:
            os.mkdir(self.path_lstm_data)
            self.arousal_dynamic = self._deam_arousal_dynamic()
            self.valence_dynamic = self._deam_valence_dynamic()
            self.lstm_data_tensor, self.lstm_labels_tensor = self._lstm_tensors()
            torch.save(self.lstm_data_tensor, self.path_lstm_data + "data_tensor.pt")
            torch.save(self.lstm_labels_tensor, self.path_lstm_data + "labels_tensor.pt")

    def _deam_metadata(self) -> pd.DataFrame:
        """
                Returns metadata dataframe, with added columns, representing start and end of sample in miliseconds
        """
        metadata = pd.read_csv(self.path_metadata)
        sample_start = []
        sample_end = []
        for _, song in metadata.iterrows():
            start = str(song["start of the segment (min.sec)"]).split(
                ".")  # calculate start and end of sample from min.sec representation to miliseconds
            start = int(start[0]) * 60 * 1000 + int(start[1].ljust(2, "0")) * 1000
            end = str(song["end of the segment (min.sec)"]).split(".")
            end = int(end[0]) * 60 * 1000 + int(end[1].ljust(2, "0")) * 1000
            sample_start.append(start)
            sample_end.append(end)
        metadata["sample_start_ms"] = sample_start
        metadata["sample_end_ms"] = sample_end
        return metadata

    def _deam_features(self) -> pd.DataFrame:
        """
                Returns dataframe consisting of averaged song features
        """
        features = pd.DataFrame()
        for _, song_metadata in self.metadata.iterrows():
            song_id = int(song_metadata["song_id"])
            if song_id in self.invalid_data_features:
                continue

            start = int(song_metadata["sample_start_ms"] / 500)  # /1000 bcs of ms and * 2 because 2Hz sampling rate
            end = int(song_metadata["sample_end_ms"] / 500)
            song_features = pd.read_csv(f"{self.path_features}{song_id}.csv", delimiter=";")[start:end]
            song_features = song_features.mean(axis=0)[1:]  # calculate average for every column
            song_features = pd.concat(
                [pd.Series(data={"song_id": song_id}, index=["song_id"]), song_features])  # add song id
            song_features = pd.DataFrame(song_features).T
            features = pd.concat([features, song_features])

        features = features.dropna()

        return features

    def _deam_labels(self) -> pd.DataFrame:
        arousal_valence_averaged = pd.read_csv(self.path_annotations_static)

        return arousal_valence_averaged

    def _deam_dataframe(self) -> pd.DataFrame:
        """
                Returns final dataframe, that was used by models, consisting of songs features and labels
        """
        final_data = pd.concat(
            [self.labels[self.labels["song_id"].isin(self.features["song_id"])].reset_index(drop=True),
             self.features.reset_index(drop=True).drop(columns="song_id")], axis=1)

        return final_data

    def scale_minmax(self, x: np.ndarray, min: float = 0.0, max: float = 1.0):
        """
                Min-max scaling, used to convert images to monochromatic scale
        """
        x_std = (x - x.min()) / (x.max() - x.min())
        x_scaled = x_std * (max - min) + min
        return x_scaled

    def save_melspectrogram(self, y: np.ndarray, sr: int, out: str, n_mels: int = 256) -> None:
        """ duration: int = 45,  , n_mels: int = 256
                Creates and saves mel-spectrogram image of song
                    Parameters:
                            y (np.ndarray [shape=(…, n)]): Audio time-series
                            sr (int): Sampling rate of y
                            out (str): Path where image should be saved
                            n_mels (int): Number of Mel bands to generate (height of image)
        """

        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mels = np.log(mels + 1e-9)  # add small number to avoid log(0)

        img = self.scale_minmax(mels, 0, 255).astype(np.uint8)  # min-max scale to fit inside 8-bit range
        img = np.flip(img, axis=0)  # put low frequencies at the bottom in image

        # save as PNG
        skimage.io.imsave(out, img)

    def _melspectrogram_images(self, duration: float = 45.0) -> None:
        """
                Prepares data for creating mel-spectrogram
                Parameters:
                            duration (float): Length of file to load, in seconds
        """
        for _, song in self.metadata.iterrows():
            path = f"{self.path_audio}{song['song_id']}.mp3"
            y, sr = librosa.load(path, duration=duration)  # load an audio file as a floating point time series
            if not os.path.exists(self.path_spectrograms + f"{song['song_id']}"):
                os.mkdir(self.path_spectrograms + f"{song['song_id']}")
            out = f"{self.path_spectrograms}{song['song_id']}/{song['song_id']}.png"

            self.save_melspectrogram(y, sr=sr, out=out)
            print(f"wrote file {song['song_id']}", out)

    def _deam_arousal_dynamic(self) -> pd.DataFrame:
        arousal = pd.read_csv(self.path_annotations_dynamic_arousal)
        return arousal

    def _deam_valence_dynamic(self) -> pd.DataFrame:
        valence = pd.read_csv(self.path_annotations_dynamic_valence)
        return valence

    def _lstm_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
                Creates tensors for LSTM model
        """
        data_tensor = None
        label_tensor = None
        for _, song in self.metadata.iterrows():
            song_id = song["song_id"]
            if song_id in self.invalid_data_features:  # invalid data, given end of sample is placed after songs end
                continue                                # (58/59 data rows instead of 60 needed)
            start = int(song["sample_start_ms"] / 500) + 30  # /1000 bcs of ms and * 2 because 2Hz sampling rate,
            # + 30 bcs annotations excludes the first 15 seconds due to instability of the annotations at the start.
            end = int(song["sample_end_ms"] / 500)
            song_features = pd.read_csv(f"{self.path_features}{song_id}.csv", delimiter=";")[start:end]

            # first is song id, 60 bcs 30 sec and 2Hz
            song_arousal = self.arousal_dynamic[self.arousal_dynamic["song_id"] == song_id].iloc[:, 1:60 + 1]
            song_arousal = song_arousal.apply(
                lambda x: (x * 5 + 5) / 10)  # values from [-1;1] to [0;10] and then to [0;1] for evaluation
            song_valence = self.valence_dynamic[self.valence_dynamic["song_id"] == song_id].iloc[:, 1:60 + 1]
            song_valence = song_valence.apply(
                lambda x: (x * 5 + 5) / 10)  # values from [-1;1] to [0;10] and then to [0;1] for evaluation

            song_data = song_features.assign(arousal=song_arousal.transpose().values)
            song_data = song_data.assign(valence=song_valence.transpose().values)
            song_labels = song_data[["arousal", "valence"]].copy()
            song_data = song_data.drop(columns=["frameTime", "arousal", "valence"])
            song_data_tensor = torch.tensor(song_data.values)
            song_labels_tensor = torch.tensor(song_labels.values)
            if data_tensor is None:
                data_tensor = song_data_tensor.unsqueeze(0)
                label_tensor = song_labels_tensor.unsqueeze(0)
            else:
                data_tensor = torch.cat((data_tensor, song_data_tensor.unsqueeze(0)))
                label_tensor = torch.cat((label_tensor, song_labels_tensor.unsqueeze(0)))

        return data_tensor, label_tensor


if __name__ == "__main__":
    deam = DataProcessing()
