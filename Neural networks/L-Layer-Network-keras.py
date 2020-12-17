import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import matplotlib.pyplot as plt
from time import gmtime, strftime
from scipy.signal import butter, lfilter


# function for predicting testset accuracy
def accuracy(y, pred):
    count = 0.0
    for i in range(0, y.shape[0]):
        if (y[i][0] == pred[i]):
            count = count + 1
    return count * 100 / y.shape[0]


# importing the dataset
X = pd.read_csv("reduced_features.csv")
Z = pd.read_csv("feature.csv")
X = X.iloc[:, :].values
Y = pd.read_csv("target_output.csv")
Y = Y.iloc[:, :].values
print(X.shape);
print(Z.shape);

# encoding categorical data


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features=[0])
Y_train = onehotencoder.fit_transform(Y_train).toarray()

# feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# initialising the ann
classifier = Sequential()

# adding dropout to the visible layer i.e betwen the input and 1st hidden layer
classifier.add(Dropout(0.3, input_shape=(175,)))

# adding input layer and 1st hidden layer
classifier.add(Dense(output_dim=10, activation='relu', init='uniform', input_dim=175))

# adding droput to hidden layer 1
classifier.add(Dropout(0.3))

# adding second hidden layer
classifier.add(Dense(output_dim=10, activation='relu', init='uniform'))

# adding droput to hidden layer 2
classifier.add(Dropout(0.3))

# adding third hidden layer
classifier.add(Dense(output_dim=10, activation='relu', init='uniform'))

# adding droput to hidden layer 3
classifier.add(Dropout(0.3))

# adding output layer
classifier.add(Dense(output_dim=13, activation='softmax', init='uniform'))

# compiling ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fitting the ANN to training set
classifier.fit(X_train, Y_train, batch_size=30, epochs=500)


# Making the confusion Matrix
from sklearn.metrics import confusion_matrix, recall_score
from  sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# predicting test sets results
Y_pred = classifier.predict(X_test)
#Y_pred1 = classifier.predict(X[5,:])
Y_test_op = ((np.argmax(Y_pred, axis=1) + 1))
np.reshape(Y_test_op, (Y_pred.shape[0], 1))
print(accuracy(Y_test, Y_test_op))
#print(recall_score(Y_test,Y_pred))

ra = RandomForestClassifier()
ra.fit(X_train,Y_train)

Y_pred1 = ra.predict(X_test)
Y_test_op1 = ((np.argmax(Y_pred1, axis=1) + 1))
np.reshape(Y_test_op1, (Y_pred1.shape[0], 1))
print(accuracy(Y_test, Y_test_op1))


# ab = AdaBoostClassifier()
# ab.fit(X_train,Y_train)
#
# Y_pred2 = ab.predict(X_test)
# Y_test_op2 = ((np.argmax(Y_pred2, axis=1) + 1))
# np.reshape(Y_test_op2, (Y_pred2.shape[0], 1))
# print(accuracy(Y_test, Y_test_op2))



cm = confusion_matrix(Y_test, Y_test_op)
print("confusion matrix")
print(cm)




LOG_DIR = "logs/"
PLOT_DIR = "plots/"



class QRSDetectorOffline(object):


    def __init__(self, ecg_data_path, verbose=True, log_data=False, plot_data=False, show_plot=False):
        """
        QRSDetectorOffline class initialisation method.
        :param string ecg_data_path: path to the ECG dataset
        :param bool verbose: flag for printing the results
        :param bool log_data: flag for logging the results
        :param bool plot_data: flag for plotting the results to a file
        :param bool show_plot: flag for showing generated results plot - will not show anything if plot is not generated
        """
        # Configuration parameters.
        self.ecg_data_path = ecg_data_path

        self.signal_frequency = 250  # Set ECG device frequency in samples per second here.

        self.filter_lowcut = 0.0
        self.filter_highcut = 15.0
        self.filter_order = 1

        self.integration_window = 15  # Change proportionally when adjusting frequency (in samples).

        self.findpeaks_limit = 0.35
        self.findpeaks_spacing = 50  # Change proportionally when adjusting frequency (in samples).

        self.refractory_period = 120  # Change proportionally when adjusting frequency (in samples).
        self.qrs_peak_filtering_factor = 0.125
        self.noise_peak_filtering_factor = 0.125
        self.qrs_noise_diff_weight = 0.25

        # Loaded ECG data.
        self.ecg_data_raw = None

        # Measured and calculated values.
        self.filtered_ecg_measurements = None
        self.differentiated_ecg_measurements = None
        self.squared_ecg_measurements = None
        self.integrated_ecg_measurements = None
        self.detected_peaks_indices = None
        self.detected_peaks_values = None

        self.qrs_peak_value = 0.0
        self.noise_peak_value = 0.0
        self.threshold_value = 0.0

        # Detection results.
        self.qrs_peaks_indices = np.array([], dtype=int)
        self.noise_peaks_indices = np.array([], dtype=int)

        # Final ECG data and QRS detection results array - samples with detected QRS are marked with 1 value.
        self.ecg_data_detected = None

        # Run whole detector flow.
        self.load_ecg_data()
        self.detect_peaks()
        self.detect_qrs()

        if verbose:
            self.print_detection_data()

        if log_data:
            self.log_path = "{:s}QRS_offline_detector_log_{:s}.csv".format(LOG_DIR,
                                                                           strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
            self.log_detection_data()

        if plot_data:
            self.plot_path = "{:s}QRS_offline_detector_plot_{:s}.png".format(PLOT_DIR,
                                                                             strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
            self.plot_detection_data(show_plot=show_plot)

    """Loading ECG measurements data methods."""

    def load_ecg_data(self):
        """
        Method loading ECG data set from a file.
        """
        self.ecg_data_raw = np.loadtxt(self.ecg_data_path, skiprows=1, delimiter=',')

    """ECG measurements data processing methods."""

    def detect_peaks(self):
        """
        Method responsible for extracting peaks from loaded ECG measurements data through measurements processing.
        """
        # Extract measurements from loaded ECG data.
        ecg_measurements = self.ecg_data_raw[:, 1]

        # Measurements filtering - 0-15 Hz band pass filter.
        self.filtered_ecg_measurements = self.bandpass_filter(ecg_measurements, lowcut=self.filter_lowcut,
                                                              highcut=self.filter_highcut, signal_freq=self.signal_frequency,
                                                              filter_order=self.filter_order)
        self.filtered_ecg_measurements[:5] = self.filtered_ecg_measurements[5]

        # Derivative - provides QRS slope information.
        self.differentiated_ecg_measurements = np.ediff1d(self.filtered_ecg_measurements)

        # Squaring - intensifies values received in derivative.
        self.squared_ecg_measurements = self.differentiated_ecg_measurements ** 2

        # Moving-window integration.
        self.integrated_ecg_measurements = np.convolve(self.squared_ecg_measurements, np.ones(self.integration_window))

        # Fiducial mark - peak detection on integrated measurements.
        self.detected_peaks_indices = self.findpeaks(data=self.integrated_ecg_measurements,
                                                     limit=self.findpeaks_limit,
                                                     spacing=self.findpeaks_spacing)

        self.detected_peaks_values = self.integrated_ecg_measurements[self.detected_peaks_indices]

    """QRS detection methods."""

    def detect_qrs(self):
        """
        Method responsible for classifying detected ECG measurements peaks either as noise or as QRS complex (heart beat).
        """
        for detected_peak_index, detected_peaks_value in zip(self.detected_peaks_indices, self.detected_peaks_values):

            try:
                last_qrs_index = self.qrs_peaks_indices[-1]
            except IndexError:
                last_qrs_index = 0

            # After a valid QRS complex detection, there is a 200 ms refractory period before next one can be detected.
            if detected_peak_index - last_qrs_index > self.refractory_period or not self.qrs_peaks_indices.size:
                # Peak must be classified either as a noise peak or a QRS peak.
                # To be classified as a QRS peak it must exceed dynamically set threshold value.
                if detected_peaks_value > self.threshold_value:
                    self.qrs_peaks_indices = np.append(self.qrs_peaks_indices, detected_peak_index)

                    # Adjust QRS peak value used later for setting QRS-noise threshold.
                    self.qrs_peak_value = self.qrs_peak_filtering_factor * detected_peaks_value + \
                                          (1 - self.qrs_peak_filtering_factor) * self.qrs_peak_value
                else:
                    self.noise_peaks_indices = np.append(self.noise_peaks_indices, detected_peak_index)

                    # Adjust noise peak value used later for setting QRS-noise threshold.
                    self.noise_peak_value = self.noise_peak_filtering_factor * detected_peaks_value + \
                                            (1 - self.noise_peak_filtering_factor) * self.noise_peak_value

                # Adjust QRS-noise threshold value based on previously detected QRS or noise peaks value.
                self.threshold_value = self.noise_peak_value + \
                                       self.qrs_noise_diff_weight * (self.qrs_peak_value - self.noise_peak_value)

        # Create array containing both input ECG measurements data and QRS detection indication column.
        # We mark QRS detection with '1' flag in 'qrs_detected' log column ('0' otherwise).
        measurement_qrs_detection_flag = np.zeros([len(self.ecg_data_raw[:, 1]), 1])
        #print(measurement_qrs_detection_flag)
        measurement_qrs_detection_flag[self.qrs_peaks_indices] = 1
        self.ecg_data_detected = np.append(self.ecg_data_raw, measurement_qrs_detection_flag, 1)
        #print("self.ecg_data_detected")
        #print(self.ecg_data_detected)

    """Results reporting methods."""

    def print_detection_data(self):
        """
        Method responsible for printing the results.
        """
        print("qrs peaks indices")
        print(self.qrs_peaks_indices)
        print("noise peaks indices")
        print(self.noise_peaks_indices)

    def log_detection_data(self):
        """
        Method responsible for logging measured ECG and detection results to a file.
        """
        with open(self.log_path, "wb") as fin:
            fin.write(b"timestamp,ecg_measurement,qrs_detected\n")
            np.savetxt(fin, self.ecg_data_detected, delimiter=",")

    def plot_detection_data(self, show_plot=False):

        def plot_data(axis, data, title='', fontsize=10):
            axis.set_title(title, fontsize=fontsize)
            axis.grid(which='both', axis='both', linestyle='--')
            axis.plot(data, color="salmon", zorder=1)

        def plot_points(axis, values, indices):
            axis.scatter(x=indices, y=values[indices], c="black", s=50, zorder=2)

        plt.close('all')
        fig, axarr = plt.subplots(6, sharex=True, figsize=(15, 18))

        plot_data(axis=axarr[0], data=self.ecg_data_raw[:, 1], title='Raw ECG measurements')
        plot_data(axis=axarr[1], data=self.filtered_ecg_measurements, title='Filtered ECG measurements')
        plot_data(axis=axarr[2], data=self.differentiated_ecg_measurements, title='Differentiated ECG measurements')
        plot_data(axis=axarr[3], data=self.squared_ecg_measurements, title='Squared ECG measurements')
        plot_data(axis=axarr[4], data=self.integrated_ecg_measurements, title='Integrated ECG measurements with QRS peaks marked (black)')
        plot_points(axis=axarr[4], values=self.integrated_ecg_measurements, indices=self.qrs_peaks_indices)
        plot_data(axis=axarr[5], data=self.ecg_data_detected[:, 1], title='Raw ECG measurements with QRS peaks marked (black)')
        plot_points(axis=axarr[5], values=self.ecg_data_detected[:, 1], indices=self.qrs_peaks_indices)

        plt.tight_layout()
        fig.savefig(self.plot_path)

        if show_plot:
            plt.show()

        plt.close()

    """Tools methods."""

    def bandpass_filter(self, data, lowcut, highcut, signal_freq, filter_order):

        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

    def findpeaks(self, data, spacing=1, limit=None):

        len = data.size
        x = np.zeros(len + 2 * spacing)
        x[:spacing] = data[0] - 1.e-6
        x[-spacing:] = data[-1] - 1.e-6
        x[spacing:spacing + len] = data
        peak_candidate = np.zeros(len)
        peak_candidate[:] = True
        for s in range(spacing):
            start = spacing - s - 1
            h_b = x[start: start + len]  # before

            start = spacing
            h_c = x[start: start + len]  # central
            start = spacing + s + 1
            h_a = x[start: start + len]  # after
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

        ind = np.argwhere(peak_candidate)
        ind = ind.reshape(ind.size)
        if limit is not None:
            ind = ind[data[ind] > limit]
        return ind


if __name__ == "__main__":
    qrs_detector = QRSDetectorOffline(ecg_data_path="ecg_data/ecg_data_1.csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=True)
