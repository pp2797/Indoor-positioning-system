import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from beacontools import BeaconScanner, IBeaconFilter, IBeaconAdvertisement
import matplotlib.pyplot as plt

# Callback function to handle beacon data
def callback(bt_addr, rssi, packet, additional_info):
    print("<%s, %d> %s %s" % (bt_addr, rssi, packet, additional_info))

# Scan for iBeacon advertisements from beacons with the specified UUID
scanner = BeaconScanner(callback, device_filter=IBeaconFilter(uuid="e5b9e3a6-27e2-4c36-a257-7698da5fc140"))
scanner.start()




# Preprocess the collected data
def preprocess_data(data):
    # Convert data to a numpy array
    data = np.array(data)

    # Standardize the features to have mean=0 and variance=1
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)

    return standardized_data

# Collect data from BLE beacons
data = collect_data()


# Preprocess the collected data
processed_data, labels = preprocess_data(data)
# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(processed_data, labels, test_size=0.2)

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)  # Change this to match the number of values you're predicting
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
model.fit(train_data, train_labels, epochs=10)

# Evaluate the model
test_loss = model.evaluate(test_data, test_labels)

print(f'Test Loss: {test_loss}')

# Predict the locations
predicted_locations = model.predict(test_data)

# Plot the actual vs predicted locations
beacon_positions = {
    'beacon1': (1, 0),
    'beacon2': (1, 5),
    'beacon3': (5, 5),
}

# Create a scatter plot of the beacon positions
for beacon, position in beacon_positions.items():
    plt.scatter(*position, label=beacon)

# Add labels and a legend
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.scatter(test_labels[:, 0], test_labels[:, 1], color='b', label='Position')
plt.legend()
plt.show()
