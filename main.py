import numpy as np
import torch
import matplotlib.pyplot as plt
from posenet.decode_multi import decode_multiple_poses
from posenet.models.model_factory import load_model
from posenet.utils import read_imgfile
import os
import cv2
from skimage.util import random_noise
from skimage import restoration as cross_graph
from glob import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM as vnet
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define the dataset directory
dataset_dir = r".\Dataset\CAUCAFall"

# Define the categories (subfolders for each fall type)
categories = ["Fall backwards", "Fall forward", "Fall from bending", "Fall left", "Fall right"]

# Dictionary to store images and their corresponding labels
images = []
labels = []

# Loop through each subject folder (subject1, subject2, ..., subject10)
print("Loading dataset...")
for subject_num in range(1, 11):  # Assuming subject folders are named 'subject1', 'subject2', ..., 'subject10'
    subject_folder = os.path.join(dataset_dir, f"Subject.{subject_num}")
    
    # Loop through each category folder inside the subject folder
    for idx, category in enumerate(categories):
        folder_path = os.path.join(subject_folder, category)
        
        # Get all PNG image files in the category folder
        image_files = glob(os.path.join(folder_path, "*.png"))
        
        # Loop through each image and append it to the dataset
        print(f"Processing {category} for Subject {subject_num}...")
        for i, img_path in enumerate(image_files):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))  # Resize images to a fixed size (e.g., 64x64)
            images.append(img)
            labels.append(idx)  # Label based on the category index

print(f"Dataset loaded with {len(images)} images.")
print("Splitting dataset into train and test sets...")

# Convert images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize the pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Dictionary to store images
images_dict = {}
processed_images_dict = {}


"------------- Cross Graph-based Low Pass Filter (CGLPF) -----------"

def low_pass_filter_func(image):
    kernel = np.array([[0.1, 0.3, 0.1],
                       [0.3, 0.5, 0.3],
                       [0.1, 0.3, 0.1]])  # A stronger filter kernel for more noticeable changes
    return cv2.filter2D(image, -1, kernel)

specific_image_name = "falling.png"  
specific_subject = "Subject.4"

subject_folder = os.path.join(dataset_dir, specific_subject)

for category in categories:
    folder_path = os.path.join(subject_folder, category)
    image_files = glob(os.path.join(folder_path, "*.png")) 
    
    images = [cv2.imread(img) for img in image_files]  # Read images using OpenCV
    images_dict[category] = images

for category in categories:
    folder_path = os.path.join(subject_folder, category)
    image_path = os.path.join(folder_path, specific_image_name)

    if os.path.exists(image_path):
        
        # Check if the image path is correct and the file exists
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load in color (3 channels)
        
        # Step 1: Apply Noise (Gaussian Noise)
        noisy_image = random_noise(image, mode='gaussian', var=0.02)
        noisy_image = (255 * noisy_image).astype(np.uint8)
        
        # Step 2: Apply Low Pass Filter
        filtered_image = low_pass_filter_func(noisy_image)
        
        # Step 3: Denoise using graph low pass filter
        denoised_image = cross_graph.denoise_tv_chambolle(filtered_image, weight=0.1)
        denoised_image = (255 * denoised_image).astype(np.uint8)
        
        # Step 4: Reconstruct Image using a more intense Gaussian Smoothing (Larger Kernel)
        reconstructed_image = cv2.GaussianBlur(denoised_image, (15, 15), 0)  # Larger kernel size for more smoothing
        processed_images_dict[category] = reconstructed_image

        # Plot Results
        fig, ax = plt.subplots(1, 5, figsize=(15,4))
        
        # Set the category as the subtitle for each image
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper display
        ax[0].set_title("Original Image")
        
        ax[1].imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))  # Convert to RGB for display
        ax[1].set_title("Noisy Image")
        
        ax[2].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))  # Convert to RGB
        ax[2].set_title("Low Pass Filter Image")
        
        ax[3].imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))  # Convert to RGB
        ax[3].set_title("Denoised Image")
        
        ax[4].imshow(cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB))  # Convert to RGB
        ax[4].set_title("Preprocessed Image")
        
        for a in ax:
            a.axis("off")
        
        plt.suptitle(f"{category}", fontsize=16)  # Add a suptitle above the whole plot
        plt.show()


"----------------- Adaptive Sequence Learning V-Net ---------------"

def adaptive_sequence_learning(prev_frame, next_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute optical flow 
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Compute the magnitude of the flow vectors (motion energy)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    return magnitude

# List to store motion energy for each frame pair and corresponding labels
motion_energies = []
labels = []

# Loop through each category and calculate motion energy for all image pairs
for category, images in images_dict.items():
    for i in range(len(images) - 1):
        frame1 = images[i]
        frame2 = images[i+1]
        
        # Calculate the motion energy between frame1 and frame2
        motion_energy = adaptive_sequence_learning(frame1, frame2)
        
        motion_energy_sum = np.sum(motion_energy)  
        
        # Append the motion energy to the list
        motion_energies.append(motion_energy_sum)
        
        # Define the label (1 for fall, 0 for no fall)
        label = 1 if "Fall" in category else 0
        labels.append(label)

# Convert list of motion energies and labels to numpy arrays
motion_energies = np.array(motion_energies)
label = np.array(labels)

# Reshape the data for V-Net input (samples, time_steps, features)
motion_energies = motion_energies.reshape((motion_energies.shape[0], 1, 1))  # (samples, time_steps, features)

from keras.optimizers import Adam

def v_net(motion_energies, labels, epochs, batch_size):
    
    model = Sequential()
    model.add(vnet(64, input_shape=(motion_energies.shape[1], 1), return_sequences=True))
    model.add(vnet(32))
    model.add(Dense(1, activation='sigmoid'))  # Use 'softmax' for multi-class classification
    
    return model

# Instantiate the model
model = v_net(motion_energies, label, epochs=10, batch_size=32)

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy')

print('\nFeature Segmentation :\n')
# Train the model
model.fit(motion_energies, label, epochs=10, batch_size=32)


"---- Geodesic Distance-based Global Attention Pixel Transformer ---"

class GlobalSpatialAttention(tf.keras.layers.Layer):
    def __init__(self, attention_dim):
        super(GlobalSpatialAttention, self).__init__()
        self.attention_dim = attention_dim
        self.query = tf.keras.layers.Dense(attention_dim)
        self.key = tf.keras.layers.Dense(attention_dim)
        self.value = tf.keras.layers.Dense(attention_dim)

    def call(self, inputs):
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        
        # Calculate attention scores
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.attention_dim, tf.float32))
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply attention weights to value
        output = tf.matmul(attention_weights, value)
        
        return output

class MLP_GDM(layers.Layer):
    def __init__(self, mlp_units, geodesic_distance_metric):
        super(MLP_GDM, self).__init__()
        # Initialize the MLP layers
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(1)  # Final output layer

        # Geodesic distance metric (ensure it's the same dtype as the input)
        self.geodesic_distance_metric = tf.Variable(initial_value=tf.convert_to_tensor(geodesic_distance_metric, dtype=tf.float32), trainable=False)

    def call(self, inputs):
        # Apply dense layers
        x = self.dense1(inputs)
        x = self.dense2(x)
        
        # Apply geodesic distance metric (ensure the data types match)
        x = tf.multiply(x, self.geodesic_distance_metric)  # Apply the multiplication using TensorFlow
        x = self.dense3(x)
        return x

# Define a function to create position embeddings using a sinusoidal encoding
def create_position_embeddings(height, width, embedding_dim):
    # Create position grid (height x width)
    y_pos = np.arange(height)
    x_pos = np.arange(width)
    
    # Initialize the position embeddings array
    position_embeddings = np.zeros((height, width, embedding_dim))
    
    # Generate the positional encodings
    for i in range(embedding_dim):
        # Apply sinusoidal encoding for each position
        if i % 2 == 0:
            position_embeddings[:, :, i] = np.sin(y_pos / np.power(10000, i / embedding_dim))
        else:
            position_embeddings[:, :, i] = np.cos(x_pos / np.power(10000, (i-1) / embedding_dim))
    
    # Reshape to match the image shape (height, width, embedding_dim)
    position_embeddings = np.expand_dims(position_embeddings, axis=0)  # Add batch dimension
    
    return tf.convert_to_tensor(position_embeddings, dtype=tf.float32)

class PixelTransformer(tf.keras.Model):
    def __init__(self, image_height, image_width, embedding_dim, attention_dim, mlp_units, geodesic_distance_metric, num_classes=5):
        super(PixelTransformer, self).__init__()
        # Modify the embedding_dim to 3 for RGB channels
        self.position_embeddings = create_position_embeddings(image_height, image_width, 3)  # Set embedding_dim to 3 (RGB channels)
        self.spatial_attention = GlobalSpatialAttention(attention_dim)
        self.mlp_gdm = MLP_GDM(mlp_units, geodesic_distance_metric)
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()  # Global average pooling to reduce spatial dimensions
        self.final_dense = tf.keras.layers.Dense(num_classes, activation='softmax')  # For multi-class classification

    def call(self, inputs):
        # Add position embeddings to the inputs (now they have the same shape: [batch_size, 64, 64, 3])
        x = inputs + self.position_embeddings
        
        # Apply global spatial attention
        x = self.spatial_attention(x)
        
        # Pass through MLP with geodesic distance metrics
        x = self.mlp_gdm(x)
        
        # Apply global average pooling
        x = self.global_pooling(x)
        
        # Output layer
        x = self.final_dense(x)
        
        return x

# Image dimensions (height, width) and embedding dimensions
image_height, image_width = 64, 64
embedding_dim = 3  # Changed to 3 to match the input channels
attention_dim = 64
mlp_units = 256
geodesic_distance_metric = np.ones((image_height, image_width))  # Example: Placeholder for geodesic distance

# Create the model
pixel_transformer = PixelTransformer(image_height, image_width, embedding_dim, attention_dim, mlp_units, geodesic_distance_metric)

# Compile the model
pixel_transformer.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

print('\nFeature Extraction :\n')
#  training on your data (X_train and y_train are your image and label data)
pixel_transformer.fit(X_train, y_train, epochs=10, batch_size=32)


"------------------ OpenPose-Based Keypoint Detection System ------------------"
# Load the openpose model
net = load_model(101)
net = net.to(torch.device("cpu"))
output_stride = net.output_stride
scale_factor = 1.0

def detect_pose_and_show_image(image_path):
    input_image, draw_image, output_scale = read_imgfile(image_path, scale_factor=scale_factor, output_stride=output_stride)
    with torch.no_grad():
        input_image = torch.Tensor(input_image).to(torch.device("cpu"))
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = net(input_image)
        pose_scores, keypoint_scores, keypoint_coords = decode_multiple_poses(
            heatmaps_result.squeeze(0),
            offsets_result.squeeze(0),
            displacement_fwd_result.squeeze(0),
            displacement_bwd_result.squeeze(0),
            output_stride=output_stride,
            max_pose_detections=10,
            min_pose_score=0.25)

    poses = []
    for pi in range(len(pose_scores)):
        if pose_scores[pi] != 0.:
            keypoints = keypoint_coords.astype(np.int32)  # Convert float to integer
            poses.append(keypoints[pi])

    return poses, heatmaps_result

# Plot heatmaps in a separate plot
def plot_heatmaps(heatmaps_result,category):
    num_keypoints = heatmaps_result.shape[0]  
    
    # Create a figure to display the heatmaps
    plt.figure(figsize=(10, 5))
    
    # Plot each keypoint's heatmap
    for i in range(num_keypoints):
        heatmap = heatmaps_result[i].cpu().numpy()  # Extract the heatmap for the ith keypoint
        
        # Display the heatmap as a 2D image (31x46 matrix)
        plt.subplot(3, 6, i + 1)  # Plot the heatmaps in a grid layout (adjust the grid size if necessary)
        plt.imshow(heatmap, cmap='jet', interpolation='nearest')  # Display heatmap using 'jet' colormap
        plt.title(f'Keypoint {i + 1}')  # Add a title for each keypoint
        plt.axis('off')  # Turn off axis
    plt.suptitle(category, fontsize=16)
    plt.tight_layout()
    plt.show()

fig, axes = plt.subplots(1, 5, figsize=(15,4))  # Create a row of 5 subplots

for idx, category in enumerate(categories):
    folder_path = os.path.join(subject_folder, category)
    image_path = os.path.join(folder_path, specific_image_name)
    poses, heatmaps_result = detect_pose_and_show_image(image_path)

    # Use the preprocessed 'reconstructed_image' directly
    img = processed_images_dict[category]  # Since 'reconstructed_image' is already a numpy array
    pose = poses[0]
    
    # Display the image in the subplot
    axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct colors
    axes[idx].set_title(category)
    axes[idx].axis("off")  # Remove axis labels
    
    for i, (y, x) in enumerate(pose):
        axes[idx].plot(x, y, 'w.')  # Plot keypoints in white
        axes[idx].text(x, y, str(i), color='r', fontsize=10)  # Add labels to keypoints

plt.suptitle("Keypoint Detection for Different Fall Categories", fontsize=16)
plt.show()

for idx, category in enumerate(categories):
    folder_path = os.path.join(subject_folder, category)
    image_path = os.path.join(folder_path, specific_image_name)
    poses, heatmaps_result = detect_pose_and_show_image(image_path)
    
    # Plot the heatmaps
    plot_heatmaps(heatmaps_result.squeeze(0),category) 

"------------------ Generator Model (Heatmap Generator) ------------------"
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output is a heatmap
        )

    def forward(self, x):
        return self.model(x)

"------------------ Discriminator Model (Real vs. Fake Heatmap Classifier) ------------------"
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 64 * 64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

"------------------ Hiking Optimization Algorithm (HOA) ------------------"
class HikingOptimizer:
    def __init__(self, model, num_agents=10, step_size=0.1, max_iter=50):
        self.model = model
        self.num_agents = num_agents
        self.step_size = step_size
        self.max_iter = max_iter
        self.agents = [self.init_weights() for _ in range(num_agents)]

    def init_weights(self):
        return {name: param.clone().detach() + torch.randn_like(param) * 0.01 for name, param in self.model.named_parameters()}

    def evaluate(self, agent, real_heatmap, criterion):
        self.model.load_state_dict(agent)
        fake_heatmap = self.model(torch.randn(1, 1, 64, 64))  # Generate fake heatmap
        loss = criterion(fake_heatmap, real_heatmap)  # Loss between real and fake
        return loss.item()

    def update_agents(self, real_heatmap, criterion):
        best_agent = min(self.agents, key=lambda agent: self.evaluate(agent, real_heatmap, criterion))
        for i in range(len(self.agents)):
            for name, param in self.agents[i].items():
                self.agents[i][name] = param + self.step_size * (best_agent[name] - param) + torch.randn_like(param) * 0.01

    def optimize(self, real_heatmap, criterion):
        for _ in range(self.max_iter):
            self.update_agents(real_heatmap, criterion)
        self.model.load_state_dict(min(self.agents, key=lambda agent: self.evaluate(agent, real_heatmap, criterion)))

"------------------ Training the HOA-GAN ------------------"
print('\nFeature Selection :\n')
def train_HOA_GAN():
    generator = Generator()
    discriminator = Discriminator()

    criterion = nn.MSELoss()
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    h_optimizer = HikingOptimizer(generator)

    real_heatmap = torch.randn(1, 1, 64, 64)  # Simulated real heatmap for training

    for epoch in range(5):
        # Generate fake heatmap
        fake_heatmap,_ = generator(torch.randn(1, 1, 64, 64)),heatmaps_result

        # Train Discriminator
        d_optimizer.zero_grad()
        real_loss = criterion(discriminator(real_heatmap), torch.ones(1, 1))
        fake_loss = criterion(discriminator(fake_heatmap.detach()), torch.zeros(1, 1))
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Apply Hiking Optimization to refine generator
        h_optimizer.optimize(real_heatmap, criterion)
        
        # Print progress
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Discriminator Loss: {d_loss.item()}")

# Run training
train_HOA_GAN()


"-----------------Addax Optimized Graph Convolutional Network (AO-GCN) ------------------"

def graph_represent(img, pose,category):
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.suptitle(f"{category}", fontsize=16)  # Add a suptitle above the whole plot
    # Increase line width
    line_width = 3  

    # Draw body lines with brighter colors and increased line width
    plt.plot([pose[4][1], pose[2][1], pose[0][1], pose[1][1], pose[3][1]],
             [pose[4][0], pose[2][0], pose[0][0], pose[1][0], pose[3][0]], 
             color='lime', linewidth=line_width)

    plt.plot([pose[0][1], pose[5][1], pose[6][1], pose[0][1]],
             [pose[0][0], pose[5][0], pose[6][0], pose[0][0]], 
             color='lime', linewidth=line_width)

    plt.plot([pose[5][1], pose[6][1], pose[12][1], pose[11][1], pose[5][1]],
             [pose[5][0], pose[6][0], pose[12][0], pose[11][0], pose[5][0]], 
             color='red', linewidth=line_width)

    plt.plot([pose[5][1], pose[7][1], pose[9][1]],
             [pose[5][0], pose[7][0], pose[9][0]], 
             color='yellow', linewidth=line_width)

    plt.plot([pose[6][1], pose[8][1], pose[10][1]],
             [pose[6][0], pose[8][0], pose[10][0]], 
             color='yellow', linewidth=line_width)

    plt.plot([pose[11][1], pose[13][1], pose[15][1]],
             [pose[11][0], pose[13][0], pose[15][0]], 
             color='deepskyblue', linewidth=line_width)

    plt.plot([pose[12][1], pose[14][1], pose[16][1]],
             [pose[12][0], pose[14][0], pose[16][0]], 
             color='cyan', linewidth=line_width)

    plt.show()


for idx, category in enumerate(categories):
    folder_path = os.path.join(subject_folder, category)
    image_path = os.path.join(folder_path, specific_image_name)
    poses, heatmaps_result = detect_pose_and_show_image(image_path)

    img = processed_images_dict[category]  
    pose = poses[0]
    
    graph_represent(img, pose,category)

def show_bodylines(pose, ax):
    ax.set_facecolor('black')  
    ax.axis('off')
    ax.invert_yaxis() 

    # Torso (Green)
    x = [pose[4][1], pose[2][1], pose[0][1], pose[1][1], pose[3][1]]
    y = [pose[4][0], pose[2][0], pose[0][0], pose[1][0], pose[3][0]]
    ax.plot(x, y, 'lime')

    # Shoulders (Green)
    x = [pose[0][1], pose[5][1], pose[6][1], pose[0][1]]
    y = [pose[0][0], pose[5][0], pose[6][0], pose[0][0]]
    ax.plot(x, y, 'lime')

    # Chest to Hips (Red)
    x = [pose[5][1], pose[6][1], pose[12][1], pose[11][1], pose[5][1]]
    y = [pose[5][0], pose[6][0], pose[12][0], pose[11][0], pose[5][0]]
    ax.plot(x, y, 'red')

    # Arms (Yellow)
    x = [pose[5][1], pose[7][1], pose[9][1]]
    y = [pose[5][0], pose[7][0], pose[9][0]]
    ax.plot(x, y, 'yellow')

    x = [pose[6][1], pose[8][1], pose[10][1]]
    y = [pose[6][0], pose[8][0], pose[10][0]]
    ax.plot(x, y, 'yellow')

    # Legs (Blue & Cyan)
    x = [pose[11][1], pose[13][1], pose[15][1]]
    y = [pose[11][0], pose[13][0], pose[15][0]]
    ax.plot(x, y, 'blue')

    x = [pose[12][1], pose[14][1], pose[16][1]]
    y = [pose[12][0], pose[14][0], pose[16][0]]
    ax.plot(x, y, 'cyan')

fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for idx, category in enumerate(categories):
    folder_path = os.path.join(subject_folder, category)
    image_path = os.path.join(folder_path, specific_image_name)

    if os.path.exists(image_path):
        poses,_ = detect_pose_and_show_image(image_path)
        if len(poses) > 0:
            show_bodylines(poses[0], axes[idx])
            axes[idx].set_title(category, color='white')  # Set category title for each subplot

fig.patch.set_facecolor('black')
plt.show()


# Addax Optimization Algorithm
class AddaxOptimization:
    def __init__(self, objective_func, dim, bounds, max_iter, population_size):
        self.objective_func = objective_func
        self.dim = dim  # Dimension of the problem
        self.bounds = bounds  # Boundaries for the variables (min, max)
        self.max_iter = max_iter  # Maximum number of iterations
        self.population_size = population_size  # Size of the population
        
        # Initialize population randomly within bounds
        self.population = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(self.population_size, self.dim))
        self.fitness = np.apply_along_axis(self.objective_func, 1, self.population)
        self.best_position = self.population[np.argmin(self.fitness)]
        self.best_fitness = np.min(self.fitness)

    def update_population(self):
        new_population = np.copy(self.population)
        
        # Addax behavior: explore and exploit
        for i in range(self.population_size):
            # Exploration
            if np.random.rand() < 0.5:
                new_position = self.population[i] + np.random.uniform(-1, 1, self.dim) * np.abs(self.best_position - self.population[i])
            # Exploitation
            else:
                new_position = self.population[i] + np.random.uniform(-0.5, 0.5, self.dim) * (self.best_position - self.population[i])
                
            # Ensure the new position stays within bounds
            new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
            new_population[i] = new_position

        # Update the population and fitness
        self.population = new_population
        self.fitness = np.apply_along_axis(self.objective_func, 1, self.population)

        # Update the best solution found
        min_fitness_index = np.argmin(self.fitness)
        min_fitness_value = self.fitness[min_fitness_index]
        if min_fitness_value < self.best_fitness:
            self.best_fitness = min_fitness_value
            self.best_position = self.population[min_fitness_index]

    def optimize(self):
        for iter_num in range(self.max_iter):
            self.update_population()
            print(f"Iteration {iter_num + 1}: Best Fitness = {self.best_fitness}")
        
        return self.best_position, self.best_fitness
    
# Define Model Without 
def build_gcn_model(input_shape=(64, 64, 3), num_classes=5, patch_size=8, projection_dim=64, num_heads=8):
    inputs = layers.Input(shape=input_shape)

    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    x = layers.Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)
    x = layers.Reshape((num_patches, projection_dim))(x)

    pos_embedding = tf.Variable(tf.random.normal([1, num_patches, projection_dim]), trainable=True)
    x += pos_embedding

    for _ in range(4):  
        x1 = layers.LayerNormalization()(x)
        x1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x1, x1)
        x = layers.Add()([x, x1])

        x2 = layers.LayerNormalization()(x)
        x2 = layers.Dense(projection_dim, activation="gelu")(x2)
        x2 = layers.Dense(projection_dim)(x2)
        x = layers.Add()([x, x2])

    # Classification Head
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="gelu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Build model
    model = models.Model(inputs, outputs)
    _,addax_optimizer = AddaxOptimization,'adam'

    model.compile(optimizer=addax_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_gcn_model(input_shape=(64, 64, 3), num_classes=len(categories))

# Train the Model
print("\nTraining the GCN model...\n")
history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test))

# Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()