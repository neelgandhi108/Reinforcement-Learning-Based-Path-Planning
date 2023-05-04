# Reinforcement-Learning-Based-Path-Planning

## I. Image Processing Techniques

Uses various libraries such as cv2, matplotlib, numpy, os, and skimage.io
Functions:
show(): displays the image
overlay_mask(): overlays the mask on the original image
convex_cnt(): finds contours in the image, draws convex contours, and returns a binary mask
dilate_img(): performs image dilation on the input binary image
Image processing steps:
Reads the input image using skimage.io.imread() and resizes it using cv2.resize()
Blurs the image using cv2.GaussianBlur()
Converts the blurred image to HSV color space using cv2.cvtColor()
Defines two ranges of color values using np.array()
Thresholds the HSV image using cv2.inRange() with the defined color ranges
Combines the resulting binary images using element-wise addition
Performs morphological operations using cv2.morphologyEx() with a structuring element of elliptical shape created using cv2.getStructuringElement()
Dilates the resulting binary image using dilate_img()
Finds the contours in the dilated image, draws convex contours, and creates a binary mask using convex_cnt()
Overlays the mask on the original image using overlay_mask()
Displays the binary mask using show_mask()
## II. Proximal Policy Optimization Algorithm

Reinforcement learning algorithm that improves upon the previous Policy Gradient algorithm
Uses two neural networks:
Actor network that learns to predict an action for a given state
Critic network that learns to predict the value of a given state
Uses the advantage function to compute the advantage of taking an action in a given state over taking the average action in that state
Trains the actor and critic networks to maximize the advantage using gradient ascent
Configurable parameters:
LOAD: boolean value that determines whether to load a previously trained model or train a new one
EP_MAX: number of iterations inside the environment
EP_LEN: maximum number of steps for every iteration
GAMMA: discount factor that determines the importance of future rewards
A_LR: learning rate for the actor network
C_LR: learning rate for the critic network
BATCH: standard number of batches inside every iteration
A_UPDATE_STEPS: number of training iterations with each mini-batch for the actor
C_UPDATE_STEPS: number of training iterations with each mini-batch for the critic
Defines a CarDCENV class that represents the environment in which the agent learns:
Has 8 sensors that detect the distance between the agent and the goal
Has an action dimension of 1 and a state dimension of the number of sensors plus 1
View window is determined by the shape of the map
Has a step function that takes an action as input and returns the next state, reward, and whether the episode is terminal or not
Has a reset function that resets the state of the environment 
Has a render function that displays the current state of the environment. 

The main function of the code first configures the parameters for the PPO algorithm, and then initializes the environment using the CarDCENV class. If the LOAD variable is set to True, it loads a previously trained model using the tf.keras.models.load_model() function. Otherwise, it creates a new model using the keras.Sequential() function and trains it using the PPO algorithm.

During the training process, the agent takes actions in the environment based on the predictions of the actor network, and receives rewards based on the distance between the agent and the goal. The critic network is used to estimate the value of each state, which is used to compute the advantage function. The actor and critic networks are updated using the PPO algorithm, and the process is repeated for multiple iterations.

After the training is complete, the final model is saved using the tf.keras.models.save_model() function. The code also includes a function called evaluate() that evaluates the performance of the trained model in the environment, and returns the average reward over multiple episodes.

Overall, this code demonstrates the use of the PPO algorithm for reinforcement learning in an environment with multiple sensors and an action dimension of 1. It also demonstrates image processing techniques to obtain a binary format map from an input image file.
