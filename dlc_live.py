from pypylon import pylon
import cv2
import numpy as np
from dlclive import DLCLive, Processor
from display import Display
import time
from datetime import datetime
import pandas as pd
import keyboard  # For detecting keypresses
import serial

# Global variable for recording state
recording = False
pose_data_list = []
arduino_data = None

def toggle_recording(event):
    global recording, pose_data_list
    if event.name == 'g':
        if recording:
            # Stop recording and save data to CSV
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = f"pose_data_{timestamp}.csv"
            
            # Save pose data to CSV
            df = pd.DataFrame(pose_data_list, columns=['timestamp', 'pose', 'arduino_data'])
            df.to_csv(output_filename, index=False)
            
            print(f"Recording stopped. Data saved to {output_filename}")
            pose_data_list = []  # Reset list after saving
            recording = False
        else:
            print("Recording started")
            recording = True


def read_arduino_data(ser):
    """Read data from Arduino via serial."""
    if ser.in_waiting > 0:
        try:
            data = ser.readline().decode('utf-8').strip()  # Read and decode the string
            return data  # Return the entire string (e.g., "Left, 50Hz, 500ms")
        except Exception as e:
            print(f"Error reading from serial: {e}")
    return None

def main():
    # Initialize the camera
    global recording, pose_data_list
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    
    # Open the camera
    camera.Open()
    
    # Initialize DeepLabCut Live
    dlc_proc = Processor()
    dlc_live = DLCLive(r"C:\temp\Experiment\InclinedStimulationModel-LF-2024-09-01\exported-models\DLC_InclinedStimulationModel_resnet_50_iteration-0_shuffle-1", processor=dlc_proc)
    
    #Initialize the Display object (with a colormap, radius and cutoff)
    display = Display(cmap = "bmy", radius = 5, pcutoff = 0.5)
    
    # Start grabbing images
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    grab_result = camera.RetrieveResult(50, pylon.TimeoutHandling_ThrowException)
    image = grab_result.Array
    
    # Convert to a format suitable for DeepLabCut Live
    image = np.array(image)
    dlc_live.init_inference(image)

    # Initialize serial communication with Arduino
    ser = serial.Serial('COM3', 9600, timeout=1)  # Change 'COM3' to the correct port for your Arduino

    
    keyboard.on_press(toggle_recording)
    try:
        while camera.IsGrabbing():
            # print(time.time())
            # Wait for an image and then retrieve it
            grab_result = camera.RetrieveResult(50, pylon.TimeoutHandling_ThrowException)
            
            if grab_result.GrabSucceeded():
                # Access the image data
                image = grab_result.Array
                image = np.array(image)

                # Get pose estimation from the image
                pose = dlc_live.get_pose(image)

                # Check if recording is enabled
                if recording:
                    # Get current timestamp
                    current_time = time.time()
                    
                    #Read arduino data from serial 
                    arduino_data = read_arduino_data(ser)


                    # Save pose data and timestamp in the list
                    pose_data_list.append((current_time, pose.tolist(), arduino_data))


                display.display_frame(image, pose)

                # Press 'q' to exit the loop
                if keyboard.is_pressed('q'):
                    break

            grab_result.Release()


            
    finally:
        # Stop grabbing and release resources
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()
        display.destroy()

if __name__ == '__main__':
    main()