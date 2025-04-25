import glob
from pathlib import Path
import subprocess
import cv2
import time
import threading
import os
from data_collection.config import BaseConfig as bc
import shutil

# Define the video devices
devices = ['/dev/CAM_LEFT']
#devices = ['/dev/video0', '/dev/video2', '/dev/video4']
class LogitechCamController:
    def __init__(self):
        self.cams = []
        for cam in bc.LOGITECH_CAM_NAMES:
            device = f"/dev/{cam}"
            cap = self.initialize_cam(device)
            self.cams.append({cam:cap})
            # image_dir = f"{bc.IMAGE_DUMP}/{cam}"
            # if os.path.exists(image_dir):
            #     shutil.rmtree(image_dir)
            #     print(f"Folder '{image_dir}' already existed. Deleted it.")

            # # Now create the new folder
            # os.makedirs(image_dir)
    # Function to set camera parameters using the bash script
    def set_camera_parameters(self,device):
        subprocess.run(["/home/simon/interbotix_ws/src/aloha/data_collection/cams/set_v4l2_settings.sh", device], check=True)

    # Function to create a GStreamer pipeline for a given device
    def create_gst_pipeline(self,device):
        return (
            f"v4l2src device={device} ! "
            "image/jpeg, width=1280, height=720, framerate=60/1 ! "
            "jpegdec ! "
            "videoconvert ! "
            "appsink"
        )

    def initialize_cam(self, device):
        self.set_camera_parameters(device)

        # Create GStreamer pipeline
        gst_pipeline = self.create_gst_pipeline(device)

        # Initialize video capture
        #print(f"Trying to open {device} with pipeline: {gst_pipeline}")
        #print(cv2.getBuildInformation())
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            print(f"Failed to open camera {device}.")
            raise ImportError
            return

        # Verify current settings
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        #print(f"Camera {device} settings -> Resolution: {int(width)}x{int(height)}, FPS: {fps:.2f}")

        return cap
    # Function to capture video from a given device
    def capture_video(self,cap, cam_name):

        start_time = time.time()
        frame_count = 0.0
        bc.STOPEVENT.clear()
        print(bc.STOPEVENT.is_set())
        while not bc.STOPEVENT.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to grab frame.")
                raise FileNotFoundError
                
            img = self.crop_img_and_inform(frame, cam_name)
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                print(f" FPS: {frame_count / elapsed_time:.2f}")
                frame_count = 0
                start_time = time.time()

            #cv2.imshow(cam_name, img)
            #cv2.imwrite(f"{bc.IMAGE_DUMP}/{cam_name}/{time.time()}.jpg", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        #cv2.destroyWindow("")
    def crop_img_and_inform(self,img, cam_name):
        img = img
        if cam_name == 'CAM_TOP':
            # img = img[:350,50:500,:]#[80:,50:630,:] #[:,:,:]
            # img = img[50:690, 260:900:, :]
            img = img[150:690, 310:850:, :]
            # img=cv2.resize(img, (420, 340))
            img=cv2.resize(img, (224, 224))
            ts=time.time()
            stamp = {"time_stamp":ts, "frame":img}
            bc.top_cam.append(stamp)
            bc.NEW_IMAGES_TOP = True
        elif cam_name == 'CAM_LEFT':
            img = img[:,:,:]#[:,:,:]
            img=cv2.resize(img, (224, 224))
            ts=time.time()
            stamp = {"time_stamp":ts, "frame":img}
            bc.left_cam.append(stamp)
            bc.NEW_IMAGE_LEFT = True
        elif cam_name == 'CAM_RIGHT':
            img = img[:,:,:]#[:,:,:]
            img=cv2.resize(img, (224, 224))
            ts=time.time()
            stamp = {"time_stamp":ts, "frame":img}
            bc.right_cam.append(stamp)
            bc.NEW_IMAGE_RIGHT = True
        else:
            raise NotImplementedError
        return img

    def start_capture(self):
        # Create and start threads for each camera
        threads = []
        for cam in self.cams:
            cam_name = str(list(cam.keys())[0])
            thread = threading.Thread(target=self.capture_video, args=(list(cam.values())[0], cam_name))
            thread.start()
            threads.append(thread)

        # Wait for all threads to finish
        return threads

        #cv2.destroyAllWindows()



"""synchronise images from folder with leader timesteps"""

def map_images(reference_times, path):
        images=[]
        for cam in bc.LOGITECH_CAM_NAMES:
            if cam == 'CAM_TOP':
                images.append({"name": cam, "images":bc.top_cam})
            elif cam == 'CAM_LEFT':
                images.append({"name": cam, "images":bc.left_cam})
            elif cam == 'CAM_RIGHT':
                images.append({"name": cam, "images":bc.x})
        find_closest_images_before(images, reference_times, path)

def get_sorted_images(folder_path):
    image_paths = glob.glob(os.path.join(Path(folder_path), "*.jpg"))
    
    # Extract timestamps from filenames
    image_timestamps = []
    for path in image_paths:
        
        try:
            ts = os.path.basename(path)
            ts = os.path.splitext(ts)[0]
            ts = float(ts)
            image_timestamps.append((ts, path))
        except ValueError:
            continue  # Skip files that don't match expected float format
    
    # Sort images by timestamp
   
    return sorted(image_timestamps, key=lambda x: x[0])


# def find_closest_images_before(images: list, reference_timestamps, img_dir):
#     """
#     For each reference timestamp, find the closest image before itsn
#     Images is a dictionary
#     It includes dictionaries like this : {"name": cam, "images":bc.top_cam}
    
#     stamp: {"name":ts, "frame":img}

#     """
#     for ref_time in reference_timestamps:
#         closest = None
#         for elem in images:
#             closest = None
#             for stamp in elem["images"]:
#                 if stamp["name"] <= ref_time:
#                     closest = stamp
#                 else:
#                     break
#             name = elem["name"]
#             dir = f"{img_dir}/{name}_orig/"
#             cv2.imwrite(dir + str(closest["name"]) + ".jpg", closest["frame"])
                
#                 # with open(destination, 'w') as f:
#                 #     f.close()


def find_closest_images_before(images: list, reference_timestamps, img_dir):
    """
    For each reference timestamp, find the closest image before it.
    
    Parameters:
        images: List of dictionaries. Each dictionary has:
                - "name": camera name (str)
                - "images": list of image dicts, each with:
                    - "time_stamp": timestamp (comparable)
                    - "frame": image frame (as used by cv2)
                    
        reference_timestamps: List of reference timestamps to match against.
        img_dir: Output directory path to save matched frames.
    """
    
    for elem in images:
        cam_name = elem["name"]
        image_list = sorted(elem["images"], key=lambda x: x["name"])  # Ensure sorted by timestamp
        dir_path = os.path.join(img_dir, f"{cam_name}_orig")
        os.makedirs(dir_path, exist_ok=True)

        for ref_time in reference_timestamps:
            closest = None
            for stamp in image_list:
                if stamp["time_stamp"] <= ref_time:
                    closest = stamp
                else:
                    break  # Since sorted, no need to go further

            if closest is not None:
                filename = os.path.join(dir_path, f"{closest['time_stamp']}.jpg")
                cv2.imwrite(filename, closest["frame"])

                
def get_last_img():
    images = {}  
    test_t = time.time()
    images["images_top"] = cv2.cvtColor(bc.top_cam[-1]["frame"], cv2.COLOR_RGB2BGR)
    images["images_wrist_left"] = cv2.cvtColor(bc.left_cam[-1]["frame"], cv2.COLOR_RGB2BGR)
    images["images_wrist_right"] = cv2.cvtColor(bc.right_cam[-1]["frame"], cv2.COLOR_RGB2BGR)
    print("\n\n\n img_collection_time: " + str((time.time() - test_t)))
    return images

if __name__ == "__main__":
    controller = LogitechCamController()
    controller.start_capture()