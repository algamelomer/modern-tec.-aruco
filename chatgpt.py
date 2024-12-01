from enum import Enum
import cv2
import numpy as np 
import os 
import matplotlib.pyplot as plt 

class ArucoType(Enum):
    DICT_4X4_50 = cv2.aruco.DICT_4X4_50
    DICT_4X4_100 = cv2.aruco.DICT_4X4_100
    DICT_4X4_250 = cv2.aruco.DICT_4X4_250
    DICT_4X4_1000 = cv2.aruco.DICT_4X4_1000
    DICT_5X5_50 = cv2.aruco.DICT_5X5_50
    DICT_5X5_100 = cv2.aruco.DICT_5X5_100
    DICT_5X5_250 = cv2.aruco.DICT_5X5_250
    DICT_5X5_1000 = cv2.aruco.DICT_5X5_1000
    DICT_6X6_50 = cv2.aruco.DICT_6X6_50
    DICT_6X6_100 = cv2.aruco.DICT_6X6_100
    DICT_6X6_250 = cv2.aruco.DICT_6X6_250
    DICT_6X6_1000 = cv2.aruco.DICT_6X6_1000
    DICT_7X7_50 = cv2.aruco.DICT_7X7_50
    DICT_7X7_100 = cv2.aruco.DICT_7X7_100
    DICT_7X7_250 = cv2.aruco.DICT_7X7_250
    DICT_7X7_1000 = cv2.aruco.DICT_7X7_1000
    DICT_ARUCO_ORIGINAL = cv2.aruco.DICT_ARUCO_ORIGINAL
    DICT_APRILTAG_16h5 = cv2.aruco.DICT_APRILTAG_16h5
    DICT_APRILTAG_25h9 = cv2.aruco.DICT_APRILTAG_25h9
    DICT_APRILTAG_36h10 = cv2.aruco.DICT_APRILTAG_36h10
    DICT_APRILTAG_36h11 = cv2.aruco.DICT_APRILTAG_36h11

class ArucoMarkers(): 
    def __init__(self): 
        self.dir = os.path.dirname(os.path.abspath(__file__))

    # Generate multiple ArUco markers and save them
    def generate_multiple_aruco_markers(self, aruco_type, start_id, num_markers, marker_width_pixels):
        print(f'Generating {num_markers} ArUco Markers...')
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_type.value)

        for marker_id in range(start_id, start_id + num_markers):
            marker_image = np.zeros((marker_width_pixels, marker_width_pixels), dtype=np.uint8)
            marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_width_pixels, marker_image, 1)

            border_width = 5
            marker_image = cv2.copyMakeBorder(
                marker_image,
                border_width, border_width, border_width, border_width,
                cv2.BORDER_CONSTANT,
                value=[255,255,255]
            )

            # Save each ArUco marker with a unique filename
            marker_filename = os.path.join(self.dir, f'aruco_marker_{marker_id}.png')
            cv2.imwrite(marker_filename, marker_image)
            print(f'Saved marker with ID {marker_id} to {marker_filename}')

        print(f'Successfully generated {num_markers} markers.')

    # Detect multiple ArUco markers and estimate pose
        # Detect multiple ArUco markers and estimate pose
    def aruco_marker_pose_estimation(self, aruco_type, camera_matrix, dist_coeffs): 
        print('Detecting ArUco Markers...')
        cap = cv2.VideoCapture(0)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_type.value) 
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        world_points = np.array([[0., 0., 0.],  # top left
                                 [1., 0., 0.],  # top right
                                 [1., 1., 0.],  # bottom right
                                 [0., 1., 0.]])  # bottom left

        while True: 
            ret, frame = cap.read() 
            
            if not ret: 
                break 

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(frame_gray)

            if ids is not None: 
                # Draw all detected markers and their IDs on the frame
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for i, corner in enumerate(corners): 
                    # Perform pose estimation for each detected marker
                    _, rvecs, tvecs = cv2.solvePnP(world_points, corner, camera_matrix, dist_coeffs)
                    
                    # Print the translation vector (tvecs) to get the position in x, y, z
                    position = tvecs.flatten()  # x, y, z position of the marker
                    print(f"Marker ID: {ids[i][0]}")
                    print(f"Position: x = {position[0]:.2f}, y = {position[1]:.2f}, z = {position[2]:.2f}")

                    # Calculate and print the distance (Euclidean distance) to the marker
                    distance = np.linalg.norm(position)
                    print(f"Distance to marker: {distance:.2f} meters\n")
                    
                    # Draw pose axis for each marker
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, 1)

                    # Display the marker ID on the frame
                    cv2.putText(frame, f"ID: {ids[i][0]}", 
                                (int(corner[0][0][0]), int(corner[0][0][1]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Display the distance to the marker on the frame
                    cv2.putText(frame, f"Dist: {distance:.2f}m", 
                                (int(corner[0][0][0]), int(corner[0][0][1]) - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Show the frame with detected markers and their IDs
            cv2.imshow('ArUco Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        cap.release()
        cv2.destroyAllWindows()

# Run the function to generate 40 ArUco markers
def run_generate_40_aruco_markers(aruco_type, marker_width_pixels):
    aruco_marker = ArucoMarkers()
    aruco_marker.generate_multiple_aruco_markers(aruco_type, start_id=0, num_markers=40, marker_width_pixels=marker_width_pixels)

# Run the pose estimation on detected ArUco markers
def run_aruco_marker_pose_estimation(aruco_type): 
    aruco_marker = ArucoMarkers() 
    # Example camera matrix and distortion coefficients (you need to calibrate for your camera)
    camera_matrix = np.array([
        [1432.0, 0.0,    983.0], 
        [0.0,    1434.0, 561.0], 
        [0.0,    0.0,    1.0]
    ])  
    dist_coeffs = np.array([0.05994318, -0.26432366, -0.00135378, -0.00081574,  0.29707202])
    aruco_marker.aruco_marker_pose_estimation(aruco_type, camera_matrix, dist_coeffs)

if __name__ == '__main__': 
    # Uncomment this to generate 40 ArUco markers
    # run_generate_40_aruco_markers(ArucoType.DICT_6X6_250, marker_width_pixels=200)

    # Uncomment this to run ArUco marker pose estimation
    run_aruco_marker_pose_estimation(ArucoType.DICT_6X6_250)
