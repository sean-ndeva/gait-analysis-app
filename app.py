import streamlit as st
import tempfile, os
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio # User requested to keep this despite imageio.v3 import
import imageio.v3 as iio # User requested to keep this
import cv2
from io import BytesIO
from datetime import datetime
from PIL import Image

# Streamlit page configuration
st.set_page_config(page_title="Stride Buddy - AI Posture Coach", layout="wide")
st.title("👣 Stride Buddy App - AI Posture Coach")
st.write("Get real-time feedback on your exercise posture using your webcam or by uploading a video.")

# Sidebar inputs
st.sidebar.header("Optional Personal Information")
height_cm = st.sidebar.slider("Your height (cm)", 140, 210, 175)
weight_kg = st.sidebar.slider("Your weight (kg)", 40, 150, 70)
age_group = st.sidebar.selectbox("Your age group", ["18-24", "25-30", "31-40", "41-50", "51-60", "60+"])

bmi = weight_kg / ((height_cm / 100) ** 2)
st.sidebar.write(f"**Your BMI:** {bmi:.1f}")

st.sidebar.header("Customize Detection Sensitivity")
asym_threshold = st.sidebar.slider("Max acceptable angle difference (°)", 5, 30, 10)
stiff_threshold = st.sidebar.slider("Min acceptable knee flexion (°)", 90, 140, 120) # For uploaded video analysis

# Display user info
st.markdown(f"""
**User Info**
- Height: {height_cm} cm
- Weight: {weight_kg} kg
- Age group: {age_group}
- BMI: {bmi:.1f}
""")

# Reference norms by age group
age_norms = {
    "18-24": {"hip": 0.02, "arm": 5},
    "25-30": {"hip": 0.025, "arm": 6},
    "31-40": {"hip": 0.03, "arm": 7},
    "41-50": {"hip": 0.035, "arm": 8},
    "51-60": {"hip": 0.04, "arm": 9},
    "60+": {"hip": 0.045, "arm": 10}
}
norms = age_norms.get(age_group, {"hip": 0.03, "arm": 7})

# Initialize MediaPipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define Exercises and their landmark connections for angle calculation
EXERCISES = {
    "Squat": {
        "description": "Focus on keeping your back straight, chest up, and knees tracking over your toes.",
        "landmark_points": {
            "right_knee": [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
            "left_knee": [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value],
            "right_hip": [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value],
            "left_hip": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value],
        },
        "phase_detection": {
            "primary_angles_for_phase": ["left_knee", "right_knee"],
            "phases": [
                {"name": "start_position", "condition_angle_above": 155},
                {"name": "squat_descent", "condition_angle_below": 120, "condition_angle_above_min": 65}
            ]
        },
        "feedback_rules": {
            "start_position": {
                "right_knee": {"min": 160, "max": 180, "feedback": "Stand up straight!"},
                "left_knee": {"min": 160, "max": 180, "feedback": "Stand up straight!"},
                "right_hip": {"min": 160, "max": 180, "feedback": "Keep your hips extended."},
                "left_hip": {"min": 160, "max": 180, "feedback": "Keep your hips extended."},
            },
            "squat_descent": {
                "right_knee": {"min": 70, "max": 110, "feedback": "Squat deeper if comfortable."},
                "left_knee": {"min": 70, "max": 110, "feedback": "Squat deeper if comfortable."},
                "right_hip": {"min": 70, "max": 110, "feedback": "Hips back, chest up!"},
                "left_hip": {"min": 70, "max": 110, "feedback": "Hips back, chest up!"},
            },
            "general_form": {
                "knee_hip_alignment_right": {"check_type": "x_alignment", "landmark1": mp_pose.PoseLandmark.RIGHT_KNEE.value, "landmark2": mp_pose.PoseLandmark.RIGHT_HIP.value, "tolerance_body_prop": 0.05, "feedback": "Keep right knee aligned with hip."},
                "knee_hip_alignment_left": {"check_type": "x_alignment", "landmark1": mp_pose.PoseLandmark.LEFT_KNEE.value, "landmark2": mp_pose.PoseLandmark.LEFT_HIP.value, "tolerance_body_prop": 0.05, "feedback": "Keep left knee aligned with hip."},
                "right_knee_over_foot_horizontal": {"check_type": "x_alignment", "landmark1": mp_pose.PoseLandmark.RIGHT_KNEE.value, "landmark2": mp_pose.PoseLandmark.RIGHT_ANKLE.value, "tolerance_body_prop": 0.08, "feedback": "Right knee too far forward/backward over foot."},
                "left_knee_over_foot_horizontal": {"check_type": "x_alignment", "landmark1": mp_pose.PoseLandmark.LEFT_KNEE.value, "landmark2": mp_pose.PoseLandmark.LEFT_ANKLE.value, "tolerance_body_prop": 0.08, "feedback": "Left knee too far forward/backward over foot."},
            }
        }
    },
    "Push-Up": {
        "description": "Maintain a straight body line from head to heels. Engage your core.",
        "landmark_points": {
            "right_elbow": [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value],
            "left_elbow": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value],
            "body_line_right": [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
            "body_line_left": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_ANKLE.value],
        },
        "phase_detection": {
            "primary_angles_for_phase": ["left_elbow", "right_elbow"],
            "phases": [
                {"name": "start_position", "condition_angle_above": 155},
                {"name": "pushup_descent", "condition_angle_below": 110, "condition_angle_above_min": 65}
            ]
        },
        "feedback_rules": {
            "start_position": {
                "right_elbow": {"min": 160, "max": 180, "feedback": "Arms straight in plank!"},
                "left_elbow": {"min": 160, "max": 180, "feedback": "Arms straight in plank!"},
                "body_line_right": {"min": 160, "max": 180, "feedback": "Keep your body straight!"},
                "body_line_left": {"min": 160, "max": 180, "feedback": "Keep your body straight!"}
            },
            "pushup_descent": {
                "right_elbow": {"min": 70, "max": 100, "feedback": "Lower chest towards floor."},
                "left_elbow": {"min": 70, "max": 100, "feedback": "Lower chest towards floor."},
                "body_line_right": {"min": 155, "max": 180, "feedback": "Don't sag your hips!"},
                "body_line_left": {"min": 155, "max": 180, "feedback": "Don't sag your hips!"}
            },
            "general_form": {
                 "hand_shoulder_alignment_right": {"check_type": "y_alignment", "landmark1": mp_pose.PoseLandmark.RIGHT_WRIST.value, "landmark2": mp_pose.PoseLandmark.RIGHT_SHOULDER.value, "tolerance_body_prop": 0.1, "feedback": "Align right hand under shoulder."},
                 "hand_shoulder_alignment_left": {"check_type": "y_alignment", "landmark1": mp_pose.PoseLandmark.LEFT_WRIST.value, "landmark2": mp_pose.PoseLandmark.LEFT_SHOULDER.value, "tolerance_body_prop": 0.1, "feedback": "Align left hand under shoulder."}
            }
        }
    },
    "Plank": {
        "description": "Maintain a straight line from head to heels. Engage your core and glutes.",
        "landmark_points": {
            "right_elbow": [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value],
            "left_elbow": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value],
            "body_line_hip": [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value],
            "body_line_knee": [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
        },
        "phase_detection": {
            "primary_angles_for_phase": ["body_line_hip", "body_line_knee"],
            "phases": [
                {"name": "hold_position", "condition_angle_above": 150, "condition_angle_below": 185}
            ]
        },
        "feedback_rules": {
            "hold_position": {
                "right_elbow": {"min": 80, "max": 100, "feedback": "Elbows should be bent at 90 degrees."},
                "left_elbow": {"min": 80, "max": 100, "feedback": "Elbows should be bent at 90 degrees."},
                "body_line_hip": {"min": 160, "max": 180, "feedback": "Keep your hips in line with shoulders and knees."},
                "body_line_knee": {"min": 160, "max": 180, "feedback": "Keep your body straight from knees to ankles."},
            }
        }
    },
    "Lunge": {
        "description": "Ensure your front knee is over your ankle and your back knee hovers above the ground.",
        "landmark_points": {
            "front_right_knee": [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
            "front_left_knee": [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value],
            "back_right_knee": [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value], # Angle for back knee
            "back_left_knee": [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value], # Angle for back knee
            "right_hip": [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value],
            "left_hip": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value],
        },
        "phase_detection": {
            "primary_angles_for_phase": ["front_left_knee", "front_right_knee"],
            "phases": [
                {"name": "start_position", "condition_angle_above": 155},
                {"name": "lunge_descent", "condition_angle_below": 110, "condition_angle_above_min": 65}
            ]
        },
        "feedback_rules": {
            "start_position": {
                "front_right_knee": {"min": 160, "max": 180, "feedback": "Stand tall, prepare for lunge."},
                "front_left_knee": {"min": 160, "max": 180, "feedback": "Stand tall, prepare for lunge."},
            },
            "lunge_descent": {
                "front_right_knee": {"min": 80, "max": 100, "feedback": "Front knee at 90 degrees!"},
                "front_left_knee": {"min": 80, "max": 100, "feedback": "Front knee at 90 degrees!"},
                "back_right_knee": {"min": 80, "max": 100, "feedback": "Back knee towards the floor!"},
                "back_left_knee": {"min": 80, "max": 100, "feedback": "Back knee towards the floor!"},
            },
            "general_form": {
                "front_right_knee_over_foot_horizontal": {"check_type": "x_alignment", "landmark1": mp_pose.PoseLandmark.RIGHT_KNEE.value, "landmark2": mp_pose.PoseLandmark.RIGHT_ANKLE.value, "tolerance_body_prop": 0.05, "feedback": "Front knee too far forward!"},
                "front_left_knee_over_foot_horizontal": {"check_type": "x_alignment", "landmark1": mp_pose.PoseLandmark.LEFT_KNEE.value, "landmark2": mp_pose.PoseLandmark.LEFT_ANKLE.value, "tolerance_body_prop": 0.05, "feedback": "Front knee too far forward!"},
            }
        }
    },
    "Burpee": { # Focus on the squat and push-up phases
        "description": "A full-body exercise. Focus on smooth transitions between squat, plank, push-up, and jump.",
        "landmark_points": {
            "right_knee": [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
            "left_knee": [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value],
            "right_elbow": [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value],
            "left_elbow": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value],
            "body_line_hip": [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value],
            "body_line_knee": [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
        },
        "phase_detection": {
            "primary_angles_for_phase": ["right_knee", "left_knee", "right_elbow", "left_elbow"], # More complex phase detection
            "phases": [
                {"name": "standing_start", "condition_angle_above": 150}, # Knees/elbows mostly straight
                {"name": "squat_phase", "condition_angle_below": 120, "condition_angle_above_min": 65}, # Knees bent
                {"name": "plank_phase", "condition_angle_above": 150}, # Arms straight, body straight
                {"name": "pushup_phase", "condition_angle_below": 110, "condition_angle_above_min": 65}, # Elbows bent
                {"name": "jump_phase", "condition_angle_above": 150} # Knees/elbows mostly straight (after pushup)
            ]
        },
        "feedback_rules": {
            "standing_start": {
                "right_knee": {"min": 160, "max": 180, "feedback": "Stand tall!"},
            },
            "squat_phase": {
                "right_knee": {"min": 70, "max": 110, "feedback": "Squat down!"},
            },
            "plank_phase": {
                "body_line_hip": {"min": 160, "max": 180, "feedback": "Straight body in plank!"},
            },
            "pushup_phase": {
                "right_elbow": {"min": 70, "max": 100, "feedback": "Lower chest to ground!"},
            },
            "jump_phase": {
                "right_knee": {"min": 160, "max": 180, "feedback": "Explode up!"},
            }
        }
    },
    "Mountain Climber": {
        "description": "Maintain a strong plank, bringing knees towards your chest alternately.",
        "landmark_points": {
            "right_elbow": [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value],
            "left_elbow": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value],
            "right_hip_flexion": [mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            "left_hip_flexion": [mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            "body_line_hip": [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value],
        },
        "phase_detection": {
            "primary_angles_for_phase": ["body_line_hip"], # Primarily plank stability
            "phases": [
                {"name": "plank_start", "condition_angle_above": 150, "condition_angle_below": 185}, # Stable plank
                {"name": "knee_drive", "condition_angle_below": 130, "condition_angle_above_min": 60} # Knee driven forward
            ]
        },
        "feedback_rules": {
            "plank_start": {
                "right_elbow": {"min": 160, "max": 180, "feedback": "Arms straight, strong plank!"},
                "body_line_hip": {"min": 160, "max": 180, "feedback": "Keep your body straight!"},
            },
            "knee_drive": {
                "right_hip_flexion": {"min": 70, "max": 120, "feedback": "Drive knee to chest!"},
                "left_hip_flexion": {"min": 70, "max": 120, "feedback": "Drive knee to chest!"},
            }
        }
    },
    "Jumping Jack": {
        "description": "Full body cardio. Coordinate arm and leg movements.",
        "landmark_points": {
            "right_shoulder_abduction": [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            "left_shoulder_abduction": [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value],
            "right_hip_abduction": [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value],
            "left_hip_abduction": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value],
        },
        "phase_detection": {
            "primary_angles_for_phase": ["right_shoulder_abduction", "left_shoulder_abduction"],
            "phases": [
                {"name": "start_position", "condition_angle_below": 40}, # Arms down
                {"name": "jump_open", "condition_angle_above": 70} # Arms up
            ]
        },
        "feedback_rules": {
            "start_position": {
                "right_shoulder_abduction": {"min": 0, "max": 30, "feedback": "Arms by your side!"},
                "left_shoulder_abduction": {"min": 0, "max": 30, "feedback": "Arms by your side!"},
                "right_hip_abduction": {"min": 0, "max": 30, "feedback": "Legs together!"},
                "left_hip_abduction": {"min": 0, "max": 30, "feedback": "Legs together!"},
            },
            "jump_open": {
                "right_shoulder_abduction": {"min": 80, "max": 110, "feedback": "Arms overhead!"},
                "left_shoulder_abduction": {"min": 80, "max": 110, "feedback": "Arms overhead!"},
                "right_hip_abduction": {"min": 40, "max": 70, "feedback": "Legs wide!"},
                "left_hip_abduction": {"min": 40, "max": 70, "feedback": "Legs wide!"},
            }
        }
    },
    "Single-Leg Glute Bridge": {
        "description": "Lift hips off the ground, keeping one leg extended. Squeeze glutes.",
        "landmark_points": {
            "right_hip_extension": [mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            "left_hip_extension": [mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            "right_knee_flexion": [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
            "left_knee_flexion": [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value],
        },
        "phase_detection": {
            "primary_angles_for_phase": ["right_hip_extension", "left_hip_extension"],
            "phases": [
                {"name": "start_position", "condition_angle_below": 160}, # Hips down
                {"name": "bridge_peak", "condition_angle_above": 170} # Hips up
            ]
        },
        "feedback_rules": {
            "start_position": {
                "right_hip_extension": {"min": 160, "max": 180, "feedback": "Lie flat, prepare to lift hips."},
                "left_hip_extension": {"min": 160, "max": 180, "feedback": "Lie flat, prepare to lift hips."},
            },
            "bridge_peak": {
                "right_hip_extension": {"min": 170, "max": 180, "feedback": "Hips fully extended, squeeze glutes!"},
                "left_hip_extension": {"min": 170, "max": 180, "feedback": "Hips fully extended, squeeze glutes!"},
                "right_knee_flexion": {"min": 80, "max": 100, "feedback": "Bent knee at 90 degrees."},
                "left_knee_flexion": {"min": 80, "max": 100, "feedback": "Bent knee at 90 degrees."},
            }
        }
    },
    "Cat-Cow": { # Simplified - focuses on relative hip/shoulder/knee positions for spinal movement
        "description": "Flow through spinal flexion and extension. Coordinate with breath.",
        "landmark_points": {
            "right_hip_shoulder_line": [mp_pose.PoseLandmark.RIGHT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            "left_hip_shoulder_line": [mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            "right_knee_hip_line": [mp_pose.PoseLandmark.RIGHT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_HIP.value],
            "left_knee_hip_line": [mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_HIP.value],
        },
        "phase_detection": {
            "primary_angles_for_phase": ["right_hip_shoulder_line", "left_hip_shoulder_line"],
            "phases": [
                {"name": "cat_pose", "condition_angle_below": 135}, # Spinal flexion (rounded back)
                {"name": "cow_pose", "condition_angle_above": 145} # Spinal extension (arched back)
            ]
        },
        "feedback_rules": {
            "cat_pose": { # Spinal flexion
                "right_hip_shoulder_line": {"min": 100, "max": 130, "feedback": "Arch your back up (Cat)!"},
                "left_hip_shoulder_line": {"min": 100, "max": 130, "feedback": "Arch your back up (Cat)!"},
            },
            "cow_pose": { # Spinal extension
                "right_hip_shoulder_line": {"min": 140, "max": 170, "feedback": "Drop your belly (Cow)!"},
                "left_hip_shoulder_line": {"min": 140, "max": 170, "feedback": "Drop your belly (Cow)!"},
            }
        }
    }
}


# --- Helper Functions ---
def calculate_joint_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

# --- Functions for Uploaded Video Analysis ---
def extract_landmarks(video_path):
    pose_instance = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # Ensure imageio.get_reader is used as per user's note, even if iio is imported
    reader = imageio.get_reader(video_path, "ffmpeg") 
    landmarks_per_frame = []
    for frame_data in reader:
        image = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
        results = pose_instance.process(image)
        if results.pose_landmarks:
            frame_landmarks = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
            landmarks_per_frame.append(frame_landmarks)
        else:
            landmarks_per_frame.append(np.full((33, 4), np.nan))
    reader.close()
    pose_instance.close()
    return np.array(landmarks_per_frame)

def calculate_asymmetry(landmarks_data, left_idx, right_idx):
    """
    Calculates the absolute difference in x-coordinates between a left and right landmark.
    Assumes landmarks_data is (frames, landmarks, coords).
    """
    if landmarks_data.ndim != 3 or landmarks_data.shape[1] < max(left_idx, right_idx) + 1:
        return np.array([]) # Return empty if data is not in expected format

    # FIX: Accessing x-coordinate from the correct dimension (index 0)
    left_x = landmarks_data[:, left_idx, 0] 
    right_x = landmarks_data[:, right_idx, 0]
    
    # Handle NaN values: if either is NaN, the difference is NaN
    diff = np.abs(left_x - right_x)
    return diff

def get_elbow_asymmetry(landmarks_data):
    left_elbow_angles, right_elbow_angles = [], []
    if landmarks_data.ndim != 3: return np.array([])
    for frame_landmarks in landmarks_data:
        for side in ["LEFT", "RIGHT"]:
            shoulder_lm = getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value
            elbow_lm = getattr(mp_pose.PoseLandmark, f"{side}_ELBOW").value
            wrist_lm = getattr(mp_pose.PoseLandmark, f"{side}_WRIST").value
            if np.isnan(frame_landmarks[shoulder_lm, 0]) or \
               np.isnan(frame_landmarks[elbow_lm, 0]) or \
               np.isnan(frame_landmarks[wrist_lm, 0]):
                angle = np.nan
            else:
                p_s = frame_landmarks[shoulder_lm, :3]
                p_e = frame_landmarks[elbow_lm, :3]
                p_w = frame_landmarks[wrist_lm, :3]
                angle = calculate_joint_angle(p_s, p_e, p_w)
            (left_elbow_angles if side == "LEFT" else right_elbow_angles).append(angle)
    return np.abs(np.array(left_elbow_angles) - np.array(right_elbow_angles))

def get_knee_flexion_stats(landmarks_data):
    left_knee_angles, right_knee_angles = [], []
    if landmarks_data.ndim != 3: return np.nan, np.nan, np.array([]), np.array([])
    for frame_landmarks in landmarks_data:
        for side in ["LEFT", "RIGHT"]:
            hip_lm = getattr(mp_pose.PoseLandmark, f"{side}_HIP").value
            knee_lm = getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value
            ankle_lm = getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value
            if np.isnan(frame_landmarks[hip_lm, 0]) or \
               np.isnan(frame_landmarks[knee_lm, 0]) or \
               np.isnan(frame_landmarks[ankle_lm, 0]):
                angle = np.nan
            else:
                p_h = frame_landmarks[hip_lm, :3]
                p_k = frame_landmarks[knee_lm, :3]
                p_a = frame_landmarks[ankle_lm, :3]
                angle = calculate_joint_angle(p_h, p_k, p_a)
            (left_knee_angles if side == "LEFT" else right_knee_angles).append(angle)
    
    clean_left = [a for a in left_knee_angles if not np.isnan(a)]
    clean_right = [a for a in right_knee_angles if not np.isnan(a)]
    min_left = np.min(clean_left) if clean_left else np.nan
    min_right = np.min(clean_right) if clean_right else np.nan
    return min_left, min_right, np.array(left_knee_angles), np.array(right_knee_angles)

def display_preview_table(landmarks_data):
    """Displays a preview table of landmark data for the first few frames."""
    if landmarks_data.size == 0:
        st.write("No landmark data to display.")
        return pd.DataFrame()
    num_frames_to_show = min(5, landmarks_data.shape[0])
    preview_data = []
    key_landmarks_indices = [
        mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]
    for i in range(num_frames_to_show):
        frame_landmarks = landmarks_data[i]
        row = {"Frame": i}
        for idx in key_landmarks_indices:
            name = mp_pose.PoseLandmark(idx).name
            # FIX: Access directly, as landmarks_data is already (frames, 33, 4)
            if not np.isnan(frame_landmarks[idx, 0]):
                row[f"{name} (X)"] = f"{frame_landmarks[idx, 0]:.2f}"
                row[f"{name} (Y)"] = f"{frame_landmarks[idx, 1]:.2f}"
            else:
                row[f"{name} (X)"] = "N/A"; row[f"{name} (Y)"] = "N/A"
        preview_data.append(row)
    df_preview = pd.DataFrame(preview_data)
    st.subheader("Landmark Data Preview (First few frames)")
    st.dataframe(df_preview)
    return df_preview

# --- MAIN APP LOGIC ---
st.header("Choose Your Video Input Method")

st.sidebar.header("Live Exercise Selection")
selected_exercise_name = st.sidebar.selectbox(
    "Select Exercise for Live Feedback:",
    list(EXERCISES.keys())
)

input_method = st.radio(
    "How would you like to analyze posture?",
    ("Live Webcam (Real-time Feedback)", "Upload Walking Videos (Detailed Gait Analysis)")
)

if input_method == "Live Webcam (Real-time Feedback)":
    st.subheader(f"Live Posture Analysis: {selected_exercise_name}")
    exercise_data = EXERCISES.get(selected_exercise_name)
    if exercise_data and "description" in exercise_data:
        st.info(f"**Instructions for {selected_exercise_name}:** {exercise_data['description']}")
    
    # Custom webcam implementation
    st.warning("Grant camera permissions to start. Ensure good lighting and that your full body is visible for best results.")

    # Initialize session state for webcam
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    if 'last_frame' not in st.session_state:
        st.session_state.last_frame = None
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = []

    # Webcam controls
    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.webcam_active:
            if st.button("🎥 Start Webcam", type="primary"):
                st.session_state.webcam_active = True
                st.rerun()
        else:
            if st.button("⏹️ Stop Webcam"):
                st.session_state.webcam_active = False
                st.rerun()
    
    with col2:
        if st.session_state.webcam_active:
            if st.button("📸 Capture Frame"):
                if st.session_state.last_frame is not None:
                    st.image(st.session_state.last_frame, caption="Captured Frame", use_container_width=True)

    # Webcam feed
    if st.session_state.webcam_active:
        # Use Streamlit's camera input but process frames continuously
        camera_img = st.camera_input("Real-time posture analysis", key="live_webcam")
        
        if camera_img:
            # Process the frame
            with st.spinner("Analyzing posture..."):
                # Convert to OpenCV format
                image = Image.open(camera_img)
                img_array = np.array(image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Store the frame
                st.session_state.last_frame = img_array
                
                # Process with MediaPipe
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    results = pose.process(img_bgr)
                    
                    if results.pose_landmarks:
                        # Create annotated image
                        img_annotated = img_array.copy()
                        mp_drawing.draw_landmarks(
                            img_annotated,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                        )
                        
                        # Calculate angles and provide feedback
                        landmarks_mp = results.pose_landmarks.landmark
                        img_h, img_w, _ = img_annotated.shape
                        
                        current_angles_dict = {}
                        feedback_messages = []
                        
                        # Get exercise configuration
                        exercise_config = EXERCISES.get(selected_exercise_name)
                        if exercise_config and "landmark_points" in exercise_config:
                            for angle_name, landmark_indices in exercise_config["landmark_points"].items():
                                if len(landmark_indices) == 3:
                                    p1_idx, p2_idx, p3_idx = landmark_indices
                                    
                                    if (landmarks_mp[p1_idx].visibility > 0.3 and 
                                        landmarks_mp[p2_idx].visibility > 0.3 and 
                                        landmarks_mp[p3_idx].visibility > 0.3):
                                        
                                        pt1 = [landmarks_mp[p1_idx].x, landmarks_mp[p1_idx].y, landmarks_mp[p1_idx].z]
                                        pt2 = [landmarks_mp[p2_idx].x, landmarks_mp[p2_idx].y, landmarks_mp[p2_idx].z]
                                        pt3 = [landmarks_mp[p3_idx].x, landmarks_mp[p3_idx].y, landmarks_mp[p3_idx].z]
                                        
                                        angle = calculate_joint_angle(pt1, pt2, pt3)
                                        current_angles_dict[angle_name] = angle
                                        
                                        # Draw angle on image
                                        cv2.putText(img_annotated, f"{int(angle)}°",
                                                    (int(landmarks_mp[p2_idx].x * img_w) + 10, 
                                                     int(landmarks_mp[p2_idx].y * img_h)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                        
                        # Display results
                        st.subheader("Live Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(img_annotated, caption="Live Pose Analysis", use_container_width=True)
                        
                        with col2:
                            st.write("**Current Joint Angles:**")
                            for angle_name, angle_value in current_angles_dict.items():
                                st.write(f"- {angle_name}: {angle_value:.1f}°")
                            
                            # Provide real-time feedback
                            st.write("**Feedback:**")
                            if current_angles_dict:
                                # Simple feedback logic - you can expand this with your existing rules
                                if "right_knee" in current_angles_dict:
                                    knee_angle = current_angles_dict["right_knee"]
                                    if knee_angle < 80:
                                        st.warning("🔴 Knee angle too deep")
                                    elif knee_angle > 160:
                                        st.info("🔵 Standing position")
                                    else:
                                        st.success("🟢 Good knee position")
                                
                                if "right_elbow" in current_angles_dict:
                                    elbow_angle = current_angles_dict["right_elbow"]
                                    if elbow_angle < 70:
                                        st.warning("🔴 Elbow too bent")
                                    elif elbow_angle > 170:
                                        st.info("🔵 Arm extended")
                                    else:
                                        st.success("🟢 Good arm position")
                                
                                # Store feedback for history
                                feedback_entry = {
                                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                                    "angles": current_angles_dict.copy()
                                }
                                st.session_state.feedback_history.append(feedback_entry)
                                
                                # Keep only last 10 entries
                                if len(st.session_state.feedback_history) > 10:
                                    st.session_state.feedback_history.pop(0)
                            else:
                                st.warning("No angles calculated - ensure full body visibility")
                    
                    else:
                        st.warning("No pose detected. Please ensure your full body is visible in the frame.")
        
        # Feedback history
        if st.session_state.feedback_history:
            st.subheader("Recent Feedback History")
            for i, feedback in enumerate(reversed(st.session_state.feedback_history[-5:])):
                with st.expander(f"Frame {len(st.session_state.feedback_history)-i} - {feedback['timestamp']}"):
                    for angle_name, angle_value in feedback['angles'].items():
                        st.write(f"{angle_name}: {angle_value:.1f}°")
    
    else:
        st.info("Click 'Start Webcam' to begin real-time posture analysis")
    
elif input_method == "Upload Walking Videos (Detailed Gait Analysis)":
    st.markdown("### Upload Two Walking Videos (Opposite Directions)")
    col1, col2 = st.columns(2)
    with col1:
        vid1 = st.file_uploader("East-to-West", type=["mp4","mov","avi"], key="v1")
    with col2:
        vid2 = st.file_uploader("West-to-East", type=["mp4","mov","avi"], key="v2")

    comparisons = []

    for label, video_file in [("East-to-West", vid1), ("West-to-East", vid2)]:
        if video_file:
            path = None # Initialize path
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(video_file.read())
                    path = tmp.name
            
                st.subheader(f"Video Preview ({label})")
                st.video(video_file)

                with imageio.get_reader(path, "ffmpeg") as reader: # Use imageio.get_reader
                    try:
                        # reader.count_frames() can be problematic with some files/ffmpeg versions
                        # Instead, try to read a few frames directly.
                        frame_indices_to_show = []
                        temp_frames = []
                        for i_frame, f_data in enumerate(reader):
                            if i_frame == 0: 
                                temp_frames.append(f_data)
                                frame_indices_to_show.append(i_frame)
                            # Heuristic to get a middle frame without knowing total count
                            # This part is tricky without reliable count_frames()
                            # For simplicity, let's just show first and attempt last if possible
                        
                        # If more frames were read, try to get one from near the end
                        # This is a simplified approach as iterating whole video to get last frame is slow
                        # For now, just showing the first frame for preview to avoid issues
                        if temp_frames:
                             st.image(temp_frames[0], caption=f"{label} Frame {frame_indices_to_show[0]}", use_container_width=True)
                        else:
                            st.warning(f"Could not read frames for preview from {label}.")

                    except Exception as e_preview:
                        st.warning(f"Could not generate full preview for {label}: {e_preview}")


                with st.spinner(f"Analyzing {label} video... This may take a moment."):
                    landmarks = extract_landmarks(path)
                
                if landmarks.size == 0 or np.all(np.isnan(landmarks)): 
                    st.error(f"No landmarks detected in {label} video. Please try another video or ensure full body visibility.")
                    if path and os.path.exists(path): os.remove(path) # Clean up temp file
                    continue 

                display_preview_table(landmarks)
                
                hip_diff = calculate_asymmetry(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value)
                elbow_diff = get_elbow_asymmetry(landmarks)
                left_knee_min, right_knee_min, left_knees, right_knees = get_knee_flexion_stats(landmarks)

                avg_hip = np.nanmean(hip_diff) if hip_diff.size > 0 else np.nan
                max_hip = np.nanmax(hip_diff) if hip_diff.size > 0 else np.nan
                avg_elbow = np.nanmean(elbow_diff) if elbow_diff.size > 0 else np.nan
                max_elbow = np.nanmax(elbow_diff) if elbow_diff.size > 0 else np.nan
                
                comparisons.append((label, avg_hip, max_hip, avg_elbow, max_elbow, left_knee_min, right_knee_min))

                st.subheader(f"{label} Knee Flexion Range")
                fig_knee, ax_knee = plt.subplots()
                if left_knees.size > 0 and not np.all(np.isnan(left_knees)): ax_knee.plot(left_knees, label="Left Knee Angle")
                if right_knees.size > 0 and not np.all(np.isnan(right_knees)): ax_knee.plot(right_knees, label="Right Knee Angle")
                ax_knee.axhline(stiff_threshold, color='r', linestyle='--', label=f'Stiffness Threshold ({stiff_threshold}°)')
                ax_knee.set_xlabel("Frame"); ax_knee.set_ylabel("Angle (°)")
                ax_knee.legend(); st.pyplot(fig_knee)
                if not np.isnan(left_knee_min): st.write(f"Min Left Knee Flexion: {left_knee_min:.2f}°")
                if not np.isnan(right_knee_min): st.write(f"Min Right Knee Flexion: {right_knee_min:.2f}°")

                st.subheader(f"{label} Hip Asymmetry Over Time (X-coord diff)")
                fig_hip, ax_hip = plt.subplots()
                if hip_diff.size > 0 and not np.all(np.isnan(hip_diff)): ax_hip.plot(hip_diff, label="Hip X diff (Normalized)")
                hip_asym_plot_thresh = 0.05 
                ax_hip.axhline(hip_asym_plot_thresh, color='r', linestyle='--', label=f'Asym. Threshold ({hip_asym_plot_thresh:.2f})')
                ax_hip.set_xlabel("Frame"); ax_hip.set_ylabel("Abs Diff (Normalized X)")
                ax_hip.legend(); st.pyplot(fig_hip)
                if not np.isnan(avg_hip): st.write(f"Avg Hip Asymmetry ({label}): {avg_hip:.4f}")
                if not np.isnan(max_hip): st.write(f"Max Hip Asymmetry ({label}): {max_hip:.4f}")
                if not np.isnan(avg_hip) and avg_hip > norms["hip"]: 
                    st.warning(f"Hip asymmetry ({avg_hip:.3f}) is higher than average for your age group ({norms['hip']:.3f}).")

                st.subheader(f"{label} Arm Swing Asymmetry Over Time (Elbow Angle Diff)")
                fig_arm, ax_arm = plt.subplots()
                if elbow_diff.size > 0 and not np.all(np.isnan(elbow_diff)): ax_arm.plot(elbow_diff, label="Elbow Angle Diff")
                ax_arm.axhline(asym_threshold, color='r', linestyle='--', label=f'Asym. Threshold ({asym_threshold}°)') 
                ax_arm.set_xlabel("Frame"); ax_arm.set_ylabel("Degrees")
                ax_arm.legend(); st.pyplot(fig_arm)
                if not np.isnan(avg_elbow): st.write(f"Avg Arm Asymmetry ({label}): {avg_elbow:.2f}°")
                if not np.isnan(max_elbow): st.write(f"Max Arm Asymmetry ({label}): {max_elbow:.2f}°")
                if not np.isnan(avg_elbow) and avg_elbow > norms["arm"]: 
                    st.warning(f"Arm swing asymmetry ({avg_elbow:.1f}°) is higher than average for your age group ({norms['arm']}°).")

                if (not np.isnan(avg_elbow) and avg_elbow > asym_threshold) or \
                   (not np.isnan(avg_hip) and avg_hip > hip_asym_plot_thresh): 
                    st.subheader("Recommended Corrective Exercises (General)")
                    st.markdown("- **Hip bridges**: Strengthen glutes and hamstrings\n- **Leg swings**: Improve balance and flexibility\n- **Step-ups**: Build lower body strength\n- **Lunges with form control**: Improve symmetry")

                csv_data = {
                    "Frame": list(range(len(landmarks))),
                    "Hip_X_Diff": hip_diff if hip_diff.size == len(landmarks) else [np.nan]*len(landmarks),
                    "Elbow_Angle_Diff": elbow_diff if elbow_diff.size == len(landmarks) else [np.nan]*len(landmarks),
                    "Left_Knee_Angle": left_knees if left_knees.size == len(landmarks) else [np.nan]*len(landmarks),
                    "Right_Knee_Angle": right_knees if right_knees.size == len(landmarks) else [np.nan]*len(landmarks) # FIX: Changed len_landmarks to len(landmarks)
                }
                max_len = len(landmarks)
                for key_csv in csv_data: # Renamed key to key_csv to avoid conflict
                    if key_csv != "Frame":
                        current_list = list(csv_data[key_csv])
                        if len(current_list) < max_len:
                            current_list.extend([np.nan] * (max_len - len(current_list)))
                        csv_data[key_csv] = current_list[:max_len]

                export_df = pd.DataFrame(csv_data)
                csv_buffer = BytesIO()
                export_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                st.download_button(
                    label="Download Analysis Data CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"{label.replace(' ', '_').lower()}_analysis_data.csv",
                    mime="text/csv"
                )
            finally: 
                if path and os.path.exists(path):
                    os.remove(path)

    if len(comparisons) == 2:
        st.subheader("Comparison Summary")
        df_compare = pd.DataFrame(comparisons, columns=["Video", "Avg Hip Asym (NormX)", "Max Hip Asym (NormX)", "Avg Arm Asym (°)", "Max Arm Asym (°)", "Min Left Knee (°)", "Min Right Knee (°)"])
        st.dataframe(df_compare.set_index("Video"))

st.sidebar.markdown("---")
st.sidebar.info("Stride Buddy v0.3 - AI Posture Coach (Improved Stability)")