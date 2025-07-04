import sys
import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox, QComboBox
)
import math
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent
from PyQt5.QtGui import QImage, QPixmap
from PIL import ImageFont, ImageDraw, Image

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

def draw_emoji_on_frame(frame, emoji, pos, size):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("seguiemj.ttf", size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, emoji, font=font, embedded_color=True)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_centered_emoji_on_frame(frame, emoji, center, size):
    """Draw emoji so that its center is at the given (x, y) position."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    try:
        font = ImageFont.truetype("seguiemj.ttf", size)
    except:
        font = ImageFont.load_default()
    # Create a transparent image for the emoji
    emoji_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(emoji_img)
    # Get text size using textbbox for accurate centering
    bbox = draw.textbbox((0, 0), emoji, font=font, embedded_color=True)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text(((size - w) // 2, (size - h) // 2), emoji, font=font, embedded_color=True)
    # Paste onto frame so that center is at (center[0], center[1])
    x, y = center
    paste_x = int(x - size // 2)
    paste_y = int(y - size // 2)
    img_pil.paste(emoji_img, (paste_x, paste_y), emoji_img)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def get_crown_position(face_landmarks, iw, ih):
    # Place crown above the head using forehead and chin for better height
    forehead = face_landmarks.landmark[10]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[200]
    chin = face_landmarks.landmark[152]
    # Center x between eyes and forehead
    x = int((left_eye.x * iw + right_eye.x * iw) / 2)
    # y above the forehead, distance from forehead to chin
    face_height = abs(forehead.y * ih - chin.y * ih)
    y = int(forehead.y * ih - face_height * 0.7)
    return x, y

def recognize_hand_gesture(hand_landmarks):
    # Enhanced gesture recognition: thumbs up, wave, peace, call me, high five, fist, fire, rock, ok, vulcan, pointing
    # Returns emoji index:
    # 1=👍, 2=👋, 3=✌️, 4=🤙, 5=🖐️, 6=✊, 7=🔥, 8=🤘, 9=👌, 10=🖖, 11=👉, 0=none
    tips = [4, 8, 12, 16, 20]
    pip = [3, 6, 10, 14, 18]
    fingers = []
    for i in range(1, 5):
        fingers.append(hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[pip[i]].y)
    # Thumb: check x for right hand, reversed for left
    if hand_landmarks.landmark[17].x < hand_landmarks.landmark[5].x:
        thumb = hand_landmarks.landmark[tips[0]].x > hand_landmarks.landmark[pip[0]].x
    else:
        thumb = hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[pip[0]].x

    # --- Gesture conditions ---
    # OK: thumb tip and index tip close together, others can be up or down
    ok_dist = (
        (hand_landmarks.landmark[4].x - hand_landmarks.landmark[8].x) ** 2 +
        (hand_landmarks.landmark[4].y - hand_landmarks.landmark[8].y) ** 2
    ) ** 0.5
    ok_gesture = ok_dist < 0.07

    # Vulcan: index/middle up, ring/pinky up, gap between middle and ring
    vulcan_gap = abs(hand_landmarks.landmark[12].x - hand_landmarks.landmark[16].x)
    vulcan = (
        fingers[0] and fingers[1] and fingers[2] and fingers[3] and
        vulcan_gap > 0.07
    )

    # Rock: index and pinky up, middle/ring/THUMB down
    rock = fingers[0] and not fingers[1] and not fingers[2] and fingers[3] and not thumb

    # Pointing (👉): index up, others down
    pointing = fingers[0] and not any(fingers[1:]) and not thumb

    # Peace: index and middle up, ring and pinky down (regardless of thumb)
    peace = fingers[0] and fingers[1] and not fingers[2] and not fingers[3]

    # --- Existing gestures ---
    if thumb and not any(fingers):
        return 1  # 👍
    if all(fingers) and not thumb:
        return 2  # 👋
    if peace:
        return 3  # ✌️
    if thumb and fingers[0] and not any(fingers[1:]):
        return 4  # 🤙
    if all(fingers) and thumb:
        return 5  # 🖐️
    if not thumb and not any(fingers):
        return 6  # ✊ (fist)
    if all(fingers) and thumb and hand_landmarks.landmark[0].y < 0.3:
        return 7  # 🔥
    if rock:
        return 8  # 🤘
    if ok_gesture:
        return 9  # 👌
    if vulcan:
        return 10  # 🖖
    if pointing:
        return 11  # 👉
    return 0

def eye_state(face_landmarks, iw, ih):
    # Returns (left_eye_open, right_eye_open, left_eye_center, right_eye_center)
    # Use vertical distance between upper/lower eyelid landmarks
    # Left eye: 159 (upper), 145 (lower), center: 33
    # Right eye: 386 (upper), 374 (lower), center: 263
    left_up = face_landmarks.landmark[159]
    left_down = face_landmarks.landmark[145]
    right_up = face_landmarks.landmark[386]
    right_down = face_landmarks.landmark[374]
    left_eye_center = (
        int(face_landmarks.landmark[33].x * iw),
        int(face_landmarks.landmark[33].y * ih)
    )
    right_eye_center = (
        int(face_landmarks.landmark[263].x * iw),
        int(face_landmarks.landmark[263].y * ih)
    )
    left_dist = abs(left_up.y - left_down.y)
    right_dist = abs(right_up.y - right_down.y)
    # Thresholds may need tuning depending on camera
    left_open = left_dist > 0.018
    right_open = right_dist > 0.018
    return left_open, right_open, left_eye_center, right_eye_center

def count_fingers(hand_landmarks):
    # Returns number of fingers up (max 5 per hand)
    tips = [4, 8, 12, 16, 20]
    pip = [3, 6, 10, 14, 18]
    fingers = []
    for i in range(1, 5):
        fingers.append(hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[pip[i]].y)
    # Thumb: check x for right hand, reversed for left
    if hand_landmarks.landmark[17].x < hand_landmarks.landmark[5].x:
        thumb = hand_landmarks.landmark[tips[0]].x > hand_landmarks.landmark[pip[0]].x
    else:
        thumb = hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[pip[0]].x
    return int(thumb) + sum(fingers)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    gesture_signal = pyqtSignal(str)
    eye_signal = pyqtSignal(str)
    finger_signal = pyqtSignal(str)
    face_count_signal = pyqtSignal(int)
    
    def __init__(self, feature_states, face_mesh_style, face_emoji_style, hand_emoji_style, eye_toggle, auto_hand_emoji, camera_index=0, ar_accessory_style=None, show_face_mesh=None):
        super().__init__()
        self.feature_states = feature_states
        self.face_mesh_style = face_mesh_style
        self.face_emoji_style = face_emoji_style
        self.hand_emoji_style = hand_emoji_style
        self.eye_toggle = eye_toggle
        self.auto_hand_emoji = auto_hand_emoji
        self.camera_index = camera_index
        self.ar_accessory_style = ar_accessory_style if ar_accessory_style else (lambda: 0)
        self.show_face_mesh = show_face_mesh if show_face_mesh else (lambda: True)
        self.running = False
        # Preload glasses and moustache PNGs (transparent background)
        try:
            self.glasses_img = Image.open("./ar_assets/glasses.png").convert("RGBA")

        except Exception as e:
            print(f"✗ Failed to load glasses: {e}")
            self.glasses_img = None
        try:
            self.moustache_img = Image.open("./ar_assets/moustache.png").convert("RGBA")
        except Exception as e:
            print(f"✗ Failed to load moustache: {e}")
            self.moustache_img = None

    def run(self):
        self.running = True
        use_screen = self.camera_index == -1
        if not use_screen:
            cap = cv2.VideoCapture(self.camera_index)
        hands = face = pose_est = holistic_est = selfie_seg_est = None
        last_states = [None] * 5
        last_style = None
        last_face_emoji = None
        last_hand_emoji = None
        last_eye_toggle = None

        import pyautogui

        while self.running and (use_screen or (cap and cap.isOpened())):
            # --- Get frame ---
            if use_screen:
                screen = pyautogui.screenshot()
                frame = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Check for state changes ---
            states = self.feature_states()
            style = self.face_mesh_style()
            face_emoji = self.face_emoji_style()
            hand_emoji = self.hand_emoji_style()
            eye_toggle = self.eye_toggle()
            ar_accessory = self.ar_accessory_style()
            if (states != last_states or style != last_style or
                face_emoji != last_face_emoji or hand_emoji != last_hand_emoji or
                eye_toggle != last_eye_toggle):
                if hands: hands.close()
                if face: face.close()
                if pose_est: pose_est.close()
                if holistic_est: holistic_est.close()
                if selfie_seg_est: selfie_seg_est.close()
                hands = mp_hands.Hands() if states[0] or hand_emoji != 0 else None
                # Enable face detection for face mesh, face emoji, eye detection, OR AR accessories
                if states[1] or face_emoji != 0 or eye_toggle or ar_accessory != 0:
                    face = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=5)
                else:
                    face = None
                pose_est = mp_pose.Pose() if states[2] else None
                holistic_est = mp_holistic.Holistic() if states[3] else None
                selfie_seg_est = mp_selfie_segmentation.SelfieSegmentation() if states[4] else None
                last_states = states.copy()
                last_style = style
                last_face_emoji = face_emoji
                last_hand_emoji = hand_emoji
                last_eye_toggle = eye_toggle

            gesture_emoji = ""
            eye_status = ""
            finger_count = 0

            # Holistic (includes face, hands, pose)
            if states[3] and holistic_est:
                results = holistic_est.process(rgb)
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            # --- Pose Estimation Only ---
            elif states[2] and pose_est:
                results = pose_est.process(rgb)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255,0,0), thickness=2))
            else:
                # Hand tracking and finger counting
                if (states[0] or hand_emoji != 0) and hands:
                    results = hands.process(rgb)
                    if results.multi_hand_landmarks:
                        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                            if states[0]:
                                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            # Gesture recognition
                            gesture_idx = recognize_hand_gesture(hand_landmarks)
                            gesture_emoji = self.hand_emoji_from_index(gesture_idx)
                            if gesture_emoji:
                                self.gesture_signal.emit(gesture_emoji)
                            else:
                                self.gesture_signal.emit("")
                            # Finger counting (sum for all hands, up to 10)
                            finger_count += count_fingers(hand_landmarks)
                            # Show emoji overlay
                            ih, iw, _ = frame.shape
                            wrist = hand_landmarks.landmark[0]
                            tip = hand_landmarks.landmark[12]
                            x = int(wrist.x * iw)
                            y = int(wrist.y * ih)
                            x2 = int(tip.x * iw)
                            y2 = int(tip.y * ih)
                            size = int(math.hypot(x2 - x, y2 - y) * 1.5)
                            # Determine overlay emoji
                            overlay_emoji = ""
                            flip = False
                            if self.auto_hand_emoji():
                                overlay_emoji = gesture_emoji
                                # Flip emoji for left hand (MediaPipe: handedness not directly available, so use x position)
                                if x < iw // 2 and overlay_emoji in ["✌️", "👉", "🤘", "👌", "🖖"]:
                                    flip = True
                            elif hand_emoji != 0:
                                overlay_emoji = self.hand_emoji_from_index(hand_emoji)
                                # Flip for left side if needed
                                if x < iw // 2 and overlay_emoji in ["✌️", "👉", "🤘", "👌", "🖖"]:
                                    flip = True
                            # Draw emoji (with flip for left hand)
                            if overlay_emoji:
                                if flip:
                                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                    try:
                                        font = ImageFont.truetype("seguiemj.ttf", size)
                                    except:
                                        font = ImageFont.load_default()
                                    emoji_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
                                    draw = ImageDraw.Draw(emoji_img)
                                    draw.text((0, 0), overlay_emoji, font=font, embedded_color=True)
                                    emoji_img = emoji_img.transpose(Image.FLIP_LEFT_RIGHT)
                                    # Calculate paste position and crop if needed
                                    paste_x = max(0, min(x - size // 2, frame.shape[1] - size))
                                    paste_y = max(0, min(y - size // 2, frame.shape[0] - size))
                                    img_pil.paste(emoji_img, (paste_x, paste_y), emoji_img)
                                    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                                else:
                                    frame = draw_emoji_on_frame(frame, overlay_emoji, (x - size // 2, y - size // 2), size)
                    self.finger_signal.emit(f"Fingers Up: {min(finger_count,10)}")
                else:
                    self.gesture_signal.emit("")
                    self.finger_signal.emit("Fingers Up: 0")
                # Face mesh, emoji, AR overlays, and eye detection
                face_count = 0
                # Always process face landmarks if mesh, emoji, or AR accessory is selected
                ar_accessory = self.ar_accessory_style()
                need_face = states[1] or face_emoji != 0 or eye_toggle or ar_accessory != 0
                if need_face and face:
                    results = face.process(rgb)
                    if results.multi_face_landmarks:
                        face_count = len(results.multi_face_landmarks)
                        for face_landmarks in results.multi_face_landmarks:
                            ih, iw, _ = frame.shape
                            # Eye state detection and overlay
                            if eye_toggle:
                                left_open, right_open, left_center, right_eye_center = eye_state(face_landmarks, iw, ih)
                                eye_status = f"Left Eye: {'Open' if left_open else 'Closed'}, Right Eye: {'Open' if right_open else 'Closed'}"
                                # Overlay red cross if closed
                                left_pupil = face_landmarks.landmark[468]
                                right_pupil = face_landmarks.landmark[473]
                                left_pupil_center = (int(left_pupil.x * iw), int(left_pupil.y * ih))
                                right_pupil_center = (int(right_pupil.x * iw), int(right_pupil.y * ih))
                                cross_size = 38
                                if not left_open:
                                    frame = draw_emoji_on_frame(
                                        frame, "❌",
                                        (left_pupil_center[0] - cross_size // 2, left_pupil_center[1] - cross_size // 2),
                                        cross_size
                                    )
                                if not right_open:
                                    frame = draw_emoji_on_frame(
                                        frame, "❌",
                                        (right_pupil_center[0] - cross_size // 2, right_pupil_center[1] - cross_size // 2),
                                        cross_size
                                    )
                                self.eye_signal.emit(eye_status)
                            else:
                                self.eye_signal.emit("")
                                
                            # --- Face Mesh Styles ---
                            style = self.face_mesh_style()
                            face_emoji = self.face_emoji_style()
                            
                            # Only process face mesh and emoji if eye detection is NOT enabled
                            if not eye_toggle:
                                # Draw mesh only if mesh checkbox is checked
                                if face_emoji == 0 and self.show_face_mesh():
                                    if style == 0:
                                        mp_drawing.draw_landmarks(
                                            frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                            mp_drawing.DrawingSpec(color=(80,256,80), thickness=1)
                                        )
                                    elif style == 1:
                                        mp_drawing.draw_landmarks(
                                            frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                            mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(0,255,255), thickness=2)
                                        )
                                    elif style == 2:
                                        mp_drawing.draw_landmarks(
                                            frame, face_landmarks, mp_face_mesh.FACEMESH_IRISES,
                                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(255,255,0), thickness=2)
                                        )
                                        mp_drawing.draw_landmarks(
                                            frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                            mp_drawing.DrawingSpec(color=(0,128,255), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(255,255,255), thickness=1)
                                        )
                                # --- Face Emoji Overlay ---
                                if face_emoji != 0:
                                    emoji = self.face_emoji_from_index(face_emoji)
                                    if face_emoji == 5:  # Crown
                                        x, y = get_crown_position(face_landmarks, iw, ih)
                                        left_cheek = face_landmarks.landmark[234]
                                        right_cheek = face_landmarks.landmark[454]
                                        size = int(abs(left_cheek.x * iw - right_cheek.x * iw) * 1.3)
                                        size = max(size, 40)
                                        frame = draw_emoji_on_frame(frame, emoji, (x - size // 2, y - size // 2), size)
                                    else:
                                        # Center emoji at the nose tip using centered paste
                                        nose_tip = face_landmarks.landmark[1]
                                        xc = int(nose_tip.x * iw)
                                        yc = int(nose_tip.y * ih)
                                        left_cheek = face_landmarks.landmark[234]
                                        right_cheek = face_landmarks.landmark[454]
                                        chin = face_landmarks.landmark[152]
                                        forehead = face_landmarks.landmark[10]
                                        face_width = abs(left_cheek.x * iw - right_cheek.x * iw)
                                        face_height = abs(forehead.y * ih - chin.y * ih)
                                        size = int(max(face_width, face_height) * 1.25)
                                        size = max(size, 40)
                                        frame = draw_centered_emoji_on_frame(frame, emoji, (xc, yc), size)
                            
                            # --- AR Accessories (always process, regardless of eye detection) ---
                            if ar_accessory == 1 and self.glasses_img is not None:
                                # Overlay glasses
                                left_eye = face_landmarks.landmark[33]
                                right_eye = face_landmarks.landmark[263]
                                x1, y1 = int(left_eye.x * iw), int(left_eye.y * ih)
                                x2, y2 = int(right_eye.x * iw), int(right_eye.y * ih)
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                eye_dist = int(math.hypot(x2 - x1, y2 - y1))
                                glasses_w = int(eye_dist * 2.1)
                                glasses_h = int(glasses_w * self.glasses_img.height / self.glasses_img.width)
                                angle = -math.degrees(math.atan2(y2 - y1, x2 - x1))
                                glasses = self.glasses_img.resize((glasses_w, glasses_h), Image.LANCZOS).rotate(angle, expand=True)
                                gy = int(cy - glasses.height // 2 - eye_dist * 0.35)
                                gx = int(cx - glasses.width // 2)
                                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                img_pil.paste(glasses, (gx, gy), glasses)
                                frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            if ar_accessory == 2:
                                frame = self.overlay_moustache(frame, face_landmarks, iw, ih)
                    self.face_count_signal.emit(face_count)
                else:
                    self.face_count_signal.emit(0)

            if states[4] and selfie_seg_est:
                results = selfie_seg_est.process(rgb)
                if results.segmentation_mask is not None:
                    mask = results.segmentation_mask > 0.5
                    frame = np.where(mask[..., None], frame, 0)

            # Get the current QLabel size and resize the frame to fit it
            label_w = self.parent().image_label.width() if hasattr(self.parent(), "image_label") else 640
            label_h = self.parent().image_label.height() if hasattr(self.parent(), "image_label") else 480
            # display_frame = cv2.resize(frame, (label_w, label_h))
            self.change_pixmap_signal.emit(frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        if not use_screen and cap:
            cap.release()
        if hands: hands.close()
        if face: face.close()
        if pose_est: pose_est.close()
        if holistic_est: holistic_est.close()
        if selfie_seg_est: selfie_seg_est.close()

    def stop(self):
        self.running = False
        self.wait()

    def face_emoji_from_index(self, idx):
        emojis = ["", "😎", "🤓", "😂", "😺", "👑"]
        return emojis[idx] if idx < len(emojis) else ""

    def hand_emoji_from_index(self, idx):
        # 0=None, 1=👍, 2=👋, 3=✌️, 4=🤙, 5=🖐️, 6=✊, 7=🔥, 8=🤘, 9=👌, 10=🖖, 11=👉
        emojis = ["", "👍", "👋", "✌️", "🤙", "🖐️", "✊", "🔥", "🤘", "👌", "🖖", "👉"]
        return emojis[idx] if idx < len(emojis) else ""

    def overlay_moustache(self, frame, face_landmarks, iw, ih):
        """Overlay the moustache image under the nose, following mouth and nose tilt."""
        if self.moustache_img is None:
            return frame
        # Use landmarks 1 (nose tip), 13 (upper lip), 61 (left mouth), 291 (right mouth)
        nose_tip = face_landmarks.landmark[1]
        upper_lip = face_landmarks.landmark[13]
        left_mouth = face_landmarks.landmark[61]
        right_mouth = face_landmarks.landmark[291]
        x1, y1 = int(left_mouth.x * iw), int(left_mouth.y * ih)
        x2, y2 = int(right_mouth.x * iw), int(right_mouth.y * ih)
        # Center between mouth corners
        mx = (x1 + x2) // 2
        # Place just below the nose tip, but above the upper lip
        nose_x, nose_y = int(nose_tip.x * iw), int(nose_tip.y * ih)
        lip_y = int(upper_lip.y * ih)
        # Moustache width: mouth width * 1.15
        moustache_w = int(math.hypot(x2 - x1, y2 - y1) * 1.15)
        moustache_h = int(moustache_w * self.moustache_img.height / self.moustache_img.width)
        # Angle (roll)
        angle = -math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Resize and rotate
        moustache = self.moustache_img.resize((moustache_w, moustache_h), Image.LANCZOS).rotate(angle, expand=True)
        # Place just below the nose tip, but not too low (between nose and upper lip)
        vertical_offset = int((lip_y - nose_y) * 0.35 + moustache_h * 0.08)
        gy = nose_y + vertical_offset - moustache.height // 2
        gx = mx - moustache.width // 2
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_pil.paste(moustache, (gx, gy), moustache)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class MediaPipeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MediaPipe Feature Selector (PyQt)")
        # self.setGeometry(100, 100, 1100, 650)

        # Apply dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #232629;
                color: #f8f8f2;
                font-size: 14px;
            }
            QCheckBox, QLabel {
                color: #f8f8f2;
            }
            QPushButton {
                background-color: #44475a;
                color: #f8f8f2;
                border: 1px solid #282a36;
                border-radius: 4px;
                padding: 5px 15px;
            }
            QPushButton:disabled {
                background-color: #282a36;
                color: #888;
            }
            QCheckBox::indicator {
                border: 1px solid #44475a;
                background: #282a36;
            }
            QCheckBox::indicator:checked {
                background: #6272a4;
                border: 1px solid #44475a;
            }
            QComboBox {
                background-color: #282a36;
                color: #f8f8f2;
                border: 1px solid #44475a;
                border-radius: 4px;
                padding: 2px 10px;
            }
        """)

        # Camera dropdown
        self.camera_box = QComboBox()
        self.camera_box.setFixedWidth(120)
        self.populate_cameras()

        # Face mesh style dropdown
        self.face_mesh_style_box = QComboBox()
        self.face_mesh_style_box.addItem("Tesselation (Default)", 0)
        self.face_mesh_style_box.addItem("Contours (Bold)", 1)
        self.face_mesh_style_box.addItem("Irises + Contours", 2)
        self.face_mesh_style_box.setFixedWidth(170)

        # Face emoji dropdown
        self.face_emoji_box = QComboBox()
        self.face_emoji_box.addItem("No Face Emoji", 0)
        self.face_emoji_box.addItem("😎 Sunglasses", 1)
        self.face_emoji_box.addItem("🤓 Nerd", 2)
        self.face_emoji_box.addItem("😂 Laugh", 3)
        self.face_emoji_box.addItem("😺 Cat", 4)
        self.face_emoji_box.addItem("👑 Crown", 5)
        self.face_emoji_box.setFixedWidth(130)

        # Hand emoji dropdown (add new gestures)
        self.hand_emoji_box = QComboBox()
        self.hand_emoji_box.addItem("No Hand Emoji", 0)
        self.hand_emoji_box.addItem("👍 Thumbs Up", 1)
        self.hand_emoji_box.addItem("👋 Wave", 2)
        self.hand_emoji_box.addItem("✌️ Peace", 3)
        self.hand_emoji_box.addItem("🤙 Call Me", 4)
        self.hand_emoji_box.addItem("🖐️ High Five", 5)
        self.hand_emoji_box.addItem("✊ Fist", 6)
        self.hand_emoji_box.addItem("🔥 Fire", 7)
        self.hand_emoji_box.addItem("🤘 Rock", 8)
        self.hand_emoji_box.addItem("👌 OK", 9)
        self.hand_emoji_box.addItem("🖖 Vulcan", 10)
        self.hand_emoji_box.addItem("👉 Point", 11)
        self.hand_emoji_box.setFixedWidth(130)

        # AR accessory dropdown
        self.ar_accessory_box = QComboBox()
        self.ar_accessory_box.addItem("No Accessory", 0)
        self.ar_accessory_box.addItem("Glasses", 1)
        self.ar_accessory_box.addItem("Moustache", 2)
        self.ar_accessory_box.setFixedWidth(130)

        # Feature checkboxes
        self.hand_cb = QCheckBox("Hand Tracking")
        self.face_mesh_cb = QCheckBox("Face Mesh")
        self.pose_cb = QCheckBox("Pose Estimation")
        self.holistic_cb = QCheckBox("Holistic")
        self.selfie_seg_cb = QCheckBox("Selfie Segmentation")

        # Eye detection toggle
        self.eye_toggle_cb = QCheckBox("Eye Detection")
        self.eye_toggle_cb.setChecked(False)

        # Auto hand emoji toggle
        self.auto_hand_emoji_cb = QCheckBox("Auto Hand Emoji")
        self.auto_hand_emoji_cb.setChecked(False)

        # Gesture output label
        self.gesture_label = QLabel("Hand Gesture: ")
        self.gesture_label.setFixedWidth(260)
        self.gesture_label.setStyleSheet("font-size: 22px; color: #f1fa8c;")
        self.gesture_label.setWordWrap(True)
        # Eye output label
        self.eye_label = QLabel("Eye State: ")
        self.eye_label.setFixedWidth(260)
        self.eye_label.setStyleSheet("font-size: 18px; color: #8be9fd;")
        self.eye_label.setWordWrap(True)
        # Finger count label
        self.finger_label = QLabel("Fingers Up: 0")
        self.finger_label.setFixedWidth(260)
        self.finger_label.setStyleSheet("font-size: 18px; color: #50fa7b;")
        self.finger_label.setWordWrap(True)
        # Face count label
        self.face_count_label = QLabel("Faces: 0")
        self.face_count_label.setFixedWidth(260)
        self.face_count_label.setStyleSheet("font-size: 18px; color: #ffb86c;")
        self.face_count_label.setWordWrap(True)

        # Buttons
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        # Video label (left side)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")

        # --- Controls panel (right, top) ---
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(QLabel("Camera:"))
        controls_layout.addWidget(self.camera_box)
        controls_layout.addWidget(self.hand_cb)
        controls_layout.addWidget(self.face_mesh_cb)
        controls_layout.addWidget(QLabel("Face Mesh Style:"))
        controls_layout.addWidget(self.face_mesh_style_box)
        controls_layout.addWidget(QLabel("Face Emoji:"))
        controls_layout.addWidget(self.face_emoji_box)
        controls_layout.addWidget(QLabel("Hand Emoji:"))
        controls_layout.addWidget(self.hand_emoji_box)
        controls_layout.addWidget(QLabel("AR Accessory:"))
        controls_layout.addWidget(self.ar_accessory_box)
        controls_layout.addWidget(self.pose_cb)
        controls_layout.addWidget(self.holistic_cb)
        controls_layout.addWidget(self.selfie_seg_cb)
        controls_layout.addWidget(self.eye_toggle_cb)
        controls_layout.addWidget(self.auto_hand_emoji_cb)
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch(1)

        # --- Output panel (right, bottom) ---
        output_layout = QVBoxLayout()
        self.output_title = QLabel("Output")
        self.output_title.setStyleSheet("font-size: 24px; color: #f1fa8c; font-weight: bold;")
        output_layout.addWidget(self.output_title)
        output_layout.addWidget(self.gesture_label)
        output_layout.addWidget(self.eye_label)
        output_layout.addWidget(self.finger_label)
        output_layout.addWidget(self.face_count_label)
        output_layout.addStretch(1)

        # --- Right side: controls on top, output below ---
        right_panel = QVBoxLayout()
        right_panel.addLayout(controls_layout, stretch=2)
        right_panel.addLayout(output_layout, stretch=1)

        # --- Main content: video left, right panel right ---
        content_layout = QHBoxLayout()
        content_layout.addWidget(self.image_label, stretch=4)
        content_layout.addLayout(right_panel, stretch=2)

        # --- Main vertical layout ---
        main_layout = QVBoxLayout()
        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

        # Thread
        self.thread = None

        # Signals
        self.start_btn.clicked.connect(self.start_video)
        self.stop_btn.clicked.connect(self.stop_video)

    def populate_cameras(self):
        self.camera_box.clear()
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_box.addItem(f"Camera {i}", i)
                cap.release()
        # Add screen capture option
        self.camera_box.addItem("Screen Capture", -1)

    def feature_states(self):
        # Always enable face detection if any AR accessory is selected
        ar_selected = self.ar_accessory_style() in [1, 2]  # 1=Glasses, 2=Moustache
        return [
            self.hand_cb.isChecked(),
            self.face_mesh_cb.isChecked() or ar_selected,  # face detection ON for mesh or any AR accessory
            self.pose_cb.isChecked(),
            self.holistic_cb.isChecked(),
            self.selfie_seg_cb.isChecked()
        ]

    def show_face_mesh(self):
        # Only show mesh if checkbox is checked
        return self.face_mesh_cb.isChecked()

    def face_mesh_style(self):
        return self.face_mesh_style_box.currentData()

    def face_emoji_style(self):
        return self.face_emoji_box.currentData()

    def hand_emoji_style(self):
        return self.hand_emoji_box.currentData()

    def eye_toggle(self):
        return self.eye_toggle_cb.isChecked()

    def auto_hand_emoji(self):
        return self.auto_hand_emoji_cb.isChecked()

    def ar_accessory_style(self):
        return self.ar_accessory_box.currentData()

    def start_video(self):
        if not self.thread or not self.thread.isRunning():
            cam_index = self.camera_box.currentData()
            self.thread = VideoThread(
                self.feature_states,
                self.face_mesh_style,
                self.face_emoji_style,
                self.hand_emoji_style,
                self.eye_toggle,
                self.auto_hand_emoji,
                camera_index=cam_index,
                ar_accessory_style=self.ar_accessory_style,
                show_face_mesh=self.show_face_mesh
            )
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.gesture_signal.connect(self.update_gesture)
            self.thread.eye_signal.connect(self.update_eye)
            self.thread.finger_signal.connect(self.update_finger)
            self.thread.face_count_signal.connect(self.update_face_count)
            self.thread.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

    def stop_video(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.image_label.clear()
            self.image_label.setStyleSheet("background-color: black;")
            self.gesture_label.setText("Hand Gesture: ")
            self.eye_label.setText("Eye State: ")
            self.finger_label.setText("Fingers Up: 0")
            self.face_count_label.setText("Faces: 0")

    def closeEvent(self, event):
        self.stop_video()
        event.accept()

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_gesture(self, emoji):
        if emoji:
            self.gesture_label.setText(f"Hand Gesture: {emoji}")
        else:
            self.gesture_label.setText("Hand Gesture: ")

    def update_eye(self, text):
        self.eye_label.setText(f"Eye State: {text}" if text else "Eye State: ")

    def update_finger(self, text):
        self.finger_label.setText(text)

    def update_face_count(self, count):
        self.face_count_label.setText(f"Faces: {count}")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            if self.isFullScreen():
                # Hide top controls
                for i in range(self.layout().itemAt(0).count()):
                    widget = self.layout().itemAt(0).itemAt(i).widget()
                    if widget:
                        widget.hide()
                # Enlarge outputs only (video will auto-resize)
                self.gesture_label.setStyleSheet("font-size: 38px; color: #f1fa8c; background: rgba(30,30,30,180);")
                self.eye_label.setStyleSheet("font-size: 28px; color: #8be9fd; background: rgba(30,30,30,180);")
                self.finger_label.setStyleSheet("font-size: 28px; color: #50fa7b; background: rgba(30,30,30,180);")
                self.face_count_label.setStyleSheet("font-size: 28px; color: #ffb86c; background: rgba(30,30,30,180);")
                self.output_title.setStyleSheet("font-size: 32px; color: #f1fa8c; font-weight: bold; background: rgba(30,30,30,180);")
            else:
                # Restore controls
                for i in range(self.layout().itemAt(0).count()):
                    widget = self.layout().itemAt(0).itemAt(i).widget()
                    if widget:
                        widget.show()
                self.gesture_label.setStyleSheet("font-size: 22px; color: #f1fa8c;")
                self.eye_label.setStyleSheet("font-size: 18px; color: #8be9fd;")
                self.finger_label.setStyleSheet("font-size: 18px; color: #50fa7b;")
                self.face_count_label.setStyleSheet("font-size: 18px; color: #ffb86c;")
                self.output_title.setStyleSheet("font-size: 24px; color: #f1fa8c; font-weight: bold;")
        super().changeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MediaPipeApp()
    win.show()
    sys.exit(app.exec_())