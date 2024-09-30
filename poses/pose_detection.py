# poses/pose_detection.py

from .parvatasana import detect_parvatasana
# from .another_pose import detect_another_pose   # Add other poses here

class PoseDetection:
    def __init__(self):
        # Map instructions to their respective pose detection functions
        self.pose_functions = {
            "PARVATASANA": detect_parvatasana,
            # Add other pose mappings here
            # "another_pose": detect_another_pose,
        }

    def analyze_pose(self, instructions, landmarks):
        """ Analyze the provided pose landmarks based on instructions. """
        pose_function = self.pose_functions.get(instructions.upper())

        if pose_function is None:
            return 0.0, "Unknown pose", [] , ''  # Handle unknown instructions gracefully
        
        accuracy, pose_name, correct , feedback_str = pose_function(landmarks)
        return accuracy, pose_name, correct , feedback_str
