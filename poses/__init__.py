# poses/pose.py

class Pose:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def get_info(self):
        return f"Pose: {self.name}\nDescription: {self.description}"


