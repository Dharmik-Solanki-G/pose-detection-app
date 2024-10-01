# poses/asans/pose_detection.py
from .asans.parvatasana import detect_parvatasana
from .asans.AnandaBalasana import detect_pose as detect_ananda_balasan
from .asans.ArdhaChakrasana import detect_pose as detect_ardha_chakrasana
from .asans.ArdhaPadmasana import detect_pose as detect_ardha_padmasana
from .asans.Bhujangasana import detect_pose as detect_bhujangasana
from .asans.Hastauttanasana import detect_pose as detect_hastauttanasana
from .asans.navasana import detect_pose as detect_navasana
from .asans.Phalakasana import detect_pose as detect_phalakasana
from .asans.Paschimottanasana import detect_pose as detect_paschimottanasana
from .asans.Pranamasana import detect_pose as detect_pranamasana
from .asans.purvamatsyasana import detect_pose as detect_purvamatsyasana
from .asans.savasana import detect_pose as detect_savasana
from .asans.setubandasana import detect_pose as detect_setubandasana
from .asans.Supta_Baddha_Konasana import detect_pose as detect_supta_baddha_konasana
from .asans.SuptaMatsyendrasana import detect_pose as detect_supta_matsyendrasana
from .asans.svanasana import detect_pose as detect_svanasana
from .asans.Utkatasana import detect_pose as detect_utkatasana
from .asans.Uttanasana import detect_pose as detect_uttanasana
from .asans.ViparitaKarani import detect_pose as detect_viparita_karani
from .asans.vajrasana import detect_pose as detect_vajrasana
from .asans.Virabhadrasana_I import detect_pose as detect_virabhadrasana_I
from .asans.Virabhadrasana_II import detect_pose as detect_virabhadrasana_II
# from .another_pose import detect_another_pose   # Add other poses here

class PoseDetection:
    def __init__(self):
        # Map instructions to their respective pose detection functions
        self.pose_functions = {
            "PARVATASANA": detect_parvatasana,
            "ANANDA_BALASANA": detect_ananda_balasan,
            "ARDHA_CHAKRASANA": detect_ardha_chakrasana,
            "ARDHA_PADMASANA": detect_ardha_padmasana,
            "BHUJANGASANA": detect_bhujangasana,
            "HASTAUTTANASANA": detect_hastauttanasana,
            "NAVASANA": detect_navasana,
            "PHALAKASANA": detect_phalakasana,
            "PASCHIMOTTANASANA": detect_paschimottanasana,
            "PRANAMASANA": detect_pranamasana,
            "PURVAMATSYASANA": detect_purvamatsyasana,
            "SAVASANA": detect_savasana,
            "SETUBANDASANA": detect_setubandasana,
            "SUPTA_BADDHA_KONASANA": detect_supta_baddha_konasana,
            "SUPTA_MATSYENDRASANA": detect_supta_matsyendrasana,
            "SVANASANA": detect_svanasana,
            "UTKATASANA": detect_utkatasana,
            "UTTANASANA": detect_uttanasana,
            "VIPARITA_KARANI": detect_viparita_karani,
            "VAJRASANA": detect_vajrasana,
            "VIRABHADRASANA_I": detect_virabhadrasana_I,
            "VIRABHADRASANA_II": detect_virabhadrasana_II,
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
