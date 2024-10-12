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
from .asans._1_Shwanasana import detect_pose as detect_pose_1_Shwanasana
from .asans._2_Marjarasana_A import detect_pose as detect_pose_2_Marjarasana_A
from .asans._3_Marjarasana_B import detect_pose as detect_pose_3_Marjarasana_B
from .asans._4_Tripad_marjarasana_1 import detect_pose as detect_pose_4_Tripad_marjarasana_1
from .asans._4_Tripad_marjarasana_2 import detect_pose as detect_pose_4_Tripad_marjarasana_2
from .asans._5_Swastikasana import detect_pose as detect_pose_5_Swastikasana
from .asans._6_Hastapadasana import detect_pose as detect_pose_6_Hastapadasana
from .asans._7_Ardha_shalabhasana_1 import detect_pose as detect_pose_7_Ardha_shalabhasana_1
from .asans._7_Ardha_shalabhasana_2 import detect_pose as detect_pose_7_Ardha_shalabhasana_2
from .asans._8_Uttitha_ekapadasana_1 import detect_pose as detect_pose_8_Uttitha_ekapadasana_1
from .asans._8_Uttitha_ekapadasana_2 import detect_pose as detect_pose_8_Uttitha_ekapadasana_2
from .asans._9_Ardha_pavan_muktasana_1 import detect_pose as detect_pose_9_Ardha_pavan_muktasana_1
from .asans._9_Ardha_pavan_muktasana_2 import detect_pose as detect_pose_9_Ardha_pavan_muktasana_2
from .asans._10_Uttana_vakrasana import detect_pose as detect_pose_10_Uttana_vakrasana
from .asans._11_Uttana_tadasana import detect_pose as detect_pose_11_Uttana_tadasana
from .asans._12_dwipadasana import detect_pose as detect_pose_12_dwipadasana
from .asans._13_left_Ardha_padmasana import detect_pose as detect_pose_13_left_Ardha_padmasana
from .asans._14_left_eka_pada_hastasana import detect_pose as detect_pose_14_left_eka_pada_hastasana
from .asans._15_Left_Janushirasana import detect_pose as detect_pose_15_Left_Janushirasana
from .asans._16_Left_Vakrasana import detect_pose as detect_pose_16_Left_Vakrasana
from .asans._17_Pavan_muktasana import detect_pose as detect_pose_17_Pavan_muktasana
from .asans._18_Right_ardha_padmasana import detect_pose as detect_pose_18_Right_ardha_padmasana
from .asans._19_right_Janushirasana import detect_pose as detect_pose_19_right_Janushirasana
from .asans._20_Right_vakrasana import detect_pose as detect_pose_20_Right_vakrasana
from .asans._21_Vajrasana import detect_pose as detect_pose_21_Vajrasana
from .asans._22_Viparita_Karni_mudra import detect_pose as detect_pose_22_Viparita_Karni_mudra
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

            "SHWANASANA": detect_pose_1_Shwanasana,
            "MARJARASANA_A": detect_pose_2_Marjarasana_A,
            "MARJARASANA_B": detect_pose_3_Marjarasana_B,
            "TRIPAD_MARJARASANA_1": detect_pose_4_Tripad_marjarasana_1,
            "TRIPAD_MARJARASANA_2": detect_pose_4_Tripad_marjarasana_2,
            "SWASTIKASANA": detect_pose_5_Swastikasana,
            "HASTAPADASANA": detect_pose_6_Hastapadasana,
            "ARDHA_SHALABHASANA_1": detect_pose_7_Ardha_shalabhasana_1,
            "ARDHA_SHALABHASANA_2": detect_pose_7_Ardha_shalabhasana_2,
            "UTTITHA_EKAPADASANA_1": detect_pose_8_Uttitha_ekapadasana_1,
            "UTTITHA_EKAPADASANA_2": detect_pose_8_Uttitha_ekapadasana_2,
            "ARDHA_PAVAN_MUKTASANA_1": detect_pose_9_Ardha_pavan_muktasana_1,
            "ARDHA_PAVAN_MUKTASANA_2": detect_pose_9_Ardha_pavan_muktasana_2,
            "UTTANA_VAKRASANA": detect_pose_10_Uttana_vakrasana,
            "UTTANA_TADASANA": detect_pose_11_Uttana_tadasana,
            "DWIPADASANA": detect_pose_12_dwipadasana,
            "LEFT_ARDHA_PADMASANA": detect_pose_13_left_Ardha_padmasana,
            "LEFT_EKA_PADA_HASTASANA": detect_pose_14_left_eka_pada_hastasana,
            "LEFT_JANUSHIRASANA": detect_pose_15_Left_Janushirasana,
            "LEFT_VAKRASANA": detect_pose_16_Left_Vakrasana,
            "PAVAN_MUKTASANA": detect_pose_17_Pavan_muktasana,
            "RIGHT_ARDHA_PADMASANA": detect_pose_18_Right_ardha_padmasana,
            "RIGHT_JANUSHIRASANA": detect_pose_19_right_Janushirasana,
            "RIGHT_VAKRASANA": detect_pose_20_Right_vakrasana,
            "VAJRASANA_NEW": detect_pose_21_Vajrasana,
            "VIPARITA_KARNI_MUDRA": detect_pose_22_Viparita_Karni_mudra,
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
