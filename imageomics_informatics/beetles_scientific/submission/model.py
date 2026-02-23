"""
V124 - Single MLP LP-FT BioCLIP-2 (V123 Fold 1, R²=0.4198) + Bias Correction
===============================================================================
架構：單模型推論（無 stacking），對齊 V118 proven format。
Sigma：OOF RMSE per target (SPEI_30d=0.90, SPEI_1y=0.77, SPEI_2y=0.72)
Bias：domain_weight=0.5, species_weight=0.2 (same as V118)
"""
import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np


class Model:
    def __init__(self, context=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        # BioCLIP / OpenCLIP mean & std
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

        # Fixed sigma from V123 OOF RMSE per target
        self.sigma_default = {
            "SPEI_30d": 0.90,
            "SPEI_1y":  0.77,
            "SPEI_2y":  0.72,
        }

        # Bias correction
        self.domain_weight  = 0.5
        self.species_weight = 0.2

        self.domain_bias = {
            1:   [-0.0404, -0.0491, -0.4056],
            3:   [-0.0823, -0.2568, -0.4339],
            4:   [ 0.0165,  0.2167,  0.0890],
            7:   [-0.1241,  0.0409, -0.2842],
            9:   [ 0.1546,  0.0857, -0.0610],
            11:  [-0.0837,  0.2108, -0.0521],
            32:  [-0.1766, -0.0010, -0.3524],
            46:  [-0.2201, -0.1026, -0.3650],
            99:  [ 0.0775,  0.1659,  0.0531],
            202: [-0.0803,  0.0218,  0.0079],
        }

        self.species_bias = {
            "Agonum gratiosum": [0.3184, 0.4657, 0.3339],
            "Agonum limbatum": [0.2647, -0.3763, -0.6568],
            "Agonum placidum": [-0.0996, 0.1090, -0.2896],
            "Agonum punctiforme": [-0.6691, 0.4393, 0.8729],
            "Agonum retractum": [-0.3159, 0.7583, 0.4550],
            "Amara aenea": [-0.2725, 0.1512, 0.2031],
            "Amara californica": [0.3960, 0.0403, 0.0639],
            "Amara carinata": [-0.0755, 0.0547, -0.1036],
            "Amara coelebs": [0.1416, 0.3895, 0.6168],
            "Amara conflata": [0.4053, -0.0399, -0.2507],
            "Amara confusa": [0.2479, -0.3766, -0.2417],
            "Amara impuncticollis": [0.0647, 0.3485, -0.1285],
            "Amara littoralis": [-0.2113, -0.4384, 0.6778],
            "Amara obesa": [0.1755, 0.2243, 0.1265],
            "Amara quenseli": [-0.1598, -0.1462, -0.0717],
            "Amara scitula": [0.0532, 0.1758, 0.1540],
            "Amara tenax": [0.1421, -0.2372, 0.3209],
            "Anisodactylus carbonarius": [-0.3798, -0.0615, 0.1616],
            "Anisodactylus dulcicollis": [-0.2038, 0.6270, 0.7831],
            "Anisodactylus furvus": [-0.1363, 1.5993, 1.8143],
            "Anisodactylus haplomus": [-0.3123, 0.0344, 0.5091],
            "Anisodactylus merula": [-0.7756, -0.0741, -0.3014],
            "Anisodactylus opaculus": [-0.0100, -0.5656, -0.3659],
            "Anisodactylus ovularis": [0.1083, 0.6362, 0.6544],
            "Anisodactylus similis": [0.0366, 0.0174, -0.0243],
            "Brachinus alternans": [-0.1149, -0.1582, -0.0900],
            "Brachinus cyanochroaticus": [0.1914, -0.2508, 0.0447],
            "Brachinus fumans": [-0.3930, 0.7654, 0.3800],
            "Calathus opaculus": [0.0734, 0.2131, 0.4583],
            "Calosoma affine": [0.0590, -0.3821, -0.1306],
            "Calosoma calidum": [-0.3721, -0.2384, -0.2120],
            "Calosoma frigidum": [-0.8295, -0.2895, 0.5014],
            "Calosoma scrutator": [-1.0991, 0.3342, -0.0773],
            "Carabidae sp.": [0.4569, 0.0759, -0.5451],
            "Carabus goryi": [-0.1457, -0.2155, -0.1338],
            "Carabus serratus": [-0.2225, 0.0516, 0.1947],
            "Carabus sylvosus": [-0.1760, 0.2776, 0.6470],
            "Chlaenius aestivus": [-0.1502, 0.2587, 0.1965],
            "Chlaenius emarginatus": [0.0351, 0.3360, 0.1853],
            "Chlaenius erythropus": [-0.7680, -0.5308, -0.3130],
            "Chlaenius platyderus": [-1.3387, -0.2270, 0.2525],
            "Chlaenius sericeus": [-0.9554, 0.0878, 0.0961],
            "Chlaenius tomentosus": [0.0054, 0.2419, 0.2494],
            "Cicindela punctulata": [-0.1109, -0.2975, -0.2511],
            "Cicindela punctulata punctulata": [0.3797, 0.6086, 1.3189],
            "Cratacanthus dubius": [0.2054, 0.0299, 0.0646],
            "Cyclotrachelus constrictus": [-0.9013, -0.0638, -0.2149],
            "Cyclotrachelus convivus": [0.1138, -0.0902, -0.0037],
            "Cyclotrachelus faber": [-0.4175, -0.2715, 0.0605],
            "Cyclotrachelus freitagi": [0.0790, 0.4257, 0.4662],
            "Cyclotrachelus fucatus": [-0.1266, 0.0921, 0.1313],
            "Cyclotrachelus furtivus": [-0.1153, 0.6517, 0.6596],
            "Cyclotrachelus hypherpiformis": [-0.1724, -0.3361, -0.1047],
            "Cyclotrachelus seximpressus": [0.6902, 0.3985, -0.0917],
            "Cyclotrachelus sigillatus": [-0.0968, 0.2107, 0.2264],
            "Cyclotrachelus sodalis": [0.2057, -0.3579, -0.2220],
            "Cyclotrachelus sodalis colossus": [-0.4937, -0.4026, -0.1806],
            "Cyclotrachelus sodalis sodalis": [-0.2891, -0.0453, -0.3163],
            "Cyclotrachelus torvus": [-0.2970, -0.1143, -0.0423],
            "Cylindera unipunctata": [-0.0491, 0.2913, 0.3376],
            "Cymindis cribricollis": [0.3420, 0.2101, 0.3944],
            "Cymindis neglecta": [0.0786, 0.0927, 0.1557],
            "Cymindis planipennis": [-0.0859, -0.0761, -0.0658],
            "Dicaelus dilatatus": [-0.0603, 0.6293, 0.4769],
            "Dicaelus dilatatus dilatatus": [-0.3476, 0.2081, 0.4766],
            "Dicaelus dilatatus sinuatus": [-0.3562, -0.2798, -0.1859],
            "Dicaelus elongatus": [-0.3601, 0.3751, 0.4827],
            "Dicaelus furvus": [-0.0890, -0.0058, 0.0304],
            "Dicaelus furvus carinatus": [-0.0781, 0.1029, 0.5121],
            "Dicaelus furvus furvus": [-0.5788, -0.1957, -0.1026],
            "Dicaelus politus": [-0.6922, 1.2871, 1.3517],
            "Dicaelus purpuratus": [0.5670, 0.3831, -0.1510],
            "Dicaelus purpuratus purpuratus": [-0.1999, 1.2263, 1.1087],
            "Dicaelus purpuratus splendidus": [-0.7984, -0.3798, 0.2935],
            "Dicaelus sculptilis": [-1.2923, -0.8035, -0.4378],
            "Dicaelus sculptilis sculptilis": [-0.5013, -0.2367, 0.0834],
            "Dicaelus teter": [-0.0168, -0.1122, -0.0326],
            "Dicheirus piceus": [-0.1273, -0.0562, -0.2853],
            "Discoderus parallelus": [0.3404, -0.0503, -0.0407],
            "Galerita atripes": [-0.2688, -0.4343, -0.4912],
            "Galerita bicolor": [-0.2091, 0.1520, 0.2071],
            "Harpalus amputatus": [0.2540, 0.4910, 0.3503],
            "Harpalus amputatus amputatus": [-0.0185, 0.2953, 0.1849],
            "Harpalus desertus": [0.3984, -0.0592, 0.1391],
            "Harpalus ellipsis": [0.2867, -0.1162, 0.0920],
            "Harpalus herbivagus": [-0.2877, 0.0303, 0.3661],
            "Harpalus indigens": [-0.6776, -0.3010, -0.1532],
            "Harpalus opacipennis": [-0.0294, 0.3244, 0.4021],
            "Harpalus paratus": [-0.4163, 0.3277, 0.0015],
            "Harpalus pensylvanicus": [-0.2147, -0.2737, 0.0622],
            "Harpalus protractus": [-0.4632, -0.1617, 0.0011],
            "Harpalus providens": [-0.7994, -0.0074, 0.0954],
            "Harpalus reversus": [-0.0827, 0.2398, -0.1808],
            "Harpalus somnulentus": [-0.1456, 0.0772, 0.2645],
            "Harpalus spadiceus": [0.7071, 0.9243, 0.5180],
            "Harpalus vagans": [0.1412, 0.4133, 0.5972],
            "Harpalus ventralis": [0.3602, 0.5947, -0.2389],
            "Metrius contractus": [-0.0653, -0.0094, 0.0843],
            "Myas coracinus": [-0.2500, -0.0617, -0.0118],
            "Myas cyanescens": [0.3830, 0.9174, 1.1639],
            "Omus californicus": [-0.1192, 0.1998, 0.3437],
            "Pasimachus elongatus": [0.0287, -0.0288, 0.0604],
            "Pasimachus punctulatus": [-0.0952, 0.2801, 0.6483],
            "Pasimachus sublaevis": [-0.2774, -0.0355, 0.2818],
            "Pasimachus subsulcatus": [0.3070, 0.9640, 1.1176],
            "Piosoma setosum": [0.2303, 0.0078, 0.0766],
            "Platynus decentis": [-0.1104, 0.2350, 0.2037],
            "Poecilus chalcites": [-0.0807, -0.1294, 0.0609],
            "Poecilus lucublandus": [-0.1950, 0.0489, 0.2319],
            "Poecilus scitulus": [0.4027, 0.2170, -0.1390],
            "Pterostichus (Hypherpes) sp.": [-1.1572, -0.3313, 0.5785],
            "Pterostichus acutipes": [-0.3566, 0.1679, 0.7341],
            "Pterostichus acutipes acutipes": [0.4170, 0.2919, 0.3687],
            "Pterostichus adoxus": [-0.3142, 0.9232, 1.1294],
            "Pterostichus atratus": [-0.2270, 0.3165, 0.9549],
            "Pterostichus caudicalis": [-0.1626, -0.2796, 0.6032],
            "Pterostichus coracinus": [-0.0612, 0.0153, 0.1250],
            "Pterostichus lama": [-0.3751, -0.6518, -1.0538],
            "Pterostichus melanarius melanarius": [-0.1074, -0.0135, 0.1843],
            "Pterostichus moestus": [0.2496, 0.3608, 0.3027],
            "Pterostichus mutus": [-0.3697, 0.1920, 0.1795],
            "Pterostichus novus": [-0.0461, -0.2720, -0.0189],
            "Pterostichus ordinarius": [-0.2130, -0.1355, 0.0415],
            "Pterostichus panticulatus": [0.0193, -0.0839, -0.1456],
            "Pterostichus pensylvanicus": [-0.0913, 0.2068, 0.2150],
            "Pterostichus permundus": [-0.0632, 0.0390, 0.2997],
            "Pterostichus rostratus": [-0.2050, -0.4302, -0.3009],
            "Pterostichus serripes": [-0.8020, -0.5618, 0.1732],
            "Pterostichus stygicus": [-0.0795, 0.1270, 0.2442],
            "Pterostichus trinarius": [-0.1428, 0.3751, 0.3434],
            "Pterostichus tristis": [-0.0287, -0.2329, -0.1138],
            "Scaphinotus oreophilus": [-0.2832, -0.1489, -0.0476],
            "Scarites ocalensis": [0.0266, 0.3032, 0.4110],
            "Scarites subterraneus": [-0.5552, -0.5558, -0.2718],
            "Scarites vicinus": [-0.1427, 0.0479, 0.1362],
            "Sphaeroderus canadensis canadensis": [-0.1537, -0.1246, -0.0343],
            "Sphaeroderus stenostomus": [0.0113, 1.1763, 1.1702],
            "Sphaeroderus stenostomus lecontei": [-0.0946, -0.1126, 0.0417],
            "Sphaeroderus stenostomus stenostomus": [0.0206, 0.9115, 1.3543],
            "Synuchus impunctatus": [-0.1129, 0.0562, 0.1007],
            "Tetracha virginica": [-0.1595, -0.4110, -0.0383],
        }

    def load(self):
        print("Loading V124: MLP LP-FT BioCLIP-2 Fold 1 + Bias Correction...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "mlp_lpft_fold1_fp16.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}.")

    def predict(self, datapoints):
        if self.model is None:
            self.load()

        domain_id    = datapoints[0].get("domainID", None)
        species_name = datapoints[0].get("scientificName", None)

        all_preds = []

        with torch.no_grad():
            for item in datapoints:
                if not isinstance(item, dict):
                    continue
                img = item.get("relative_img") or item.get("image")
                if img is None:
                    continue

                if hasattr(img, "convert"):
                    img = img.convert("RGB")
                elif isinstance(img, np.ndarray):
                    from PIL import Image as PILImage
                    img = PILImage.fromarray(img.astype("uint8"), "RGB")

                t = self.transform(img).unsqueeze(0).to(self.device).half()
                pred = self.model(t).float().cpu().numpy()[0]  # (3,)
                all_preds.append(pred)

        if not all_preds:
            return self._default_output()

        # Average across batch (通常 batch=1)
        final_mu = np.mean(all_preds, axis=0)  # (3,)

        # Bias correction DISABLED — testing if frozen-BioCLIP OOF bias
        # still applies to fine-tuned MLP model
        pass

        targets = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]
        result = {}
        for i, target in enumerate(targets):
            result[target] = {
                "mu":    float(final_mu[i]),
                "sigma": self.sigma_default[target],
            }
        return result

    def _default_output(self):
        return {
            "SPEI_30d": {"mu": 0.0, "sigma": 0.90},
            "SPEI_1y":  {"mu": 0.0, "sigma": 0.77},
            "SPEI_2y":  {"mu": 0.0, "sigma": 0.72},
        }
