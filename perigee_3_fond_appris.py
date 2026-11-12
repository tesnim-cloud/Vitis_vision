# -*- coding: utf-8 -*-
"""
perigee_3_fond_appris.py — BRANCHE + NŒUD + OBLONG (d1, d2, SURFACE) + d3, d4 + PÉRIGÉE (d5)
MÉTHODE DE SEGMENTATION N° 3 (une seule méthode dans ce fichier) : FOND APPRIS
    écart de couleur, pixel par pixel, à une PHOTO DU FOND sans branche (touche b) + Otsu + seuil bas.
    Compense les ombres et l'éclairage inégal. AU DÉMARRAGE : enlever la branche, appuyer sur b, remettre la branche.
DÉTECTION « INTELLIGENTE » DU NŒUD (version du 12/11) :
    • un APPENDICE épais (bourgeon, moignon de rameau) qui dépasse d'un côté de plus de 0,40 × diamètre est
      COUPÉ À SA RACINE : le bord de la tige est remplacé par un « pont » (trait orange) entre les 2 racines ;
      l'oblong, les limites et les diamètres sont donc mesurés sur la TIGE, pas sur l'appendice ;
    • le nœud est le plus grand RENFLEMENT par rapport à la TENDANCE de la tige (branche conique), cherché
      près de l'appendice s'il y en a un ; sans renflement net, il est placé sur la racine de l'appendice.
Schéma « la périgée » :
    • OBLONG (jaune), d1 (violet), d2 (bleu), centre (rouge) : exactement comme branche_noeud_oblong.py ;
    • SURFACE de l'oblong = L × H − (4 − π) × r²   (rectangle L × H moins les 4 coins enlevés par les arrondis) ;
    • d3 / d4 = diamètre EXTÉRIEUR de la branche (d'un bord du contour à l'autre) à ses 2 BOUTS :
        d3 au bout d'ENTRÉE (gauche), d4 à l'autre bout (droite) — flèches noires ;
    • LIGNE D'ANGLE (rouge) : part du SOMMET de l'oblong (haut de l'axe en travers, au-dessus du centre).
      L'ANGLE (45° au départ, touches j / k) est mesuré EN HAUT, entre l'axe en travers de l'oblong (la
      verticale du nœud) et la ligne rouge. La ligne descend vers l'entrée jusqu'au bord opposé de la branche ;
    • PÉRIGÉE (jaune) = la perpendiculaire à la branche au point de contact ; d5 (vert) = diamètre sur la périgée.
    Calcul (image redressée, s le long de la branche, t en travers vers le bas) :
        ligne rouge : t(s) = t_sommet + |s − s_centre| / tan(angle)
        contact : premier s, vers l'entrée, où t(s) atteint le bord bas lissé (interpolé → sub-pixel).
Tout ce que fait branche_noeud.py (étape 1, mêmes réglages), PLUS :
    • l'OBLONG (jaune) autour du nœud : forme « stade » = rectangle aux bouts arrondis dont
        - les côtés gauche / droit sont les 2 LIMITES parallèles du nœud (lignes rouges de l'étape 1),
        - les côtés haut / bas sont les tangentes au point le plus HAUT et le plus BAS du nœud,
        - le rayon des arrondis = ½ × le plus petit côté ;
    • les 2 DIAMÈTRES DE L'OBLONG (comme sur le schéma) : 2 segments perpendiculaires qui se croisent au
      CENTRE de l'oblong et vont d'un bord de l'oblong à l'autre :
        - d1 = GRAND diamètre (violet) = le plus long des 2 axes de l'oblong,
        - d2 = PETIT diamètre (bleu)   = le plus court, perpendiculaire à d1 ;
      l'un est le long de la branche (= longueur entre les 2 limites), l'autre en travers (= hauteur du nœud).
FENÊTRE en 4 parties :
    1 (haut gauche)  image caméra : contour (vert), oblong (jaune), 2 limites du nœud (rouge),
                     d1 (violet), d2 (bleu) et le POINT CENTRAL de l'oblong (rouge) où ils se croisent
    2 (haut droite)  masque noir et blanc (blanc = branche)
    3 (bas gauche)   graphique de la méthode d'Otsu : histogramme + seuils
    4 (bas droite)   informations de l'oblong et des diamètres (en mm si calibré)
Lancer (PC Linux, sans venv) :
    python3 perigee_3_fond_appris.py            → caméra Basler n° 0
    python3 perigee_3_fond_appris.py 1          → caméra n° 1
    python3 perigee_3_fond_appris.py photo.png  → image enregistrée (sans caméra)
Touches :
    b  APPRENDRE LE FOND : enlever la branche, appuyer sur b, remettre la branche (à refaire si la lumière
       ou l'exposition change)
    c  CALIBRER : poser un étalon (tige de diamètre connu, ex. pige Ø 10,00 mm) à la place de la branche,
       taper son diamètre dans le terminal → mesure sur 15 images → calibration.json enregistré.
       Refaire c avec un 2e étalon (ex. Ø 6,00 mm) = calibration plus juste (droite D_mm = a·D_px + b).
    r  effacer la calibration (retour en pixels)
    a  moyenne sur 15 images (valeur ± écart-type dans le terminal)
    s  enregistrer image + résultat + 4 fenêtres (PNG) et une ligne dans mesures_perigee_m3.csv
    f  anti-vibration marche / arrêt (moyenne des images tant que la branche est immobile ; active au démarrage)
    j / k  angle de la ligne rouge − 1° / + 1° (départ 45°)
    p  afficher / cacher les réglages utilisés (fenêtre 4)
    +/- exposition
    q  quitter
Si calibration.json existe à côté du script (touche c), les mesures sont en mm, sinon en pixels.
IMPORTANT : l'étalon doit être à la MÊME distance de la caméra que la branche (même support / mandrin).
"""
import os
os.environ.setdefault("QT_QPA_PLATFORMTHEME", "none")       # évite un plantage Qt/GNOME sur Ubuntu (avant import cv2)
import sys, json, math, time, csv, warnings
import numpy as np
import cv2

# ---------------------------------------------------------------------------------------------------------
# Constantes de DÉFINITION (identiques pour toutes les branches — ce ne sont pas des réglages)
# ---------------------------------------------------------------------------------------------------------
BORD_NOEUD_PCT = 30.0      # limites du nœud : base + 30 % × (pic − base)
COUPE_APP_PCT = 50.0       # appendices plus fins que 50 % du diamètre (vrille, pétiole) ignorés pour trouver la tige
MARGE_BOUTS = 0.8          # bouts de la branche ignorés : 0,8 × diamètre à chaque bout (coupes)
N_MOYENNE = 15             # images moyennées avec la touche « a »
ANGLE_DEPART = 45.0        # angle (°) en haut, entre l'axe en travers de l'oblong et la ligne rouge (touches j / k)
ENTREE = "gauche"          # côté où la branche ENTRE : "gauche" ou "droite" (périgée et d3 de ce côté)
REGLAGE = {"angle": ANGLE_DEPART}   # valeur actuelle de l'angle (changée par les touches)
APP_HAUT = 0.40            # APPENDICE (bourgeon, départ de rameau, moignon) : tout ce qui dépasse le bord de la tige
                           # de plus de 0,40 × diamètre d'UN côté. Il est coupé à sa racine (le nœud, lui, dépasse moins).
APP_PENTE = 0.30           # racine de l'appendice = là où le bord redevient presque parallèle à la tige (pente < 0,30 ≈ 17°)
N_STAB = 30                # STABILISATION : moyenne des images tant que la branche ne bouge pas (jusqu'à 30 images)
SEUIL_MVT = 12.0           # un pixel (image réduite 8×) a « bougé » si son gris change de plus de 12 niveaux…
PART_MVT = 0.002           # … et la branche a bougé si plus de 0,2 % des pixels ont bougé → la moyenne repart de zéro
BOUT_POS = 0.30            # d3 / d4 mesurés à 0,30 × diamètre du bout (on évite la face de coupe)…
BOUT_LARG = 0.20           # … sur une petite bande de 0,20 × diamètre (médiane des colonnes de la bande)

# ---------------------------------------------------------------------------------------------------------
# Réglages de SEGMENTATION : ceux de segmentation_branche.py qui donnent un bon résultat sur la vraie branche
# ---------------------------------------------------------------------------------------------------------
EXPO_US = 7000.0           # exposition (µs) — modifiable avec + / -
FLOU = 10                  # flou 2×10+1 = 21×21 (pour le MASQUE seulement)
MORPHO = 3                 # curseur « Morpho » = 3 → noyau 2×3+1 = 7×7
AIRE_MIN = 5000            # curseur « Aire min x100 » = 50 → 5000 px²
NOM_METHODE = "Fond appris"   # la seule méthode de segmentation de ce fichier (n° 3)
SEUIL_BAS_MIN = 12         # écart de couleur minimal avec le fond pour qu'un pixel CLAIR (bois pâle, bourgeon) compte
CONTRASTE_MIN = 15         # bord accepté si fond et branche diffèrent d'au moins 15 (1 = laisse passer des bords faux : bourgeon, vrille)


# =========================================================================================================
# 1. CALIBRATION (lit et écrit calibration.json ; touche c)
# =========================================================================================================
class Calibration:
    """D_mm = a·D_px + b pour les diamètres ; L_mm = a·L_px pour les longueurs. Sans fichier : pixels."""
    def __init__(self, chemin):
        self.ok, self.a, self.b, self.K, self.dist, self._maps = False, 1.0, 0.0, None, None, None
        if os.path.exists(chemin):
            d = json.load(open(chemin, encoding="utf-8"))
            if "mm_par_px" in d:
                self.ok, self.a, self.b = True, float(d["mm_par_px"]), float(d.get("decalage_diametre_mm", 0.0))
            if d.get("camera_matrix") is not None:
                self.K, self.dist = np.array(d["camera_matrix"], float), np.array(d["dist_coeffs"], float)

    def enregistrer(self, chemin, refs):
        """
        refs = liste de (diamètre mesuré en px, diamètre vrai en mm) des étalons.
        1 étalon  → D_mm = a·D_px           (a = mm par pixel)
        ≥ 2       → D_mm = a·D_px + b       (droite des moindres carrés : b corrige le petit biais du bord)
        Écrit calibration.json (même format que calibration_mm.py) et l'utilise tout de suite.
        """
        Dp = np.array([r[0] for r in refs]); Dm = np.array([r[1] for r in refs])
        if len(refs) >= 2 and np.ptp(Dm) > 0:
            a, b = np.polyfit(Dp, Dm, 1)                              # droite qui passe au plus près des étalons
        else:
            a, b = float(np.mean(Dm / Dp)), 0.0                       # simple règle de trois
        d = json.load(open(chemin, encoding="utf-8")) if os.path.exists(chemin) else {}
        d.update(mm_par_px=float(a), decalage_diametre_mm=float(b), date=time.strftime("%Y-%m-%d %H:%M"),
                 etalons=[{"px": float(p_), "mm": float(m_)} for p_, m_ in refs])
        json.dump(d, open(chemin, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        self.ok, self.a, self.b = True, float(a), float(b)
        return a, b

    def diam(self, px):
        return f"{self.a * px + self.b:.3f} mm" if self.ok else f"{px:.2f} px"

    def long(self, px):
        return f"{self.a * px:.3f} mm" if self.ok else f"{px:.2f} px"

    def vd(self, px):
        return self.a * px + self.b if self.ok else px

    def vl(self, px):
        return self.a * px if self.ok else px

    def axe(self, d):
        """Diamètre de l'oblong en mm : « diam » (bord à bord de la branche) → a·px + b ; « long » → a·px."""
        return self.vd(d["px"]) if d["type"] == "diam" else self.vl(d["px"])

    def corriger(self, img):
        """Corrige la distorsion de l'objectif si une calibration damier existe."""
        if self.K is None:
            return img
        h, w = img.shape[:2]
        if self._maps is None or self._maps[2] != (w, h):
            m1, m2 = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, (w, h), cv2.CV_16SC2)
            self._maps = (m1, m2, (w, h))
        return cv2.remap(img, self._maps[0], self._maps[1], cv2.INTER_LINEAR)


# =========================================================================================================
# 2. SOURCES D'IMAGES : caméra Basler (réglages manuels + exposition automatique maison) ou fichier
# =========================================================================================================
class SourceBasler:
    def __init__(self, index):
        from pypylon import pylon
        self.pylon = pylon
        infos = pylon.TlFactory.GetInstance().EnumerateDevices()
        if index >= len(infos):
            raise SystemExit(f"Caméra n° {index} absente ({len(infos)} trouvée(s)). Câble ? pylon Viewer fermé ?")
        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(infos[index]))
        self.cam.Open()
        di = self.cam.GetDeviceInfo()
        print("Caméra :", di.GetModelName(), "S/N", di.GetSerialNumber())
        for nom, val in [("ExposureAuto", "Off"), ("GainAuto", "Off"), ("Gain", 0.0), ("ExposureTime", EXPO_US),
                         ("AcquisitionFrameRateEnable", True), ("AcquisitionFrameRate", 20.0)]:
            self._regler(nom, val)                                    # tout en manuel : mesures répétables
        self._regler("BalanceWhiteAuto", "Once")
        self.conv = pylon.ImageFormatConverter()
        self.conv.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.expo = EXPO_US

    def _regler(self, nom, val):
        try:
            getattr(self.cam, nom).Value = val
        except Exception:
            pass

    def lire(self):
        r = self.cam.RetrieveResult(100, self.pylon.TimeoutHandling_Return)
        img = None
        if r is not None and r.IsValid():
            if r.GrabSucceeded():
                img = self.conv.Convert(r).Array.copy()
            r.Release()
        return img

    def exposition(self, facteur):
        self.expo = float(np.clip(self.expo * facteur, 50, 200000))   # comme dans segmentation_branche.py
        self._regler("ExposureTime", self.expo)

    def fermer(self):
        try:
            self.cam.StopGrabbing(); self.cam.Close()
        except Exception:
            pass


class Stabilisateur:
    """
    Supprime la « vibration » des mesures quand la branche est IMMOBILE.
    Cause : chaque image de la caméra a un bruit différent (et l'éclairage scintille un peu) → les bords, le seuil
    d'Otsu, les limites du nœud bougent de quelques dixièmes de pixel d'une image à l'autre.
    Solution : tant que rien ne bouge, on fait la MOYENNE des images (jusqu'à N_STAB). Le bruit est divisé par
    √N (÷ 5,5 pour 30 images) et TOUTES les mesures et tous les dessins viennent de cette image moyenne : ils
    restent cohérents entre eux. Après N_STAB images, l'image moyenne est FIGÉE : l'affichage ne bouge plus du tout.
    Dès que la branche bouge, la moyenne repart de la nouvelle image (pas de traînée).
    Détection du mouvement : image réduite 8× en gris, corrigée du scintillement global de la lumière, comparée
    à la moyenne ; mouvement si plus de PART_MVT des pixels changent de plus de SEUIL_MVT niveaux.
    """
    def __init__(self, n_max=N_STAB):
        self.n_max, self.acc, self.petit, self.n = n_max, None, None, 0

    def ajouter(self, img):
        p = cv2.cvtColor(cv2.resize(img, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)       # petite image grise : rapide et peu bruitée
        bouge = True
        if self.acc is not None and self.acc.shape == img.shape:
            gain = float(self.petit.mean()) / max(float(p.mean()), 1e-3)   # scintillement : luminosité globale ramenée
            bouge = float((np.abs(p * gain - self.petit) > SEUIL_MVT).mean()) > PART_MVT
        if bouge:                                                     # ça bouge → on repart de cette image
            self.acc, self.petit, self.n = img.astype(np.float32), p, 1
        elif self.n < self.n_max:                                     # immobile → moyenne cumulée des images
            self.n += 1
            a = 1.0 / self.n
            cv2.accumulateWeighted(img, self.acc, a)                  # acc = (1 − a) × acc + a × img
            self.petit = (1 - a) * self.petit + a * p
        # sinon : N_STAB images déjà moyennées → l'image est FIGÉE (plus aucune vibration) jusqu'au prochain mouvement
        return cv2.convertScaleAbs(self.acc)                          # arrondi + retour en 8 bits (rapide)


class SourceImage:
    def __init__(self, chemin):
        self.img = cv2.imread(chemin, cv2.IMREAD_COLOR)
        if self.img is None:
            raise SystemExit("Image illisible : " + chemin)
        self.expo = None

    def lire(self):
        return self.img.copy()

    def exposition(self, facteur):
        pass

    def fermer(self):
        pass


# =========================================================================================================
# 3. SEGMENTATION (réglages fixes de segmentation_branche.py) — le masque sert de GUIDE, la mesure fine est en partie 6
# =========================================================================================================
def plus_grande_composante(m):
    """Garde la plus grosse tache blanche (sert aussi pour la tige, partie 6)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m, 0
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return np.where(lab == k, 255, 0).astype(np.uint8), int(st[k, cv2.CC_STAT_AREA])


def ecart_fond_appris(img_bgr, fond_img):
    """
    FOND APPRIS : pour chaque pixel, distance entre sa couleur et celle du MÊME pixel sur la photo du fond
    (prise sans branche, touche b). 0 = fond, grand = branche. Les ombres et l'éclairage inégal du fond sont
    ainsi compensés pixel par pixel. Renvoie (e, bruit_fond), ou (None, 0) si le fond n'est pas encore appris.
    """
    if fond_img is None or fond_img.shape != img_bgr.shape:
        return None, 0.0
    f = img_bgr.astype(np.float32)
    e = np.sqrt(((f - fond_img) ** 2).sum(axis=2)) / math.sqrt(3)     # distance couleur, ramenée à l'échelle 0…255
    cadre = np.concatenate([e[:10].ravel(), e[-10:].ravel(), e[:, :10].ravel(), e[:, -10:].ravel()])
    return e, float(np.percentile(cadre, 90))                         # bruit du fond = ce qui reste sur le cadre


def segmenter(e, bruit_fond, P=None):
    """
    Masque de la branche ENTIÈRE à partir de l'écart e (écart de couleur au fond appris), avec 2 seuils (hystérésis) :
      seuil HAUT = Otsu  → les parties sûres (écorce sombre)
      seuil BAS  = juste au-dessus du bruit du fond → les parties pâles, gardées SEULEMENT si elles touchent
                   une partie sûre (une poussière isolée sur le fond n'est pas gardée).
    Renvoie (masque rempli, contour) ou (None, None).
    """
    k = FLOU * 2 + 1
    ef = cv2.GaussianBlur(e, (k, k), 0)                               # gros flou : seulement pour trouver les parties SÛRES
    e3 = cv2.GaussianBlur(e, (3, 3), 0)                               # petit flou : pour placer le CONTOUR au bon endroit
    e8 = np.clip(ef, 0, 255).astype(np.uint8)
    haut, _ = cv2.threshold(e8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # seuil haut : Otsu
    bas = min(haut, max(SEUIL_BAS_MIN, 2.0 * bruit_fond + 4))        # seuil bas, jamais au-dessus du seuil haut
    if P is not None:                                                 # mémorisé pour la fenêtre 3 (histogramme)
        P["seuil_haut"], P["seuil_bas"] = float(haut), float(bas)
        P["histo"] = np.bincount(e8[::2, ::2].ravel(), minlength=256)  # histogramme (1 pixel sur 4 suffit)
    sur = (ef > haut).astype(np.uint8)
    possible = (e3 > bas).astype(np.uint8)                            # (le gros flou ferait « gonfler » le contour)
    n, lab = cv2.connectedComponents(possible, connectivity=8)        # morceaux « possibles »
    garder = np.unique(lab[(sur > 0) & (lab > 0)])                   # ceux qui touchent une partie sûre
    m = np.isin(lab, garder).astype(np.uint8)                         # toute la branche, parties pâles comprises
    # --- seuil ADAPTATIF à 50 % : chaque pixel est comparé à la moitié du niveau de la branche AUTOUR de lui
    #     (écorce sombre : niveau ≈ 200 → seuil ≈ 100 ; bois pâle : niveau ≈ 40 → seuil ≈ 20).
    #     Ainsi le contour tombe AU MILIEU du bord partout (ni gonflé par le seuil bas, ni rongé sur le pâle).
    r = int(max(15, 0.03 * max(e.shape)))                            # rayon de voisinage ≈ 60 px
    num = cv2.boxFilter(e3 * m, -1, (2 * r + 1, 2 * r + 1), normalize=False)
    den = cv2.boxFilter(m.astype(np.float32), -1, (2 * r + 1, 2 * r + 1), normalize=False)
    niveau = num / np.maximum(den, 1.0)                               # niveau moyen de la branche autour du pixel
    m = ((m > 0) & (e3 > np.maximum(bas, 0.5 * niveau))).astype(np.uint8) * 255
    t = MORPHO * 2 + 1
    noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, noyau)                    # supprime les poussières
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, noyau)                   # bouche les petits trous
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [c for c in cs if cv2.contourArea(c) >= AIRE_MIN]
    if not cs:
        return None, None
    c = max(cs, key=cv2.contourArea)                                  # la plus grosse forme = la branche
    plein = np.zeros_like(m)
    cv2.drawContours(plein, [c], -1, 255, cv2.FILLED)                 # branche remplie, sans trous
    return plein, c


# =========================================================================================================
# 5. REDRESSEMENT : la branche devient horizontale (pleine résolution, interpolation bicubique)
# =========================================================================================================
def redresser(g, masque):
    ys, xs = np.nonzero(masque)
    if len(xs) < 100:
        return None
    sel = np.random.default_rng(0).choice(len(xs), min(len(xs), 60000), replace=False)
    pts = np.column_stack([xs[sel], ys[sel]]).astype(float)
    c = pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov((pts - c).T))                  # ACP : direction principale = axe de la branche
    u = vecs[:, np.argmax(vals)]
    if u[0] < 0:
        u = -u                                                       # s croît de la gauche vers la droite
    v = np.array([-u[1], u[0]])
    P_all = np.column_stack([xs, ys]).astype(float) - c
    s, t = P_all @ u, P_all @ v
    marge = max(30, int(0.25 * (t.max() - t.min())) + 10)            # assez de fond autour pour mesurer les niveaux
    s0, s1, t0, t1 = s.min() - marge, s.max() + marge, t.min() - marge, t.max() + marge
    W, H = int(math.ceil(s1 - s0)) + 1, int(math.ceil(t1 - t0)) + 1
    M = np.array([[u[0], u[1], -(u @ c) - s0], [v[0], v[1], -(v @ c) - t0]])
    return {"g": cv2.warpAffine(g.astype(np.float32), M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE),
            "m": cv2.warpAffine(masque, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0),
            "Minv": cv2.invertAffineTransform(M), "angle": math.degrees(math.atan2(u[1], u[0]))}


def vers_image(R, pts):
    p = np.asarray(pts, float).reshape(-1, 2)
    return np.hstack([p, np.ones((len(p), 1))]) @ R["Minv"].T


# =========================================================================================================
# 6. BORDS SUB-PIXEL ET PROFIL DE DIAMÈTRE
# =========================================================================================================
def tige(mask_r):
    """Tige sans vrille ni pétiole : ouverture avec un disque de COUPE_APP_PCT % du diamètre (calcul en réduit)."""
    larg = (mask_r > 0).sum(axis=0)
    D0 = float(np.median(larg[larg > 0]))
    k = max(3.0, D0 * COUPE_APP_PCT / 100)
    f = min(1.0, 31.0 / k)
    p = cv2.resize(mask_r, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else mask_r
    _, p = cv2.threshold(p, 127, 255, cv2.THRESH_BINARY)
    kk = max(3, int(k * f) | 1)
    t = cv2.morphologyEx(p, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    if f < 1:
        t = cv2.resize(t, (mask_r.shape[1], mask_r.shape[0]), interpolation=cv2.INTER_LINEAR)
        _, t = cv2.threshold(t, 127, 255, cv2.THRESH_BINARY)
        t = cv2.bitwise_and(t, mask_r)
    return plus_grande_composante(t)[0], D0


def guide_tige(t):
    """
    GUIDE = bord haut et bord bas APPROXIMATIFS de la tige, colonne par colonne (pixels entiers).
    1) Axe provisoire c0 : parabole robuste par les milieux des colonnes « propres » (une seule bande blanche,
       de largeur proche de la largeur médiane) → il passe DANS la tige même sous un nœud ou un appendice.
    2) Dans chaque colonne, on garde la bande blanche qui CONTIENT l'axe : si un appendice penché passe sous la
       tige avec du fond entre les deux, il n'est pas pris (on s'arrête au vrai bord de la tige).
    """
    col = t > 0
    H = col.shape[0]
    pres = np.nonzero(col.any(axis=0))[0]
    sub = col[:, pres]
    n = sub.sum(axis=0)
    prem, dern = sub.argmax(axis=0), H - 1 - sub[::-1].argmax(axis=0)
    x, mil = pres.astype(float), 0.5 * (prem + dern)
    med = float(np.median(n))
    propre = (n == dern - prem + 1) & (np.abs(n - med) < 0.2 * med)
    c0 = mil.copy()
    if propre.sum() >= 20:
        sel = propre.copy()
        for _ in range(3):                                            # parabole robuste (points aberrants rejetés)
            pp = np.polyfit(x[sel], mil[sel], 2); c0 = np.polyval(pp, x)
            res = mil - c0
            sel = propre & (np.abs(res) < max(3 * 1.4826 * np.median(np.abs(res[sel])), 1.0))
            if sel.sum() < 10:
                break
    r0 = np.clip(np.round(c0).astype(int), 0, H - 1)
    lignes = np.arange(H)[:, None]
    yt = np.where(~sub & (lignes <= r0), lignes, -1).max(axis=0) + 1  # dernier pixel noir AU-DESSUS de l'axe, + 1
    yb = np.where(~sub & (lignes >= r0), lignes, H).min(axis=0) - 1   # premier pixel noir EN DESSOUS de l'axe, − 1
    dans = sub[r0, np.arange(len(pres))]                              # l'axe est-il bien dans du blanc ?
    yt, yb = np.where(dans, yt, prem), np.where(dans, yb, dern)
    return pres, yt.astype(float), yb.astype(float), c0, med


def mediane_glissante(v, demi):
    """Médiane sur une fenêtre de ± demi points (bords prolongés) : la « ligne de base » d'un profil."""
    p = np.pad(v, demi, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(p, 2 * demi + 1), axis=1)


def couper_appendices(pres, yt, yb, c0, D0):
    """
    Coupe « intelligente » des APPENDICES ÉPAIS (bourgeon, moignon de rameau…) que l'ouverture n'enlève pas.
    Idée : un nœud est un renflement DOUX de la tige ; un appendice est une excroissance d'UN côté, haute
    (> APP_HAUT × diamètre) avec des parois raides. Pour chaque côté (haut, bas) :
      1) rayon r(s) = distance axe → bord ; ligne de base = médiane glissante sur 4 diamètres ;
      2) graines = colonnes où r dépasse la base de plus de APP_HAUT × D0 ;
      3) depuis chaque graine, on descend la paroi vers la gauche et vers la droite jusqu'à la RACINE :
         l'endroit où le bord redevient presque parallèle à la tige (pente < APP_PENTE) ;
      4) entre les 2 racines, le bord est remplacé par une DROITE (un « pont ») : c'est le bord de la tige
         tel qu'il serait sans l'appendice. Le renflement du nœud, de part et d'autre, est conservé.
    Renvoie les guides corrigés, les masques des colonnes « pont » et la liste des appendices trouvés.
    """
    n = len(pres)
    h = max(3, int(0.04 * D0))                                        # pas pour mesurer la pente
    marge = max(2, int(0.03 * D0))                                    # on recule un peu au-delà de la racine
    out, ponts, liste = {}, {}, []
    for nom, r in (("haut", c0 - yt), ("bas", yb - c0)):
        r = r.astype(float).copy()
        rs = lissage_gauss(r, max(1.0, 0.02 * D0))                    # léger lissage (écorce irrégulière)
        exc = rs - mediane_glissante(rs, int(2 * D0))                 # dépassement par rapport à la ligne de base
        graine = exc > APP_HAUT * D0
        pont = np.zeros(n, bool)
        i = 0
        while i < n:
            if not graine[i]:
                i += 1; continue
            i2 = i
            while i2 + 1 < n and graine[i2 + 1]:
                i2 += 1
            racines = []
            for depart, pas in ((i, -1), (i2, +1)):                   # descente de la paroi des 2 côtés
                j = depart
                while 0 < j < n - 1:
                    k = int(np.clip(j + pas * h, 0, n - 1))
                    pente = (rs[j] - rs[k]) / max(abs(k - j), 1)      # > 0 tant que le bord descend vers la tige
                    if exc[j] <= 0.04 * D0 or (pente < APP_PENTE and exc[j] < 0.75 * APP_HAUT * D0):
                        break
                    j += pas
                racines.append(int(np.clip(j + pas * marge, 0, n - 1)))
            j1, j2 = racines
            if j2 - j1 >= 2:
                r[j1:j2 + 1] = np.interp(np.arange(j1, j2 + 1), [j1, j2], [r[j1], r[j2]])   # le PONT (droite)
                pont[j1:j2 + 1] = True
                liste.append({"cote": nom, "s1": float(pres[j1]), "s2": float(pres[j2]),
                              "hauteur": float(exc[i:i2 + 1].max())})
            i = i2 + 1
        out[nom], ponts[nom] = r, pont
    return c0 - out["haut"], c0 + out["bas"], ponts["haut"], ponts["bas"], liste


def bords_subpixel(R, pres, yt, yb, contraste_min):
    """
    Pour chaque colonne : niveau local = moitié entre le gris du FOND et celui de la BRANCHE (médianes),
    bord = endroit où le gris traverse ce niveau, interpolé entre 2 pixels (≈ 0,05 px). Toutes les colonnes d'un coup.
    """
    g, H = R["g"], R["g"].shape[0]
    yt, yb = np.round(yt).astype(int), np.round(yb).astype(int)      # guides (voir guide_tige / couper_appendices)
    D0 = float(np.median(yb - yt + 1))
    w = int(max(3, round(0.06 * D0)))                                  # demi-fenêtre de recherche autour du guide
    e1, e2 = int(max(3, 0.05 * D0)), int(max(8, 0.15 * D0))            # bande de mesure des niveaux (hors du flou)
    lire = lambda rows: g[np.clip(rows, 0, H - 1), pres[:, None]]
    bande, fen = np.arange(e1, e2)[None, :], np.arange(-w, w + 1)[None, :]
    out = {}
    for nom, y_ref, sens in (("haut", yt, -1), ("bas", yb, +1)):
        I_f = np.median(lire(y_ref[:, None] + sens * bande), axis=1)   # fond (extérieur)
        I_b = np.median(lire(y_ref[:, None] - sens * bande), axis=1)   # branche (intérieur)
        niv = 0.5 * (I_f + I_b)
        y0 = np.round(y_ref + 0.5 * sens).astype(int)
        d = lire(y0[:, None] + fen) - niv[:, None]
        chg = (np.sign(d[:, :-1]) * np.sign(d[:, 1:]) <= 0) & (d[:, :-1] != d[:, 1:])
        cout = np.where(chg, np.abs(np.arange(2 * w)[None, :] + 0.5 - w), np.inf)   # traversée la plus proche du guide
        kb = np.argmin(cout, axis=1); ii = np.arange(len(kb))
        g0, g1 = d[ii, kb], d[ii, kb + 1]
        pos = y0 - w + kb + np.where(g0 != g1, g0 / np.where(g0 != g1, g0 - g1, 1), 0.0)
        ok = np.isfinite(cout[ii, kb]) & (np.abs(I_f - I_b) >= contraste_min) & (y_ref - e2 >= 0) & (y_ref + e2 < H)
        out[nom] = np.where(ok, pos, np.nan)
    return pres.astype(float), out["haut"], out["bas"]


def lissage_gauss(v, sigma):
    """Moyenne pondérée gaussienne qui ignore les trous (NaN)."""
    ok = np.isfinite(v)
    out = v.copy()
    if sigma >= 0.5 and ok.sum() >= 3:
        r = int(3 * sigma)
        ker = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        num = np.convolve(np.where(ok, v, 0.0), ker, mode="same")
        den = np.convolve(ok.astype(float), ker, mode="same")
        out = np.where(den > 1e-6, num / np.maximum(den, 1e-9), np.nan)
    i, okk = np.arange(len(v)), np.isfinite(out)
    if okk.sum() >= 2 and not okk.all():
        out[~okk] = np.interp(i[~okk], i[okk], out[okk])
    return out


def hampel(v, fen, k=3.0, mini=1.0):
    """Rejette (NaN) les points trop loin de la médiane de leurs voisins : bord pris sur une vrille, poussière…"""
    h = fen // 2
    if len(v) < 2 * h + 1:
        return v.copy()
    W = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.full(h, np.nan), v, np.full(h, np.nan)]), 2 * h + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(W, axis=1)
        mad = 1.4826 * np.nanmedian(np.abs(W - med[:, None]), axis=1)
    return np.where(np.abs(v - med) > np.maximum(k * np.nan_to_num(mad), mini), np.nan, v)


def profil(R, P):
    """Bords sub-pixel → rayons haut/bas autour d'une ligne centrale robuste → diamètre (symétrie si un côté est caché)."""
    t, _ = tige(R["m"])                                               # vrilles et pétioles fins enlevés
    pres, yt, yb, c0, D0 = guide_tige(t)                              # bande blanche qui contient l'axe
    if len(pres) < 30:
        return None
    yt, yb, pont_h, pont_b, appendices = couper_appendices(pres, yt, yb, c0, D0)   # appendices épais coupés à la racine
    s, haut, bas = bords_subpixel(R, pres, yt, yb, P["contraste_min"])
    idx = np.arange(len(s))
    for v, pont in ((haut, pont_h), (bas, pont_b)):                   # sur un pont il n'y a pas de vrai bord à mesurer :
        if pont.any():                                                # droite entre les bords sub-pixel des 2 racines
            ok = np.isfinite(v) & ~pont
            v[pont] = np.interp(idx[pont], idx[ok], v[ok]) if ok.sum() >= 2 else np.nan
    P["appendices"] = len(appendices)
    ok2 = np.isfinite(haut) & np.isfinite(bas)
    if ok2.sum() < max(30, 0.25 * len(s)):                            # image bougée/floue : trop peu de bords nets
        return None
    centre = 0.5 * (haut + bas)
    sel = ok2 & ~pont_h & ~pont_b                                     # l'axe est ajusté sans les colonnes « pont »
    cl = np.where(ok2, centre, np.nanmean(centre))
    for _ in range(3):                                                # ligne centrale = parabole robuste
        if sel.sum() < 10:
            break
        pp = np.polyfit(s[sel], centre[sel], 2); cl = np.polyval(pp, s)
        res = centre - cl
        sel = ok2 & (np.abs(res) < max(3 * 1.4826 * np.median(np.abs(res[sel])), 0.5))
    fen = max(5, int(0.15 * D0))
    rh, rb = hampel(cl - haut, fen), hampel(bas - cl, fen)           # rayons haut et bas, nettoyés
    m_ = max(2, int(0.04 * D0))                                       # les points collés à un bord rejeté (racine d'une
    for r_ in (rh, rb):                                               # vrille, d'un bourgeon) sont douteux : rejetés aussi
        mauvais = np.convolve((~np.isfinite(r_)).astype(float), np.ones(2 * m_ + 1), mode="same") > 0
        r_[mauvais] = np.nan
    # Sous un appendice, le pont (droite entre les racines) peut « raboter » le renflement du nœud : si l'autre
    # côté de la tige (non caché) est plus bombé, on prend son rayon (symétrie de la tige autour de son axe).
    rb = np.where(pont_b & ~pont_h & np.isfinite(rh), np.fmax(rb, rh), rb)
    rh = np.where(pont_h & ~pont_b & np.isfinite(rb), np.fmax(rh, rb), rh)
    okh, okb = np.isfinite(rh), np.isfinite(rb)
    D = np.where(okh & okb, rh + rb, np.where(okh, 2 * rh, np.where(okb, 2 * rb, np.nan)))  # 1 côté caché → symétrie
    # --- lissage choisi AUTOMATIQUEMENT selon le bruit des bords
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bruit = 1.4826 * np.nanmedian(np.abs(np.diff(D))) / math.sqrt(2)     # bruit colonne à colonne (px)
    bruit = float(bruit) if np.isfinite(bruit) else 0.5
    sig = float(np.clip(0.03 * D0 * max(1.0, bruit / 0.3), 0.02 * D0, 0.08 * D0))
    P["lissage_px"], P["bruit_bord_px"] = sig, bruit
    rh_f, rb_f = np.where(okh, rh, rb), np.where(okb, rb, rh)
    return {"s": s, "haut": haut, "bas": bas, "D": D, "D_lisse": lissage_gauss(D, sig),
            "haut_lisse": cl - lissage_gauss(rh_f, sig), "bas_lisse": cl + lissage_gauss(rb_f, sig),
            "centre": np.where(ok2, centre, cl), "D0": D0, "valides": 100 * np.isfinite(D).mean(),
            "pont_h": pont_h, "pont_b": pont_b, "appendices": appendices}


# =========================================================================================================
# 7. NŒUD : pic sub-pixel, diamètre de base sans le nœud, limites interpolées, test « vrai nœud »
# =========================================================================================================
def _traverse(sv, L, i, niveau, pas):
    while 0 <= i + pas < len(L) and L[i + pas] > niveau:
        i += pas
    if not 0 <= i + pas < len(L):
        return float(sv[i])
    a, b = L[i], L[i + pas]
    return float(sv[i] + ((a - niveau) / (a - b) if a != b else 0.0) * (sv[i + pas] - sv[i]))


def tendance(sv, L, garde):
    """Droite robuste par le profil (points « garde ») en ignorant ce qui dépasse AU-DESSUS (les renflements)."""
    sel = garde.copy()
    p = np.array([0.0, float(np.median(L[garde]))])
    for _ in range(5):
        if sel.sum() < 10:
            break
        p = np.polyfit(sv[sel], L[sel], 1)
        res = L - np.polyval(p, sv)
        sig = 1.4826 * np.median(np.abs(res[sel] - np.median(res[sel]))) + 1e-6
        nouveau = garde & (res < 1.0 * sig)                           # on garde ce qui n'est pas nettement au-dessus
        if nouveau.sum() < 0.3 * garde.sum():
            break
        sel = nouveau
    return np.polyval(p, sv)


def noeud(B, P):
    """
    NŒUD = le plus grand RENFLEMENT de la tige par rapport à sa TENDANCE (une branche est souvent conique :
    chercher simplement le diamètre maximal trouverait le gros bout, pas le nœud).
      • tendance = droite robuste par le profil de diamètre (renflements ignorés) ;
      • excès = profil − tendance ; pic = maximum de l'excès (affiné par parabole) ;
      • si un APPENDICE a été trouvé, il pousse sur le nœud : on cherche le pic près de sa racine (± 1,5 D) ;
      • limites = là où l'excès retombe à BORD_NOEUD_PCT % de sa valeur au pic ;
      • si la tige n'a PAS de renflement mesurable mais qu'il y a un appendice : le nœud est placé sur la
        racine de l'appendice (limites = racine ± 0,1 D).
    """
    sv, L, D0 = B["s"], B["D_lisse"], B["D0"]
    zone = (sv >= sv[0] + MARGE_BOUTS * D0) & (sv <= sv[-1] - MARGE_BOUTS * D0)
    if zone.sum() < 10:
        return None
    App = B.get("appendices", [])
    cherche = zone.copy()
    if App:                                                           # a priori : le nœud est près d'un appendice
        pres_app = np.zeros(len(sv), bool)
        for a in App:
            pres_app |= (sv >= a["s1"] - 1.5 * D0) & (sv <= a["s2"] + 1.5 * D0)
        if (zone & pres_app).sum() >= 10:
            cherche = zone & pres_app
    T = tendance(sv, L, zone)
    s_g = s_d = s_pic = float(sv[np.nonzero(cherche)[0][0]]); D_pic = ex_pic = 0.0; i = 0
    for _ in range(3):                                                # tendance recalculée sans la zone du nœud
        ex = L - T
        ic = np.nonzero(cherche)[0]
        i = ic[int(np.argmax(ex[ic]))]
        if 0 < i < len(L) - 1:                                        # sommet de la parabole par 3 points
            y0, y1, y2 = ex[i - 1], ex[i], ex[i + 1]
            den = y0 - 2 * y1 + y2
            dx = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if den != 0 else 0.0
            s_pic, ex_pic = sv[i] + dx * (sv[i + 1] - sv[i]), y1 - 0.25 * (y0 - y2) * dx
        else:
            s_pic, ex_pic = sv[i], ex[i]
        niv_ex = BORD_NOEUD_PCT / 100 * ex_pic
        s_g, s_d = _traverse(sv, ex, i, niv_ex, -1), _traverse(sv, ex, i, niv_ex, +1)
        hors = zone & ~((sv > s_g - 0.3 * D0) & (sv < s_d + 0.3 * D0))
        if hors.sum() > 10:
            T = tendance(sv, L, hors)
    ex = L - T
    base = float(np.interp(s_pic, sv, T))                             # diamètre de la tige (tendance) sous le pic
    ex_pic = float(np.interp(s_pic, sv, ex)) if not (0 < i < len(L) - 1) else float(ex_pic)
    D_pic = base + ex_pic
    hors = zone & ~((sv > s_g - 0.3 * D0) & (sv < s_d + 0.3 * D0))
    var_tige = float(1.4826 * np.median(np.abs(ex[hors] - np.median(ex[hors])))) if hors.sum() > 10 else 0.0
    seuil = max(0.03 * base, 6 * var_tige)                            # le renflement doit dépasser le bruit de la tige
    P["seuil_noeud_px"] = seuil
    ok, par = ex_pic >= seuil, "renflement"
    if not ok and App:                                                # pas de renflement net → nœud = racine de l'appendice
        a = max(App, key=lambda q: q["hauteur"])
        s_g, s_d = a["s1"] - 0.1 * D0, a["s2"] + 0.1 * D0
        s_pic = 0.5 * (s_g + s_d)
        base = float(np.interp(s_pic, sv, T)); D_pic = float(np.interp(s_pic, sv, L)); ok, par = True, "appendice"
    return {"ok": bool(ok), "par": par, "s_pic": float(s_pic), "D_pic": float(D_pic), "base": base,
            "renfl": 100 * (D_pic / base - 1), "niveau": base + BORD_NOEUD_PCT / 100 * (D_pic - base),
            "s_g": float(s_g), "s_d": float(s_d), "longueur": float(s_d - s_g),
            "s_debut": float(sv[0]), "a": float(sv[np.nonzero(zone)[0][0]]), "b": float(sv[np.nonzero(zone)[0][-1]])}


# =========================================================================================================
# 7b. OBLONG autour du nœud + GRAND et PETIT diamètre (ÉTAPE 2)
#     Tout est calculé dans l'image REDRESSÉE (s = le long de la branche, t = en travers, t augmente vers le bas) :
#     les côtés gauche/droit de l'oblong sont donc EXACTEMENT perpendiculaires à la branche.
# =========================================================================================================
def axe_branche(B, N):
    """
    Une vraie branche est souvent COUDÉE au nœud : l'axe calculé sur toute la branche (ACP, partie 5) est alors
    penché par rapport à la tige de chaque côté du nœud. On calcule donc l'AXE LOCAL :
      • ligne centrale = milieu des bords haut et bas, SANS la zone du nœud (une bosse d'un seul côté décalerait
        le milieu), lissée sur ½ diamètre ; dans le nœud elle est remplacée par une droite entre les 2 côtés ;
      • ligne et pente obtenues par une droite des moindres carrés glissante (± 1 diamètre) ;
      • inclinaison locale θ(s) = pente de cette ligne → cos θ sert à corriger les diamètres : une coupe verticale
        d'une tige penchée de θ mesure D / cos θ au lieu de D.
    """
    sv, D0 = B["s"], B["D0"]
    cen = 0.5 * (B["haut_lisse"] + B["bas_lisse"])
    cen = np.where((sv > N["s_g"] - 0.1 * D0) & (sv < N["s_d"] + 0.1 * D0), np.nan, cen)   # on enlève le nœud
    # droite des moindres carrés GLISSANTE sur ± 1 diamètre (sommes calculées par convolution : rapide).
    # Contrairement à un lissage, une régression n'est pas biaisée aux bouts de la branche.
    ok = np.isfinite(cen).astype(float); c0 = np.nan_to_num(cen)
    ker = np.ones(2 * max(3, int(D0)) + 1)
    conv = lambda x: np.convolve(x, ker, mode="same")
    S0, S1, S2 = conv(ok), conv(sv * ok), conv(sv * sv * ok)
    C1, SC = conv(c0 * ok), conv(sv * c0 * ok)
    det = S0 * S2 - S1 ** 2
    valide = (S0 >= 5) & (np.abs(det) > 1e-9)
    pente = np.where(valide, (S0 * SC - S1 * C1) / np.where(valide, det, 1), np.nan)
    cs = np.where(valide, (C1 - pente * S1) / np.where(valide, S0, 1) + pente * sv, np.nan)
    i = np.arange(len(sv)); k = np.isfinite(cs)
    if k.sum() >= 2:                                                  # trous (au milieu du nœud) bouchés en ligne droite
        cs = np.interp(i, i[k], cs[k]); pente = np.interp(i, i[k], pente[k])
    else:
        cs, pente = 0.5 * (B["haut_lisse"] + B["bas_lisse"]), np.zeros_like(sv)
    B["cen_axe"], B["pente"], B["cos"] = cs, pente, 1.0 / np.sqrt(1.0 + pente ** 2)


def repere_local(B, N):
    """
    Repère du NŒUD : axe u = droite entre C_g et C_d, les centres de la branche sur les 2 limites
    (comme l'étape 3 « point milieu »), v = perpendiculaire (vers le bas). Origine o = C_g.
    Un point (s, t) de l'image redressée a pour coordonnées locales a = (P − o)·u (le long), b = (P − o)·v (en travers).
    """
    sv, cs = B["s"], B["cen_axe"]
    Cg = np.array([N["s_g"], np.interp(N["s_g"], sv, cs)])
    Cd = np.array([N["s_d"], np.interp(N["s_d"], sv, cs)])
    u = (Cd - Cg) / max(np.linalg.norm(Cd - Cg), 1e-9)
    v = np.array([-u[1], u[0]])                                       # u penche peu → v pointe vers le bas (t croissant)
    return Cg, u, v


def oblong(B, N):
    """
    OBLONG dans le repère LOCAL du nœud (et plus dans le repère de toute la branche) :
      côtés gauche / droit = droites PERPENDICULAIRES à l'axe local passant par C_g et C_d (les 2 limites) ;
      côtés haut / bas = tangentes au point le plus haut / le plus bas des BORDS du nœud entre ces 2 droites.
    Ainsi, même si la branche est coudée, l'oblong colle au nœud (il ne déborde plus sous la branche).
    """
    sv = B["s"]
    o, u, v = repere_local(B, N)
    a_g, a_d = 0.0, float((np.array([N["s_d"], np.interp(N["s_d"], sv, B["cen_axe"])]) - o) @ u)
    pts_h = np.column_stack([sv, B["haut_lisse"]]) - o                # bord haut (lissé) dans le repère local
    pts_b = np.column_stack([sv, B["bas_lisse"]]) - o                 # bord bas
    ah, bh, ab, bb = pts_h @ u, pts_h @ v, pts_b @ u, pts_b @ v
    mh, mb = (ah >= a_g) & (ah <= a_d), (ab >= a_g) & (ab <= a_d)     # points ENTRE les 2 limites
    if mh.sum() < 3 or mb.sum() < 3:
        return None
    b_h, b_b = float(np.nanmin(bh[mh])), float(np.nanmax(bb[mb]))     # point le plus HAUT / le plus BAS du nœud
    L, H = a_d - a_g, b_b - b_h
    X = lambda a, b: tuple(o + a * u + b * v)                         # repère local → image redressée (s, t)
    a_c, b_c = 0.5 * (a_g + a_d), 0.5 * (b_h + b_b)
    return {"s_g": N["s_g"], "s_d": N["s_d"], "L": L, "H": H, "rayon": 0.5 * min(L, H),
            "o": o, "u": u, "v": v, "a_g": a_g, "a_d": a_d, "b_h": b_h, "b_b": b_b, "X": X,
            "centre": X(a_c, b_c), "sommet": X(a_c, b_h),            # sommet = haut de l'axe en travers
            "D_pic": N["D_pic"]}


def diametres_oblong(O):
    """
    Les 2 diamètres de l'oblong (schéma d1 / d2), dans l'image redressée (s le long de la branche, t en travers) :
      (calculés dans le REPÈRE LOCAL du nœud : a le long de l'axe local, b en travers)
      axe « long de la branche » : de (a_g, b_c) à (a_d, b_c)     → longueur L (entre les 2 limites)
      axe « travers »             : de (a_c, b_h) à (a_c, b_b)     → longueur H (du haut au bas du nœud)
    Ils se croisent au CENTRE (s_c, t_c) de l'oblong. Leurs 4 bouts sont SUR le contour de l'oblong
    (sur les côtés droits ou au sommet des arrondis). d1 = le plus long, d2 = le plus court.
    "type" sert à la conversion en mm : "long" (entre 2 limites) ou "diam" (entre 2 bords de la branche).
    """
    X, a_c, b_c = O["X"], 0.5 * (O["a_g"] + O["a_d"]), 0.5 * (O["b_h"] + O["b_b"])
    axe_s = {"px": O["L"], "p1": X(O["a_g"], b_c), "p2": X(O["a_d"], b_c), "type": "long", "sens": "le long de la branche"}
    axe_t = {"px": O["H"], "p1": X(a_c, O["b_h"]), "p2": X(a_c, O["b_b"]), "type": "diam", "sens": "en travers de la branche"}
    d1, d2 = (axe_s, axe_t) if O["L"] >= O["H"] else (axe_t, axe_s)
    return d1, d2


def surface_oblong(O, calib):
    """
    Surface d'un stade (oblong) de longueur L, hauteur H et rayon r = ½ min(L, H) :
        S = L × H − (4 − π) × r²     (chaque coin arrondi enlève r² − π r² / 4 au rectangle)
    En mm² si calibré (L en mm par a·px, H en mm par a·px + b), sinon en px².
    """
    L, H = calib.vl(O["L"]), calib.vd(O["H"])
    r = 0.5 * min(L, H)
    return L * H - (4 - math.pi) * r * r


def diametres_bouts(B):
    """
    d3 / d4 = diamètre EXTÉRIEUR de la branche à ses 2 bouts : distance entre le bord haut et le bord bas
    SUB-PIXEL MESURÉS (le contour extérieur réel, sans reconstruction par symétrie), dans une bande de
    BOUT_LARG × D0 placée à BOUT_POS × D0 du bout. Valeur = médiane des colonnes de la bande.
    d3 = bout d'ENTRÉE, d4 = l'autre bout.
    """
    sv, D0 = B["s"], B["D0"]
    ext = B["bas"] - B["haut"]                                         # bord à bord, colonne par colonne (NaN si un bord manque)
    bouts = {"gauche": sv[0] + BOUT_POS * D0, "droite": sv[-1] - BOUT_POS * D0}
    out = {}
    for nom, cote in (("d3", ENTREE), ("d4", "droite" if ENTREE == "gauche" else "gauche")):
        s_m = bouts[cote]
        m = (np.abs(sv - s_m) <= 0.5 * BOUT_LARG * D0) & np.isfinite(ext)
        if m.sum() < 3:
            out[nom] = None
            continue
        D = float(np.median(ext[m] * B["cos"][m]))                    # × cos θ : diamètre PERPENDICULAIRE à la tige
        th = float(np.median(B["haut"][m]))
        pen = float(np.interp(s_m, sv, B["pente"]))  # pente de la tige à cet endroit
        n = np.array([-pen, 1.0]) / math.hypot(pen, 1.0)              # direction perpendiculaire à la tige
        ctr = np.array([s_m, 0.5 * (th + float(np.median(B["bas"][m])))])   # milieu entre les 2 bords
        out[nom] = {"px": D, "s": float(s_m), "p1": tuple(ctr - 0.5 * D * n), "p2": tuple(ctr + 0.5 * D * n), "cote": cote}
    return out["d3"], out["d4"]


def perigee(B, O, angle_deg, entree=ENTREE):
    """
    Ligne rouge partant du SOMMET de l'oblong P0 (haut de l'axe en travers), faisant l'angle « angle_deg » avec
    l'axe EN TRAVERS de l'oblong (repère local du nœud), et descendant vers l'entrée :
        direction = cos(angle) × v + sin(angle) × (± u)      (v = en travers vers le bas, u = le long du nœud)
    Point de contact P1 = 1er point où la ligne atteint le BORD BAS lissé de la branche (interpolé, sub-pixel).
    Périgée = perpendiculaire à la TIGE en P1 (inclinaison locale de la tige) ; d5 = diamètre de la tige en P1.
    """
    sv, D0 = B["s"], B["D0"]
    sens = -1.0 if entree == "gauche" else 1.0
    a = math.radians(angle_deg)
    P0 = np.array(O["sommet"])
    dr = math.cos(a) * O["v"] + math.sin(a) * sens * O["u"]           # direction de la ligne rouge
    lam = np.arange(0.0, 6.0 * (sv[-1] - sv[0]), 0.5)                 # on avance de 0,5 px à la fois
    pts = P0 + lam[:, None] * dr
    dedans = (pts[:, 0] >= sv[0]) & (pts[:, 0] <= sv[-1])
    pts, lam = pts[dedans], lam[dedans]
    if len(pts) < 2:
        return None
    g = np.interp(pts[:, 0], sv, B["bas_lisse"]) - pts[:, 1]          # > 0 : la ligne est encore au-dessus du bord bas
    k = np.nonzero(g <= 0)[0]
    if len(k) == 0 or k[0] == 0:
        return None
    k = k[0]
    f = g[k - 1] / (g[k - 1] - g[k]) if g[k - 1] != g[k] else 0.0
    P1 = pts[k - 1] + f * (pts[k] - pts[k - 1])                       # point de contact, sub-pixel
    s_p = float(P1[0])
    pen = float(np.interp(s_p, sv, B["pente"]))   # inclinaison de la tige à cet endroit
    n = np.array([-pen, 1.0]) / math.hypot(pen, 1.0)                 # perpendiculaire à la tige (vers le bas)
    d5 = float(np.interp(s_p, sv, B["D_lisse"]) * np.interp(s_p, sv, B["cos"]))   # diamètre ⟂ à la tige
    haut = P1 - d5 * n                                                # l'autre bout de la périgée (bord haut)
    reste = (s_p - sv[0]) if entree == "gauche" else (sv[-1] - s_p)
    a_p = float((P1 - O["o"]) @ O["u"])                               # position de la périgée le long du nœud
    return {"angle": angle_deg, "P0": tuple(P0), "P1": tuple(P1), "s_p": s_p, "haut": tuple(haut), "bas": tuple(P1),
            "n": n, "d5": d5, "dist_centre": abs(0.5 * (O["a_g"] + O["a_d"]) - a_p), "dist_entree": float(reste),
            "dans_noeud": bool(O["a_g"] < a_p < O["a_d"]), "fiable": bool(reste > 0.3 * D0)}


def contour_oblong(O, n=24):
    """Points (s, t) du contour de l'oblong : 4 quarts de cercle reliés → les 4 côtés droits viennent tout seuls."""
    s0, s1, t0, t1, r = O["a_g"], O["a_d"], O["b_h"], O["b_b"], O["rayon"]    # dans le repère LOCAL du nœud
    coins = [((s1 - r, t0 + r), -90, 0),                              # coin haut-droit
             ((s1 - r, t1 - r), 0, 90),                               # coin bas-droit
             ((s0 + r, t1 - r), 90, 180),                             # coin bas-gauche
             ((s0 + r, t0 + r), 180, 270)]                            # coin haut-gauche
    pts = []
    for (cs, ct), a0, a1 in coins:
        for a in np.linspace(math.radians(a0), math.radians(a1), n):
            pts.append(O["X"](cs + r * math.cos(a), ct + r * math.sin(a)))  # point du quart de cercle → (s, t)
    return np.array(pts)


def analyser(img, calib, fond_img=None):
    """Toute la chaîne pour une image. Renvoie un dictionnaire de résultats."""
    t0 = time.perf_counter()
    img = calib.corriger(img)                                         # distorsion (si calibration damier)
    P = {"contraste_min": CONTRASTE_MIN}                              # + lissage et seuil nœud ajoutés par profil()/noeud()
    res = {"img": img, "P": P, "ok": False, "msg": ""}
    e, bruit_fond = ecart_fond_appris(img, fond_img)                  # 0 = fond, grand = branche
    if e is None:
        res.update(masque=None, contour=None, msg="Fond pas encore appris : enleve la branche puis appuie sur b")
        return res
    P["bruit_fond"] = bruit_fond
    masque, contour = segmenter(e, bruit_fond, P)                     # masque = branche entière
    res.update(masque=masque, contour=contour)
    if masque is None:
        res["msg"] = "Aucune branche trouvee"; return res
    g = e                                                             # bords sub-pixel mesurés sur l'écart NON flouté
    R = redresser(g, masque)
    B = profil(R, P) if R is not None else None
    if B is None:
        res["msg"] = "Branche floue / qui bouge / contraste trop faible"; return res
    N = noeud(B, P)
    res.update(R=R, B=B, N=N, ms=(time.perf_counter() - t0) * 1000)
    if N is None:
        res["msg"] = "Branche trop courte"
    elif not N["ok"]:
        res["msg"] = f"Pas de noeud net (renflement {N['renfl']:.1f} % = dans le bruit)"
    res["ok"] = N is not None and N["ok"]
    if res["ok"]:
        axe_branche(B, N)                                             # axe local + inclinaison (branche coudée)
    res["O"] = oblong(B, N) if res["ok"] else None                    # oblong seulement si un vrai nœud est trouvé
    if res["O"] is not None:
        res["O"]["d1"], res["O"]["d2"] = diametres_oblong(res["O"])      # les 2 diamètres de l'oblong (d1 ≥ d2)
        res["O"]["d3"], res["O"]["d4"] = diametres_bouts(B)                # diamètres extérieurs aux 2 bouts
        res["O"]["perigee"] = perigee(B, res["O"], REGLAGE["angle"])       # ligne d'angle → périgée, d5
    return res


# =========================================================================================================
# 8. AFFICHAGE
# =========================================================================================================
def ligne(img, pts, couleur, ep, ferme=False):
    p = np.round(np.asarray(pts) * 16).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], ferme, couleur, ep, cv2.LINE_AA, shift=4)      # précision 1/16 px


def fleche(img, p1, p2, couleur, ep):
    """Double flèche de p1 à p2 (pointes aux 2 bouts), précision 1/16 px."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    L = float(np.hypot(*(p2 - p1)))
    if L < 1:
        return
    ligne(img, [p1, p2], couleur, ep)
    u = (p2 - p1) / L; n = np.array([-u[1], u[0]]); t = min(0.25 * L, 9 * ep)
    for bout, sg in ((p1, 1), (p2, -1)):
        ligne(img, [bout + sg * t * u + 0.5 * t * n, bout, bout + sg * t * u - 0.5 * t * n], couleur, ep)


def etiquette(img, texte, pos, couleur, taille):
    """Texte avec un contour blanc (lisible sur la branche)."""
    (lw, lh), _ = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, taille, max(1, int(2 * taille)))
    x = int(np.clip(pos[0], 5, img.shape[1] - lw - 5))                # l'étiquette reste DANS l'image
    y = int(np.clip(pos[1], lh + 5, img.shape[0] - 5))
    cv2.putText(img, texte, (x, y), cv2.FONT_HERSHEY_SIMPLEX, taille, (255, 255, 255), max(2, int(5 * taille)), cv2.LINE_AA)
    cv2.putText(img, texte, (x, y), cv2.FONT_HERSHEY_SIMPLEX, taille, couleur, max(1, int(2 * taille)), cv2.LINE_AA)


def bandeau(img, lignes):
    for i, l in enumerate(lignes):
        y = 36 + 34 * i
        cv2.putText(img, l, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, l, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


TUILE_W, TUILE_H = 800, 500         # taille de chacune des 4 fenêtres à l'écran (px)


def titre(tuile, texte):
    """Bandeau gris foncé avec un titre en haut d'une tuile."""
    cv2.rectangle(tuile, (0, 0), (tuile.shape[1], 30), (50, 50, 50), -1)
    cv2.putText(tuile, texte, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return tuile


def vue_camera(res):
    """FENÊTRE 1 : image caméra + contour, oblong, 2 limites, diamètre central et son point central. Sans texte."""
    out = res["img"].copy()
    ep = max(2, int(round(out.shape[1] / TUILE_W * 2)))              # traits assez épais pour rester visibles une fois réduits
    if res.get("contour") is not None:
        cv2.drawContours(out, [res["contour"]], -1, (0, 255, 0), max(1, ep // 2), cv2.LINE_AA)   # contour (vert)
    R, B, O = res.get("R"), res.get("B"), res.get("O")
    if O is None:
        return out
    for cle, pont in (("haut_lisse", B["pont_h"]), ("bas_lisse", B["pont_b"])):          # PONTS : bord de la tige sous un
        k = np.nonzero(pont)[0]                                                          # appendice ignoré (orange)
        for grp in np.split(k, np.nonzero(np.diff(k) > 1)[0] + 1):
            if len(grp) >= 2:
                ligne(out, vers_image(R, np.column_stack([B["s"][grp], B[cle][grp]])), (0, 140, 255), ep)
    ligne(out, vers_image(R, contour_oblong(O)), (0, 215, 255), ep, ferme=True)          # OBLONG (jaune)
    for a_l in (O["a_g"], O["a_d"]):                                                     # 2 LIMITES (rouge), ⟂ à l'axe local
        ligne(out, vers_image(R, [O["X"](a_l, O["b_h"] - 0.2 * B["D0"]), O["X"](a_l, O["b_b"] + 0.2 * B["D0"])]),
              (0, 0, 255), ep)
    for d, coul in ((O["d1"], (255, 0, 255)), (O["d2"], (255, 120, 0))):                 # d1 (violet) et d2 (bleu)
        ligne(out, vers_image(R, [d["p1"], d["p2"]]), coul, ep)
        for p in vers_image(R, [d["p1"], d["p2"]]):                                      # petites flèches = bouts du diamètre
            cv2.circle(out, tuple(int(v) for v in np.round(p * 16)), int(2.5 * ep * 16), coul, -1, cv2.LINE_AA, shift=4)
    D0, VI, tx = B["D0"], (lambda pts: vers_image(R, pts)), 0.45 * ep
    for nom in ("d3", "d4"):                                                             # d3 / d4 : flèches noires aux bouts
        d = O.get(nom)
        if d is not None:
            p1, p2 = VI([d["p1"], d["p2"]])
            fleche(out, p1, p2, (0, 0, 0), ep)
            etiquette(out, nom, VI([(d["s"], d["p2"][1] + 0.28 * D0)])[0] - [12 * ep, 0], (0, 0, 0), tx)
    Pg = O.get("perigee")
    if Pg is not None:
        sens = -1 if ENTREE == "gauche" else 1
        ligne(out, VI([Pg["P0"], Pg["P1"]]), (0, 0, 255), ep)                            # LIGNE D'ANGLE (rouge)
        r_arc = 0.45 * D0                                                                # ARC en haut : axe en travers ↔ ligne rouge
        arc = [tuple(np.array(Pg["P0"]) + r_arc * (math.cos(a) * O["v"] + math.sin(a) * sens * O["u"]))
               for a in np.radians(np.linspace(0, Pg["angle"], 24))]
        ligne(out, VI(arc), (0, 0, 255), max(1, ep // 2))
        etiquette(out, f"{Pg['angle']:.0f} deg", VI([(Pg["P0"][0] - sens * 0.12 * D0, Pg["P0"][1] - 0.12 * D0)])[0]
                  - [30 * ep if sens < 0 else 0, 0], (0, 0, 255), 0.8 * tx)
        n, H_, B_ = Pg["n"], np.array(Pg["haut"]), np.array(Pg["bas"])
        ligne(out, VI([H_ - 0.35 * D0 * n, B_ + 0.35 * D0 * n]), (0, 200, 240), ep + 1)  # PÉRIGÉE (jaune), ⟂ à la tige
        etiquette(out, "perigee", VI([H_ - 0.45 * D0 * n])[0] - [28 * ep, 0], (0, 170, 230), tx)
        dec = sens * 0.12 * D0 * np.array([n[1], -n[0]])                                 # d5 juste à côté (côté entrée)
        p1, p2 = VI([H_ + dec, B_ + dec])
        fleche(out, p1, p2, (40, 170, 40), ep)
        etiquette(out, "d5", VI([0.5 * (H_ + B_) + 3.5 * dec])[0] - [10 * ep, 0], (40, 170, 40), tx)
        cv2.circle(out, tuple(int(v) for v in np.round(VI([Pg["P1"]])[0] * 16)), int(2.5 * ep * 16), (0, 0, 255), -1,
                   cv2.LINE_AA, shift=4)                                                 # point de contact
        res["perigee_xy"] = VI([Pg["haut"], Pg["bas"]])
    c = vers_image(R, [O["centre"]])[0]                                                  # POINT CENTRAL = croisement d1 × d2
    res["point_centre"] = c
    cc = tuple(int(v) for v in np.round(c * 16))                                         # coordonnées × 16 (précision 1/16 px)
    cv2.circle(out, cc, int(5 * ep * 16), (0, 0, 0), -1, cv2.LINE_AA, shift=4)            # bord noir
    cv2.circle(out, cc, int(4 * ep * 16), (0, 0, 255), -1, cv2.LINE_AA, shift=4)          # point rouge
    return out


def vue_masque(res, shape):
    """FENÊTRE 2 : masque noir et blanc (blanc = branche)."""
    m = res.get("masque")
    if m is None:
        m = np.zeros(shape[:2], np.uint8)                             # pas de branche → tout noir
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def graphique_otsu(res, w=TUILE_W, h=TUILE_H):
    """
    FENÊTRE 3 : histogramme de l'« écart de couleur au fond » (0 = fond, grand = branche), hauteur en échelle log.
    Otsu choisit le seuil (rouge) qui sépare le mieux les 2 bosses : fond (à gauche) et branche (à droite).
    Le seuil bas (orange) récupère les parties pâles de la branche (bleu clair).
    """
    g = np.full((h, w, 3), 255, np.uint8)
    P = res.get("P", {})
    hist = P.get("histo")
    if hist is None:
        return g
    x0, x1, y0, y1 = 60, w - 20, 80, h - 60                          # zone du tracé
    hl = np.log10(hist.astype(float) + 1)                             # log : le fond a beaucoup plus de pixels que la branche
    hmax = max(hl.max(), 1e-6)
    X = lambda v: int(x0 + v * (x1 - x0) / 256)                       # valeur 0…255 → position x
    haut, bas = P.get("seuil_haut", 128), P.get("seuil_bas", 0)
    for v in range(256):                                              # une barre par valeur
        coul = (170, 170, 170) if v <= bas else ((230, 180, 120) if v <= haut else (40, 80, 140))
        y = int(y1 - hl[v] / hmax * (y1 - y0))
        cv2.rectangle(g, (X(v), y), (max(X(v), X(v + 1) - 1), y1), coul, -1)
    cv2.line(g, (x0, y1), (x1, y1), (0, 0, 0), 1)                     # axe horizontal
    for v in (0, 50, 100, 150, 200, 255):                             # graduations
        cv2.putText(g, str(v), (X(v) - 10, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(g, (X(haut), y0 - 10), (X(haut), y1), (0, 0, 255), 2)    # seuil d'Otsu (rouge)
    cv2.line(g, (X(bas), y0 - 10), (X(bas), y1), (0, 140, 255), 2)   # seuil bas (orange)
    cv2.putText(g, f"Otsu = {haut:.0f}", (X(haut) + 6, y0 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(g, f"bas = {bas:.0f}", (X(bas) + 6, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2, cv2.LINE_AA)
    pct = 100 * hist[int(bas) + 1:].sum() / max(hist.sum(), 1)
    cv2.putText(g, "gris = fond    bleu clair = parties pales    marron = branche sure", (x0, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    axe = "ecart de couleur au fond (0 = fond)"
    cv2.putText(g, f"{axe}   |   branche = {pct:.1f} % de l'image", (x0, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return g


def panneau_infos(res, calib, etat, w=TUILE_W, h=TUILE_H):
    """FENÊTRE 4 : informations de l'oblong et des diamètres du nœud (en mm si calibré, sinon en px)."""
    g = np.full((h, w, 3), 255, np.uint8)
    L = []                                                            # lignes : (texte, couleur, épaisseur) ou (nom, valeur, couleur, épaisseur)
    O, B, P = res.get("O"), res.get("B"), res.get("P", {})
    u = "mm" if calib.ok else "px"
    if O is not None:
        d1, d2 = O["d1"], O["d2"]
        u2 = "mm2" if calib.ok else "px2"
        Pg = O.get("perigee")
        if Pg is not None:
            L += [("PERIGEE (jaune)", (0, 150, 200), 2),
                  ("Angle (en haut, axe en travers / ligne rouge)", f"{Pg['angle']:9.1f} deg  (j/k)", (0, 0, 220), 1),
                  ("d5 = diametre sur la perigee", f"{calib.vd(Pg['d5']):9.3f} {u}", (40, 150, 40), 2),
                  ("Perigee -> centre de l'oblong", f"{calib.vl(Pg['dist_centre']):9.3f} {u}", (0, 0, 0), 1)]
            if Pg["dans_noeud"]:
                L.append(("ATTENTION : perigee DANS le noeud -> augmenter l'angle (k)", (0, 0, 220), 1))
            if not Pg["fiable"]:
                L.append(("ATTENTION : perigee au bout / bord de l'image -> d5 douteux", (0, 0, 220), 1))
        else:
            L.append((f"PERIGEE : la ligne a {REGLAGE['angle']:.0f} deg ne touche pas le bord -> diminuer l'angle (j)",
                      (0, 0, 220), 1))
        v3 = f"{calib.vd(O['d3']['px']):9.3f} {u}" if O.get("d3") else "    -"
        v4 = f"{calib.vd(O['d4']['px']):9.3f} {u}" if O.get("d4") else "    -"
        L += [("d3 = diametre exterieur, bout d'entree", v3, (0, 0, 0), 1),
              ("d4 = diametre exterieur, autre bout", v4, (0, 0, 0), 1),
              ("SURFACE de l'oblong", f"{surface_oblong(O, calib):11.2f} {u2}", (0, 140, 200), 2)]
        L += [("OBLONG (jaune)", (0, 140, 200), 2),
              ("d1 = GRAND diametre (violet)", f"{calib.axe(d1):9.3f} {u}  ({'en travers' if 'travers' in d1['sens'] else 'le long'})", (150, 0, 150), 2),
              ("d2 = PETIT diametre (bleu)", f"{calib.axe(d2):9.3f} {u}", (200, 90, 0), 2),
              ("Rayon des arrondis", f"{calib.vl(O['rayon']):9.3f} {u}", (0, 0, 0), 1)]
    else:
        L.append((res.get("msg") or "Pas de noeud", (0, 0, 200), 2))
    if B is not None:
        L.append((f"Qualite : bords valides {B['valides']:.0f} %, bruit {P.get('bruit_bord_px', 0):.2f} px, {res.get('ms', 0):.0f} ms",
                  (90, 90, 90), 1))
    if B is not None and P.get("appendices"):
        L.append((f"Appendice(s) ignore(s) : {P['appendices']}  (trait orange = bord de la tige)", (0, 110, 230), 1))
    if res.get("N") and res["N"].get("par") == "appendice":
        L.append(("Noeud place sur la racine de l'appendice (pas de renflement net)", (0, 110, 230), 1))
    L.append((f"Segmentation : {NOM_METHODE}", (120, 60, 0), 2))
    L.append(("Mesures en mm (calibre)" if calib.ok else "En PIXELS : pas calibre -> touche c avec un etalon",
              (0, 130, 0) if calib.ok else (0, 0, 220), 2))
    for t in etat:                                                    # moyenne / calibration en cours, réglages
        L.append((t, (200, 80, 0), 1))
    y = 52
    for ligne_ in L:
        if len(ligne_) == 4:                                          # (nom, valeur, couleur, épaisseur) → 2 colonnes alignées
            t, val, coul, ep = ligne_
            cv2.putText(g, val, (395, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 if ep < 2 else 0.6, coul, ep, cv2.LINE_AA)
        else:                                                         # (texte, couleur, épaisseur) → une seule colonne
            t, coul, ep = ligne_
        cv2.putText(g, t, (14 if not t.startswith("diametre limite") else 34, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 if ep < 2 else 0.6, coul, ep, cv2.LINE_AA)
        y += 21 if ep < 2 else 25
    cv2.putText(g, "f anti-vibration  j/k angle  b fond  c calibrer  r  a moyenne  s  p  +/-  q", (14, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    return g


def mosaique(res, calib, etat):
    """Assemble les 4 fenêtres en une seule image (2 × 2), séparées par un trait sombre."""
    W, H = TUILE_W, TUILE_H
    t1 = titre(cv2.resize(vue_camera(res), (W, H), interpolation=cv2.INTER_AREA), "1 - Camera : oblong d1 d2, d3 d4, angle, PERIGEE, d5")
    t2 = titre(cv2.resize(vue_masque(res, res["img"].shape), (W, H), interpolation=cv2.INTER_AREA), "2 - Masque noir et blanc (blanc = branche)")
    t3 = titre(graphique_otsu(res, W, H), f"3 - Histogramme et seuils : {NOM_METHODE}")
    t4 = titre(panneau_infos(res, calib, etat, W, H), "4 - Informations : perigee, d1 ... d5, surface")
    sep_v = np.full((H, 4, 3), 30, np.uint8)
    sep_h = np.full((4, 2 * W + 4, 3), 30, np.uint8)
    return np.vstack([np.hstack([t1, sep_v, t2]), sep_h, np.hstack([t3, sep_v, t4])])


# =========================================================================================================
# 9. PROGRAMME PRINCIPAL
# =========================================================================================================
def main():
    arg = next((a for a in sys.argv[1:] if not a.startswith("--")), "0")
    dossier = os.path.dirname(os.path.abspath(__file__))
    calib = Calibration(os.path.join(dossier, "calibration.json"))
    print("Calibration :", f"OK ({calib.a:.5f} mm/px)" if calib.ok else "absente → mesures en pixels")
    src = SourceBasler(int(arg)) if arg.isdigit() else SourceImage(arg)
    fen = "Branche + noeud + perigee  -  q quitter"
    cv2.namedWindow(fen, cv2.WINDOW_NORMAL)
    montrer_P, moy, erreurs, dernier_moy = False, None, set(), []
    stab, stab_actif = Stabilisateur(), True                          # anti-vibration (touche f pour l'activer / l'arrêter)
    fond_img = None                                                   # photo du fond sans branche (touche b)
    fichier_cal = os.path.join(dossier, "calibration.json")
    etalons, cal_px, ref_mm = [], None, None                          # calibration : étalons mesurés / mesure en cours
    print(f"Segmentation : {NOM_METHODE}")
    print(">>> ENLÈVE la branche et appuie sur b pour apprendre le fond, puis remets la branche. <<<")
    print(f"Périgée : angle {REGLAGE['angle']:.0f}° en haut (j / k), entrée de la branche à {ENTREE}")
    print("Touches : c calibrer · r effacer calibration · a moyenne · s enregistrer · p réglages · "
          "+/- exposition · q quitter")
    try:
        while True:
            img = src.lire()
            if img is None:
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                continue
            if stab_actif:
                img = stab.ajouter(img)                               # image moyenne tant que la branche est immobile
            try:                                                  # une image ratée (branche qui bouge vite, flou…)
                                                                  # ne doit JAMAIS arrêter le programme
                res = analyser(img, calib, fond_img)
                if cal_px is not None and res.get("B") is not None:      # CALIBRATION en cours (touche c)
                    D = res["B"]["D"]                                     # diamètre brut sub-pixel, en PIXELS
                    n = len(D); Dc = D[int(0.2 * n):int(0.8 * n)]         # on garde le milieu de l'étalon (pas les bouts)
                    if np.isfinite(Dc).sum() > 20:
                        cal_px.append(float(np.nanmedian(Dc)))            # 1 valeur par image
                    if len(cal_px) >= N_MOYENNE:
                        px, sd = float(np.mean(cal_px)), float(np.std(cal_px, ddof=1))
                        etalons.append((px, ref_mm))
                        a_, b_ = calib.enregistrer(fichier_cal, etalons)
                        print(f"Étalon Ø {ref_mm:.3f} mm = {px:.3f} px (± {sd:.3f} px sur {len(cal_px)} images)")
                        print(f"→ calibration.json enregistré : D_mm = {a_:.6f} × D_px + {b_:+.4f}   "
                              f"({len(etalons)} étalon(s) ; 1 mm = {1 / a_:.2f} px)")
                        if len(etalons) >= 2:
                            for p_, m_ in etalons:
                                print(f"   contrôle : étalon {m_:.3f} mm → mesuré {a_ * p_ + b_:.4f} mm (écart {a_ * p_ + b_ - m_:+.4f})")
                        cal_px = None
                if moy is not None and res["ok"] and res["O"] is not None:   # moyenne en cours (touche a)
                    N, O = res["N"], res["O"]
                    nan, Pg = float("nan"), O["perigee"]
                    moy.append((calib.axe(O["d1"]), calib.axe(O["d2"]), surface_oblong(O, calib),
                                calib.vd(O["d3"]["px"]) if O["d3"] else nan, calib.vd(O["d4"]["px"]) if O["d4"] else nan,
                                calib.vd(Pg["d5"]) if Pg else nan, calib.vl(Pg["dist_centre"]) if Pg else nan))
                    if len(moy) >= N_MOYENNE:
                        a = np.array(moy)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            sd = np.nanstd(a, axis=0, ddof=1); mo = np.nanmean(a, axis=0)
                        u = "mm" if calib.ok else "px"
                        print(f"\n=== MOYENNE sur {len(a)} images ({u}) : valeur ± écart-type (répétabilité) ===")
                        dernier_moy = [f"MOYENNE {len(a)} images ({u}) : valeur +/- ecart-type"]
                        for nom, m_, s_ in zip(("d1", "d2", "surface oblong", "d3", "d4", "d5 (perigee)", "perigee->centre"), mo, sd):
                            print(f"  {nom:20s} = {m_:10.4f} ± {s_:.4f}")
                            if nom in ("d5 (perigee)", "perigee->centre", "surface oblong"):
                                dernier_moy.append(f"  {nom} = {m_:.3f} +/- {s_:.3f}")
                        moy = None
                etat = list(dernier_moy)                                  # texte en plus pour la fenêtre 4
                if moy is not None:
                    etat = [f"Moyenne en cours : {len(moy)}/{N_MOYENNE} - ne bouge pas la branche"]
                if cal_px is not None:
                    etat.append(f"CALIBRATION etalon {ref_mm:.3f} mm : {len(cal_px)}/{N_MOYENNE} - ne bouge pas")
                etat.append(f"Anti-vibration : {stab.n}/{N_STAB} images" + (" - STABLE (fige)" if stab.n >= N_STAB else " - en cours...") + "  (f = arreter)" if stab_actif
                            else "Anti-vibration ARRETEE (f = activer)")
                if montrer_P:
                    P = res["P"]
                    etat.append(f"[reglages] flou {2 * FLOU + 1}, morpho {2 * MORPHO + 1}, contraste min {CONTRASTE_MIN}, "
                                f"bord noeud {BORD_NOEUD_PCT:.0f} %, lissage {P.get('lissage_px', 0):.1f} px")
                dessin = vue_camera(res)                                  # image pleine résolution (pour s)
                ecran = mosaique(res, calib, etat)                        # les 4 fenêtres
                cv2.imshow(fen, ecran)
            except Exception as e:
                nom = f"{type(e).__name__}: {e}"
                if nom not in erreurs:                            # chaque erreur n'est écrite qu'une fois
                    erreurs.add(nom); print("[image ignorée]", nom)
                aff = cv2.resize(img, (2 * TUILE_W + 4, 2 * TUILE_H + 4))
                bandeau(aff, ["Image ignoree (branche qui bouge / floue)"])
                cv2.imshow(fen, aff)
                res, dessin, ecran = None, img, aff
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('b'):                                       # apprendre le fond (SANS branche !)
                fond_img = cv2.GaussianBlur(calib.corriger(img), (5, 5), 0).astype(np.float32)
                print("Fond mémorisé. Remets la branche.")
            elif k in (ord('j'), ord('k')):                           # angle de la ligne rouge
                REGLAGE["angle"] = float(np.clip(REGLAGE["angle"] + (1 if k == ord('k') else -1), 1, 89))
                print(f"Angle : {REGLAGE['angle']:.0f}°")
            elif k == ord('f'):                                       # anti-vibration marche / arrêt
                stab_actif = not stab_actif; stab = Stabilisateur()
                print("Anti-vibration :", "activée" if stab_actif else "arrêtée")
            elif k == ord('p'):
                montrer_P = not montrer_P
            elif k == ord('c'):                                       # CALIBRATION : étalon de diamètre connu
                try:
                    ref_mm = float(input("Place l'étalon (tige calibrée) à la place de la branche.\n"
                                         "Diamètre de l'étalon en mm (ex. 10.00) : ").replace(",", "."))
                    cal_px = []; print(f"Mesure de l'étalon sur {N_MOYENNE} images…")
                except ValueError:
                    print("Nombre invalide, calibration annulée.")
            elif k == ord('r'):                                       # repartir de zéro
                etalons, cal_px = [], None
                calib.ok, calib.a, calib.b = False, 1.0, 0.0
                if os.path.exists(fichier_cal):
                    d = json.load(open(fichier_cal, encoding="utf-8"))
                    for cle in ("mm_par_px", "decalage_diametre_mm", "etalons"):
                        d.pop(cle, None)
                    json.dump(d, open(fichier_cal, "w", encoding="utf-8"), indent=2)
                print("Calibration effacée → mesures en pixels.")
            elif k in (ord('+'), ord('=')):
                src.exposition(1.5); print(f"Exposition {src.expo / 1000:.2f} ms")
            elif k == ord('-'):
                src.exposition(1 / 1.5); print(f"Exposition {src.expo / 1000:.2f} ms")
            elif k == ord('a'):
                moy = []; print(f"Moyenne sur {N_MOYENNE} images…")
            elif k == ord('s'):
                st = time.strftime("%Y%m%d_%H%M%S"); base = os.path.join(dossier, f"perigee_m3_{st}")
                cv2.imwrite(base + "_image.png", img); cv2.imwrite(base + "_resultat.png", dessin)
                cv2.imwrite(base + "_4fenetres.png", ecran)
                f = os.path.join(dossier, "mesures_perigee_m3.csv"); neuf = not os.path.exists(f)
                with open(f, "a", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh, delimiter=";")
                    if neuf:
                        w.writerow(["date", "unite", "d1_grand_diametre", "d1_direction", "d2_petit_diametre", "d2_direction",
                                    "rayon_arrondis", "surface_oblong", "d3_exterieur_entree", "d4_exterieur_autre_bout",
                                    "angle_deg", "d5_perigee", "perigee_vers_centre", "perigee_dans_noeud",
                                    "perigee_haut_x_px", "perigee_haut_y_px", "perigee_bas_x_px", "perigee_bas_y_px",
                                    "centre_x_px", "centre_y_px", "methode"])
                    N, O = (res.get("N"), res.get("O")) if res else (None, None)
                    if O is not None:
                        cx, cy = res.get("point_centre", (np.nan, np.nan))
                        Pg, nan = O["perigee"], float("nan")
                        xy = np.ravel(res.get("perigee_xy", np.full(4, np.nan))) if Pg else np.full(4, np.nan)
                        w.writerow([st, "mm" if calib.ok else "px",
                                    f"{calib.axe(O['d1']):.4f}", O["d1"]["sens"], f"{calib.axe(O['d2']):.4f}", O["d2"]["sens"],
                                    f"{calib.vl(O['rayon']):.4f}", f"{surface_oblong(O, calib):.4f}",
                                    f"{calib.vd(O['d3']['px']) if O['d3'] else nan:.4f}",
                                    f"{calib.vd(O['d4']['px']) if O['d4'] else nan:.4f}",
                                    f"{REGLAGE['angle']:.1f}", f"{calib.vd(Pg['d5']) if Pg else nan:.4f}",
                                    f"{calib.vl(Pg['dist_centre']) if Pg else nan:.4f}", (Pg["dans_noeud"] if Pg else ""),
                                    *[f"{v:.2f}" for v in xy], f"{cx:.2f}", f"{cy:.2f}", NOM_METHODE])
                    else:
                        print("Pas d'oblong sur cette image : rien écrit dans le CSV.")
                print("Enregistré :", base + "_*.png  +", f)
    finally:
        src.fermer()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
