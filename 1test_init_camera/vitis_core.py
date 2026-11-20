# -*- coding: utf-8 -*-
"""
vitis_core.py — Cœur de la vision de la machine Vitis Optima (caméra 1 : branche, nœud, oblong, point milieu, Périgée).

Ce fichier ne se lance pas tout seul : il est utilisé par
    etape1_branche_noeud.py      → branche + nœud (limites précises)
    etape2_oblong_diametres.py   → + oblong autour du nœud, grand / petit diamètre
    etape3_point_milieu.py       → + point milieu du nœud (croisement de 2 droites)
    etape4_perigee.py            → + Périgée (sommet du petit axe de l'ellipse du nœud)
    calibration_mm.py            → conversion pixels → millimètres (calibration.json)

Organisation (les numéros de sections sont repris dans les fichiers Word) :
    S0  Réglages par défaut
    S1  Calibration px → mm (lecture, conversion, correction de distorsion)
    S2  Sources d'images (caméra Basler, fichier image)
    S3  Segmentation classique (4 méthodes) → masque de la branche
    S4  Redressement de la branche (axe principal) en pleine résolution
    S5  Bords SUB-PIXEL et profil de diamètre D(s)
    S6  Nœud : pic + limites (étape 1)
    S7  Oblong + grand / petit diamètre (étape 2)
    S8  Point milieu (étape 3)
    S9  Appendices (vrille / bourgeon) + ellipse RANSAC + Périgée (étape 4)
    S10 Analyse complète + dessin + application (fenêtre, curseurs, clavier, moyenne)

Convention géométrique (important pour la précision) :
    • Un pixel (i, j) a son CENTRE aux coordonnées entières (x = j, y = i). Le bord gauche du pixel est à x = j − 0,5.
    • Les bords sub-pixel sont les positions où le gris passe à mi-chemin entre fond et branche (règle des 50 %).
    • Le diamètre = distance entre le bord haut et le bord bas → c'est une vraie distance entre 2 bords (pas un comptage de pixels).
"""
import os
os.environ.setdefault("QT_QPA_PLATFORMTHEME", "none")   # évite un plantage Qt/GNOME sur Ubuntu (avant import cv2)
import sys, json, math, time, csv
import numpy as np
import cv2

# =====================================================================================================
# S0  RÉGLAGES PAR DÉFAUT (modifiables ici ou avec les curseurs de la fenêtre)
# =====================================================================================================
DEFAUT = {
    # --- caméra
    "expo_us": 8000.0,          # temps d'exposition (µs)
    "gain_db": 0.0,             # gain (dB) : 0 = moins de bruit
    "fps": 20.0,                # cadence demandée à la caméra
    # --- segmentation (S3)
    "methode": 1,               # 1 Otsu, 2 seuil manuel, 3 couleur, 4 fond appris
    "seuil": 100,               # seuil manuel (méthode 2)
    "seuil_fond": 30,           # écart minimal avec le fond (méthode 4)
    "flou": 2,                  # demi-taille du flou avant seuillage (2 → noyau 5×5)
    "morpho": 2,                # demi-taille du nettoyage morphologique (2 → 5×5)
    "aire_min": 5000,           # aire minimale (px²) d'une forme gardée
    # --- tige / bords (S5)
    "coupe_app_pct": 60,        # appendices plus fins que 60 % du diamètre = supprimés (vrille, pétiole)
    "contraste_min": 15,        # contraste minimal fond/branche (niveaux de gris) pour accepter un bord
    "lissage_pct": 4,           # lissage du profil : écart-type = 4 % du diamètre de la tige
    # --- nœud (S6)
    "bord_pct": 20,             # limites du nœud : base + 20 % de la hauteur du renflement
    "renfl_min_pct": 5,         # renflement minimal pour accepter un nœud (% du diamètre de base)
    "marge_bouts_pct": 80,      # bouts de la branche ignorés : 80 % du diamètre à chaque bout
    # --- Périgée (S9)
    "perigee_cote": "vrille",   # vrille | mandrin | haut | bas  (côté choisi pour le sommet du petit axe)
    "mandrin_cote": "droite",   # côté du mandrin dans l'image (utilisé si perigee_cote = mandrin ou vrille absente)
    "ransac_iter": 300,         # essais RANSAC pour l'ellipse
    "ransac_tol_px": 1.5,       # distance max (px) d'un point « bon » à l'ellipse
    # --- moyenne
    "n_moyenne": 15,            # nombre d'images moyennées avec la touche « a »
}
COLONNES_CSV = ["diametre_base", "bruit_bord", "diametre_noeud_max", "longueur_noeud", "renflement_pct", "position_noeud",
                "grand_diametre", "petit_diametre", "oblong_longueur", "oblong_hauteur", "milieu_s", "milieu_ecart_pic", "diametre_au_milieu",
                "ellipse_grand_axe", "ellipse_petit_axe", "ellipse_rms_px", "perigee_s", "perigee_t"]
NOMS_METHODES = {1: "Otsu", 2: "Seuil manuel", 3: "Couleur (saturation)", 4: "Fond appris"}


# =====================================================================================================
# S1  CALIBRATION px → mm
# =====================================================================================================
class Calibration:
    """
    Contient la conversion pixels → mm, lue dans calibration.json (écrit par calibration_mm.py).
      a  : mm par pixel (échelle)
      b  : décalage en mm appliqué aux DIAMÈTRES (corrige le biais de bord ; 0 si une seule référence)
      K, dist : matrice caméra + distorsion (optionnel, calibration damier)
    Sans fichier : ok = False → les mesures restent en pixels (on ne fabrique jamais des mm faux).
    """
    def __init__(self, chemin="calibration.json"):
        self.ok, self.a, self.b, self.K, self.dist, self.info = False, 1.0, 0.0, None, None, ""
        self._maps = None
        if chemin and os.path.exists(chemin):
            with open(chemin, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.a = float(d.get("mm_par_px", 1.0))
            self.b = float(d.get("decalage_diametre_mm", 0.0))
            self.ok = "mm_par_px" in d
            if d.get("camera_matrix") is not None:
                self.K = np.array(d["camera_matrix"], np.float64)
                self.dist = np.array(d["dist_coeffs"], np.float64)
            self.info = d.get("info", "")

    def diam(self, px):
        """Diamètre en px → texte (mm si calibré). Formule : D_mm = a·D_px + b."""
        return f"{self.a * px + self.b:.3f} mm" if self.ok else f"{px:.2f} px"

    def long(self, px):
        """Longueur / position le long de la branche (pas de décalage de bord)."""
        return f"{self.a * px:.3f} mm" if self.ok else f"{px:.2f} px"

    def val_diam(self, px):
        return self.a * px + self.b if self.ok else px

    def val_long(self, px):
        return self.a * px if self.ok else px

    def unite(self):
        return "mm" if self.ok else "px"

    def corriger(self, img):
        """Corrige la distorsion de l'objectif si une calibration damier existe (sinon renvoie l'image telle quelle)."""
        if self.K is None:
            return img
        h, w = img.shape[:2]
        if self._maps is None or self._maps[2] != (w, h):
            m1, m2 = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, (w, h), cv2.CV_16SC2)
            self._maps = (m1, m2, (w, h))
        return cv2.remap(img, self._maps[0], self._maps[1], cv2.INTER_LINEAR)


# =====================================================================================================
# S2  SOURCES D'IMAGES
# =====================================================================================================
class SourceBasler:
    """Caméra Basler (pypylon) en réglages manuels, image BGR. Fermer pylon Viewer avant !"""
    def __init__(self, index, P):
        from pypylon import pylon
        self.pylon = pylon
        infos = pylon.TlFactory.GetInstance().EnumerateDevices()
        if index >= len(infos):
            raise SystemExit(f"Caméra n° {index} absente ({len(infos)} trouvée(s)). Vérifie le câble / ferme pylon Viewer.")
        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(infos[index]))
        self.cam.Open()
        di = self.cam.GetDeviceInfo()
        self.nom = f"{di.GetModelName()} S/N {di.GetSerialNumber()}"
        print("Caméra :", self.nom)
        for nom, val in [("ExposureAuto", "Off"), ("GainAuto", "Off"), ("Gain", P["gain_db"]),
                         ("ExposureTime", P["expo_us"]), ("AcquisitionFrameRateEnable", True),
                         ("AcquisitionFrameRate", P["fps"])]:
            self._regler(nom, val)
        self._regler("BalanceWhiteAuto", "Once")               # caméra couleur : balance des blancs une fois
        self.conv = pylon.ImageFormatConverter()
        self.conv.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.expo = P["expo_us"]

    def _regler(self, nom, val):
        try:
            getattr(self.cam, nom).Value = val
        except Exception:
            pass

    def lire(self):
        r = self.cam.RetrieveResult(100, self.pylon.TimeoutHandling_Return)   # 100 ms max → fenêtre réactive
        img = None
        if r is not None and r.IsValid():
            if r.GrabSucceeded():
                img = self.conv.Convert(r).Array.copy()
            r.Release()
        return img

    def exposition(self, facteur):
        self.expo = float(np.clip(self.expo * facteur, 30, 300000))
        self._regler("ExposureTime", self.expo)

    def fermer(self):
        try:
            self.cam.StopGrabbing(); self.cam.Close()
        except Exception:
            pass


class SourceImage:
    """Image enregistrée (pour tester sans caméra). Renvoie toujours la même image."""
    def __init__(self, chemin):
        self.img = cv2.imread(chemin, cv2.IMREAD_COLOR)
        if self.img is None:
            raise SystemExit("Image illisible : " + chemin)
        self.nom, self.expo = os.path.basename(chemin), 0

    def lire(self):
        return self.img.copy()

    def exposition(self, f):
        pass

    def fermer(self):
        pass


def ouvrir_source(arg, P):
    """'0', '1' → caméra Basler n° 0/1 ; sinon → fichier image."""
    return SourceBasler(int(arg), P) if arg.isdigit() else SourceImage(arg)


# =====================================================================================================
# S3  SEGMENTATION CLASSIQUE → masque binaire de la branche (255 = branche)
# =====================================================================================================
def segmenter(img, P, fond=None, inverser=False):
    """Renvoie (masque, contour). Le masque sert seulement de GUIDE : la précision vient des bords sub-pixel (S5)."""
    k = 2 * int(P["flou"]) + 1
    flou = cv2.GaussianBlur(img, (k, k), 0)
    gris = cv2.cvtColor(flou, cv2.COLOR_BGR2GRAY)
    m = int(P["methode"])
    if m == 1:                                                            # Otsu (fond clair, branche sombre)
        _, msk = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif m == 2:                                                          # seuil manuel
        _, msk = cv2.threshold(gris, int(P["seuil"]), 255, cv2.THRESH_BINARY_INV)
    elif m == 3:                                                          # saturation (branche colorée, fond blanc)
        s = cv2.cvtColor(flou, cv2.COLOR_BGR2HSV)[:, :, 1]
        _, msk = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:                                                                 # fond appris
        if fond is None:
            return np.zeros(gris.shape, np.uint8), None
        d = cv2.cvtColor(cv2.absdiff(flou, fond), cv2.COLOR_BGR2GRAY)
        _, msk = cv2.threshold(d, int(P["seuil_fond"]), 255, cv2.THRESH_BINARY)
    if inverser:
        msk = cv2.bitwise_not(msk)
    t = 2 * int(P["morpho"]) + 1
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
    msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, el)                       # enlève les poussières
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, el)                      # bouche les petits trous
    cs, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [c for c in cs if cv2.contourArea(c) >= P["aire_min"]]
    if not cs:
        return msk * 0, None
    c = max(cs, key=cv2.contourArea)                                      # la plus grande forme = la branche
    propre = np.zeros_like(msk)
    cv2.drawContours(propre, [c], -1, 255, cv2.FILLED)
    return propre, c


# =====================================================================================================
# S4  REDRESSEMENT (pleine résolution, recadré) : la branche devient horizontale
# =====================================================================================================
def redresser(gris, masque, marge=30):
    """
    Calcule l'axe de la branche (ACP sur les pixels du masque) et construit une image redressée :
      axe s (colonnes) = le long de la branche, axe t (lignes) = en travers.
    Renvoie dict : gray_r (float32, interpolation BICUBIQUE), mask_r, M (image→redressé), Minv, angle (°).
    """
    ys, xs = np.nonzero(masque)
    if len(xs) < 100:
        return None
    if len(xs) > 60000:                                                   # sous-échantillonnage pour l'ACP (rapide)
        sel = np.random.default_rng(0).choice(len(xs), 60000, replace=False)
        xs_, ys_ = xs[sel], ys[sel]
    else:
        xs_, ys_ = xs, ys
    pts = np.column_stack([xs_, ys_]).astype(np.float64)
    c = pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov((pts - c).T))
    u = vecs[:, np.argmax(vals)]                                          # direction de la branche
    if u[0] < 0:                                                          # s croît de gauche à droite dans l'image
        u = -u
    v = np.array([-u[1], u[0]])                                           # direction perpendiculaire (vers le bas)
    P_all = np.column_stack([xs, ys]).astype(np.float64) - c
    s, t = P_all @ u, P_all @ v
    marge = max(marge, int(0.25 * (t.max() - t.min())) + 10)              # assez de fond autour pour mesurer les niveaux
    s0, s1, t0, t1 = s.min() - marge, s.max() + marge, t.min() - marge, t.max() + marge
    W, H = int(math.ceil(s1 - s0)) + 1, int(math.ceil(t1 - t0)) + 1
    # M : (x, y) image → (s − s0, t − t0) redressé
    M = np.array([[u[0], u[1], -(u @ c) - s0],
                  [v[0], v[1], -(v @ c) - t0]], np.float64)
    gray_r = cv2.warpAffine(gris.astype(np.float32), M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    mask_r = cv2.warpAffine(masque, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    Minv = cv2.invertAffineTransform(M)
    angle = math.degrees(math.atan2(u[1], u[0]))
    return {"gray_r": gray_r, "mask_r": mask_r, "M": M, "Minv": Minv, "angle": angle, "u": u, "v": v}


def vers_image(R, pts):
    """Points du repère redressé (s, t) → coordonnées image (x, y). pts : liste ou tableau N×2."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    return np.hstack([p, np.ones((len(p), 1))]) @ R["Minv"].T


# =====================================================================================================
# S5  BORDS SUB-PIXEL + PROFIL DE DIAMÈTRE
# =====================================================================================================
def plus_grande_composante(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return np.where(lab == k, 255, 0).astype(np.uint8)


def tige(mask_r, P):
    """Enlève les appendices fins (vrille, pétiole) : ouverture avec un disque de coupe_app_pct % du diamètre."""
    larg = (mask_r > 0).sum(axis=0)
    D0 = float(np.median(larg[larg > 0])) if (larg > 0).any() else 0.0
    k = max(3, D0 * P["coupe_app_pct"] / 100)
    f = min(1.0, 31.0 / k)                                                # on travaille en réduit : rapide (la tige n'est qu'un GUIDE)
    petit = cv2.resize(mask_r, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else mask_r
    _, petit = cv2.threshold(petit, 127, 255, cv2.THRESH_BINARY)
    kk = max(3, int(k * f) | 1)
    t = cv2.morphologyEx(petit, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    if f < 1:
        t = cv2.resize(t, (mask_r.shape[1], mask_r.shape[0]), interpolation=cv2.INTER_LINEAR)
        _, t = cv2.threshold(t, 127, 255, cv2.THRESH_BINARY)
        t = cv2.bitwise_and(t, mask_r)                                    # jamais plus large que la branche
    return plus_grande_composante(t), D0


def _croisements(G, y0, niveau, w):
    """
    Version « vectorisée » (toutes les colonnes d'un coup) : pour chaque colonne j, G[j] est le profil de gris
    lu de y0[j] − w à y0[j] + w. On cherche où il traverse niveau[j] (changement de signe) ; si plusieurs
    traversées, on garde la plus proche du bord du masque (au centre de la fenêtre). Interpolation linéaire
    entre les 2 pixels qui encadrent le niveau → position SUB-PIXEL.
    """
    d = G - niveau[:, None]
    chg = (np.sign(d[:, :-1]) * np.sign(d[:, 1:]) <= 0) & (d[:, :-1] != d[:, 1:])
    k = np.arange(d.shape[1] - 1)[None, :]
    centre = (d.shape[1] - 1) / 2.0
    cout = np.where(chg, np.abs(k + 0.5 - centre), np.inf)                 # distance au bord du masque
    kb = np.argmin(cout, axis=1)
    ok = np.isfinite(cout[np.arange(len(kb)), kb])
    g0 = d[np.arange(len(kb)), kb]; g1 = d[np.arange(len(kb)), kb + 1]
    frac = np.where(g0 != g1, g0 / np.where(g0 != g1, g0 - g1, 1), 0.0)
    pos = y0 - w + kb + frac                                              # position du bord (sub-pixel)
    return np.where(ok, pos, np.nan)


def bords_subpixel(R, tige_r, P):
    """
    Pour chaque colonne s de l'image redressée :
      • bord du masque « tige » = position approximative (± 1-2 px),
      • niveaux locaux : gris du FOND (au-dessus / au-dessous) et gris de la BRANCHE (à l'intérieur), par médiane,
      • bord précis = croisement du niveau 50 % (moyenne fond/branche) → précision ~ 0,1 px.
    Renvoie dict : s, haut, bas (sub-pixel, NaN si bord non fiable), D = bas − haut, centre.
    """
    g = R["gray_r"]
    H = g.shape[0]
    col = tige_r > 0
    pres = np.nonzero(col.any(axis=0))[0]
    if len(pres) == 0:
        return {"s": pres.astype(np.float64), "haut": np.array([]), "bas": np.array([]), "D": np.array([]), "centre": np.array([])}
    sub = col[:, pres]
    yt = sub.argmax(axis=0)                                               # 1er pixel de tige (haut)
    yb = H - 1 - sub[::-1].argmax(axis=0)                                 # dernier pixel de tige (bas)
    D0 = float(np.median(yb - yt + 1))
    w = int(max(3, round(0.06 * D0)))                                     # demi-fenêtre de recherche autour du bord
    e1, e2 = int(max(3, 0.05 * D0)), int(max(8, 0.15 * D0))               # bande où on mesure les niveaux (hors du flou)
    lire = lambda rows: g[np.clip(rows, 0, H - 1), pres[:, None]]         # lit g aux lignes 'rows' de chaque colonne
    bande = np.arange(e1, e2)[None, :]
    fen = np.arange(-w, w + 1)[None, :]
    out = {}
    for nom, y_m, sens in (("haut", yt - 0.5, -1), ("bas", yb + 0.5, +1)):
        y_ref = (yt if nom == "haut" else yb)[:, None]
        I_fond = np.median(lire(y_ref + sens * bande), axis=1)            # fond, du côté extérieur
        I_br = np.median(lire(y_ref - sens * bande), axis=1)              # branche, du côté intérieur
        niveau = 0.5 * (I_fond + I_br)
        y0 = np.round(y_m).astype(int)
        G = lire(y0[:, None] + fen)
        pos = _croisements(G, y0.astype(float), niveau, w)
        ok = np.abs(I_fond - I_br) >= P["contraste_min"]                  # contraste suffisant ?
        ok &= (y_ref[:, 0] - e2 >= 0) & (y_ref[:, 0] + e2 < H)            # bandes dans l'image
        out[nom] = np.where(ok, pos, np.nan)
    return {"s": pres.astype(np.float64), "haut": out["haut"], "bas": out["bas"],
            "D": out["bas"] - out["haut"], "centre": 0.5 * (out["haut"] + out["bas"])}


def lissage_gauss(v, sigma):
    """Lissage gaussien 1-D qui ignore les NaN (trous) : moyenne pondérée des voisins valides."""
    ok = np.isfinite(v)
    if sigma < 0.5 or ok.sum() < 3:
        out = v.copy()
    else:
        r = int(3 * sigma)
        ker = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        num = np.convolve(np.where(ok, v, 0.0), ker, mode="same")
        den = np.convolve(ok.astype(float), ker, mode="same")
        out = np.where(den > 1e-6, num / np.maximum(den, 1e-9), np.nan)
    idx = np.arange(len(v))                                               # comble les derniers trous éventuels
    okk = np.isfinite(out)
    if okk.sum() >= 2 and not okk.all():
        out[~okk] = np.interp(idx[~okk], idx[okk], out[okk])
    return out


def filtre_hampel(v, fen, k=3.0, mini=1.0):
    """
    Filtre de Hampel : un point est rejeté (NaN) s'il s'écarte de la médiane de ses voisins
    de plus de k × écart robuste (MAD), avec un minimum de 'mini' pixel. Sert à éliminer un bord
    trouvé sur une vrille ou un bourgeon collé à la tige au lieu du vrai bord.
    """
    h = fen // 2
    if len(v) < 2 * h + 1:
        return v.copy()
    pad = np.concatenate([np.full(h, np.nan), v, np.full(h, np.nan)])
    W = np.lib.stride_tricks.sliding_window_view(pad, 2 * h + 1)           # une ligne = les voisins d'un point
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(W, axis=1)
            mad = 1.4826 * np.nanmedian(np.abs(W - med[:, None]), axis=1)
    rejet = np.abs(v - med) > np.maximum(k * np.nan_to_num(mad, nan=0.0), mini)
    return np.where(rejet, np.nan, v)


def ligne_centrale(s, centre, ok):
    """Ligne centrale robuste : parabole ajustée sur les milieux (haut+bas)/2, en rejetant les points aberrants."""
    sel = ok.copy()
    for _ in range(3):
        if sel.sum() < 10:
            break
        p = np.polyfit(s[sel], centre[sel], 2)
        res = centre - np.polyval(p, s)
        mad = 1.4826 * np.median(np.abs(res[sel])) + 1e-6
        sel = ok & (np.abs(res) < max(3 * mad, 0.5))
    return np.polyval(p, s) if ok.sum() >= 10 else np.where(ok, centre, np.nanmean(centre) if ok.any() else 0.0)


def profil(R, P):
    """
    Profil complet de la branche : bords sub-pixel + diamètre brut + diamètre lissé.
    Un bourgeon ou une vrille CACHE le vrai bord d'un côté (souvent juste au nœud !). Dans ce cas :
      rayon haut = ligne centrale − bord haut, rayon bas = bord bas − ligne centrale,
      si un seul bord est fiable → D = 2 × rayon de ce côté (symétrie de la tige par rapport à son axe).
    """
    t, D0 = tige(R["mask_r"], P)
    B = bords_subpixel(R, t, P)
    if len(B["s"]) < 30:
        return None
    sv = B["s"]
    okh, okb = np.isfinite(B["haut"]), np.isfinite(B["bas"])
    if (okh & okb).sum() < max(30, 0.25 * len(sv)):                       # image bougée/floue : trop peu de bords nets
        return None
    cl = ligne_centrale(sv, B["centre"], okh & okb)                       # axe de la branche (robuste)
    rh, rb = cl - B["haut"], B["bas"] - cl                                # rayons haut / bas
    fen = max(5, int(0.15 * D0))
    rh, rb = filtre_hampel(rh, fen), filtre_hampel(rb, fen)               # rejette les bords « collés » à une vrille
    okh, okb = np.isfinite(rh), np.isfinite(rb)
    B["D"] = np.where(okh & okb, rh + rb, np.where(okh, 2 * rh, np.where(okb, 2 * rb, np.nan)))
    B["D_un_cote"] = (okh ^ okb)                                          # colonnes estimées avec un seul bord
    rh_f, rb_f = np.where(okh, rh, rb), np.where(okb, rb, rh)              # rayon manquant ← rayon de l'autre côté
    B["centre"] = np.where(okh & okb, B["centre"], cl)
    B["ligne_centrale"] = cl
    sig = max(1.0, P["lissage_pct"] / 100 * D0)
    B["D_lisse"] = lissage_gauss(B["D"], sig)
    B["haut_lisse"] = cl - lissage_gauss(rh_f, sig)
    B["bas_lisse"] = cl + lissage_gauss(rb_f, sig)
    B["D0"], B["sigma"], B["tige_r"] = D0, sig, t
    B["valides_pct"] = 100 * np.isfinite(B["D"]).mean()                   # % de colonnes avec 2 bords fiables
    return B


# =====================================================================================================
# S6  NŒUD : pic sub-pixel + limites sub-pixel (ÉTAPE 1)
# =====================================================================================================
def _interp(sv, v, x):
    return float(np.interp(x, sv, v))


def _traverse(sv, L, i0, niveau, pas):
    """Part de l'indice i0 (pic) et avance (pas = −1 ou +1) jusqu'à passer sous 'niveau' ; position interpolée."""
    i = i0
    while 0 <= i + pas < len(L) and L[i + pas] > niveau:
        i += pas
    if not (0 <= i + pas < len(L)):
        return float(sv[i])
    a, b = L[i], L[i + pas]                                               # a > niveau ≥ b
    f = (a - niveau) / (a - b) if a != b else 0.0
    return float(sv[i] + f * (sv[i + pas] - sv[i]))


def noeud(B, P):
    """
    1) on ignore les bouts (coupes) ; 2) diamètre de base = médiane de la TIGE hors nœud (2 itérations) ;
    3) pic = maximum du profil lissé, affiné par une parabole (sub-pixel) ;
    4) limites = croisement du niveau base + bord_pct % × (pic − base), interpolé (sub-pixel).
    """
    sv, L = B["s"], B["D_lisse"]
    D0 = B["D0"]
    marge = P["marge_bouts_pct"] / 100 * D0
    zone = (sv >= sv[0] + marge) & (sv <= sv[-1] - marge)
    if zone.sum() < 10:
        return None
    iz = np.nonzero(zone)[0]
    i_pic = iz[int(np.argmax(L[iz]))]
    # --- pic sub-pixel (parabole sur 3 points)
    if 0 < i_pic < len(L) - 1:
        y0, y1, y2 = L[i_pic - 1], L[i_pic], L[i_pic + 1]
        den = y0 - 2 * y1 + y2
        dx = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        dx = float(np.clip(dx, -1, 1))
        s_pic = sv[i_pic] + dx * (sv[i_pic + 1] - sv[i_pic])
        D_pic = y1 - 0.25 * (y0 - y2) * dx
    else:
        s_pic, D_pic = sv[i_pic], L[i_pic]
    base = float(np.median(L[iz]))
    s_g = s_d = s_pic
    for _ in range(3):                                                    # base recalculée SANS la zone du nœud
        niveau = base + P["bord_pct"] / 100 * (D_pic - base)
        s_g = _traverse(sv, L, i_pic, niveau, -1)
        s_d = _traverse(sv, L, i_pic, niveau, +1)
        exclu = (sv > s_g - 0.3 * D0) & (sv < s_d + 0.3 * D0)
        tg = zone & ~exclu
        if tg.sum() > 10:
            base = float(np.median(L[tg]))
    niveau = base + P["bord_pct"] / 100 * (D_pic - base)
    renfl = 100 * (D_pic / base - 1) if base > 0 else 0
    # bruit de mesure du bord : écart brut − lissé sur la tige (MAD robuste)
    tg = zone & ~((sv > s_g - 0.3 * D0) & (sv < s_d + 0.3 * D0)) & np.isfinite(B["D"])
    bruit = float(1.4826 * np.median(np.abs(B["D"][tg] - L[tg]))) if tg.sum() > 10 else float("nan")
    return {"ok": renfl >= P["renfl_min_pct"], "s_pic": float(s_pic), "D_pic": float(D_pic), "base": base,
            "renfl": float(renfl), "niveau": float(niveau), "s_g": s_g, "s_d": s_d, "longueur": s_d - s_g,
            "s_debut": float(sv[0]), "s_fin": float(sv[-1]), "a": float(sv[iz[0]]), "b": float(sv[iz[-1]]),
            "bruit_px": bruit}


# =====================================================================================================
# S7  OBLONG + GRAND / PETIT DIAMÈTRE (ÉTAPE 2)
# =====================================================================================================
def oblong(B, N):
    """
    Oblong = rectangle aux bouts arrondis encadrant le nœud :
      côtés gauche / droit = les 2 LIMITES parallèles du nœud (s_g, s_d),
      côtés haut / bas     = tangentes au point le plus haut / le plus bas du nœud,
      rayon des arrondis   = la moitié du plus petit côté → forme « stade » (oblong).
    Grand diamètre = diamètre maximal du nœud (au pic) ; petit diamètre = diamètre minimal dans le nœud (aux limites).
    """
    sv = B["s"]
    m = (sv >= N["s_g"]) & (sv <= N["s_d"])
    if m.sum() < 3:
        return None
    t_haut = float(np.nanmin(B["haut_lisse"][m]))                         # point le plus haut du nœud
    t_bas = float(np.nanmax(B["bas_lisse"][m]))                           # point le plus bas
    L, Hh = N["s_d"] - N["s_g"], t_bas - t_haut
    D_g = _interp(sv, B["D_lisse"], N["s_g"]); D_d = _interp(sv, B["D_lisse"], N["s_d"])
    i_min = np.nonzero(m)[0][int(np.argmin(B["D_lisse"][m]))]
    return {"s_g": N["s_g"], "s_d": N["s_d"], "t_haut": t_haut, "t_bas": t_bas, "L": L, "H": Hh,
            "rayon": 0.5 * min(L, Hh), "centre": (0.5 * (N["s_g"] + N["s_d"]), 0.5 * (t_haut + t_bas)),
            "D_grand": N["D_pic"], "s_grand": N["s_pic"],
            "D_petit": float(min(D_g, D_d, B["D_lisse"][i_min])), "D_g": D_g, "D_d": D_d}


def contour_oblong(O, n=24):
    """Points (s, t) du contour de l'oblong (rectangle aux coins arrondis, rayon = moitié du petit côté)."""
    s0, s1, t0, t1, r = O["s_g"], O["s_d"], O["t_haut"], O["t_bas"], O["rayon"]
    coins = [((s1 - r, t0 + r), -90, 0), ((s1 - r, t1 - r), 0, 90), ((s0 + r, t1 - r), 90, 180), ((s0 + r, t0 + r), 180, 270)]
    pts = []
    for (cs, ct), a0, a1 in coins:
        for a in np.linspace(math.radians(a0), math.radians(a1), n):
            pts.append((cs + r * math.cos(a), ct + r * math.sin(a)))
    return np.array(pts)


# =====================================================================================================
# S8  POINT MILIEU (ÉTAPE 3)
# =====================================================================================================
def point_milieu(B, N, O):
    """
    Point milieu de l'OBLONG, construit comme demandé :
      1) les 2 côtés parallèles de l'oblong sont les diamètres de la branche aux 2 limites du nœud ;
         on prend le CENTRE de chacun : C_g et C_d → la droite C_g–C_d traverse la branche par son milieu ;
      2) les 2 autres côtés (haut et bas) ont aussi un centre : le milieu entre les 2 limites ;
         la droite qui les relie est un DIAMÈTRE, perpendiculaire à l'axe, au milieu de l'oblong ;
      3) M = croisement de ces 2 droites = point milieu du nœud.
    On donne aussi l'écart entre M et le pic du nœud (dissymétrie du nœud).
    """
    sv = B["s"]
    cen = lissage_gauss(B["centre"], B["sigma"])                          # ligne centrale (milieux des diamètres), lissée
    Cg = np.array([N["s_g"], _interp(sv, cen, N["s_g"])])
    Cd = np.array([N["s_d"], _interp(sv, cen, N["s_d"])])
    d = Cd - Cg
    if np.linalg.norm(d) < 1e-6:
        return None
    u = d / np.linalg.norm(d)                                             # direction de l'axe local
    nrm = np.array([-u[1], u[0]])                                         # perpendiculaire (direction des diamètres)
    Q = np.array([0.5 * (N["s_g"] + N["s_d"]), 0.5 * (O["t_haut"] + O["t_bas"])])   # milieu des côtés haut/bas
    lam = float((Q - Cg) @ u)                                             # croisement : M = Cg + λ·u, (M − Q) ⟂ u
    M = Cg + lam * u
    s_m = float(M[0])
    D_m = _interp(sv, B["D_lisse"], s_m)                                  # diamètre de la branche au point milieu
    G1, G2 = M - 0.5 * D_m * nrm, M + 0.5 * D_m * nrm                     # diamètre passant par M
    return {"Cg": Cg, "Cd": Cd, "M": M, "u": u, "n": nrm, "G1": G1, "G2": G2, "D_milieu": D_m,
            "inclinaison_deg": math.degrees(math.atan2(u[1], u[0])),
            "ecart_pic": float(N["s_pic"] - s_m)}


# =====================================================================================================
# S9  APPENDICES + ELLIPSE RANSAC + PÉRIGÉE (ÉTAPE 4)
# =====================================================================================================
def appendices(R, B, N):
    """Vrille / bourgeon = parties du masque qui ne sont pas la tige, près du nœud."""
    app = cv2.bitwise_and(R["mask_r"], cv2.bitwise_not(cv2.dilate(B["tige_r"], np.ones((5, 5), np.uint8))))
    app = cv2.morphologyEx(app, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(app, 8)
    cen = lissage_gauss(B["centre"], B["sigma"])
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 0.02 * B["D0"] ** 2:
            continue
        ys, xs = np.nonzero(lab == i)
        if abs(np.median(xs) - N["s_pic"]) > 1.5 * B["D0"] + N["longueur"]:
            continue                                                       # trop loin du nœud
        (_, _), (w, h), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        Lg, Wd = max(w, h), max(1.0, min(w, h))
        off = ys - np.interp(xs, B["s"], cen)                             # + = sous l'axe (vers le bas de l'image redressée)
        k = int(np.argmax(np.abs(off)))
        out.append({"type": "vrille" if Lg / Wd >= 2.5 else "bourgeon", "aire": float(len(xs)),
                    "cote": "bas" if off.mean() > 0 else "haut", "pointe": (float(xs[k]), float(ys[k])),
                    "base": (float(np.median(xs)), float(np.interp(np.median(xs), B["s"], cen) + np.sign(off.mean()) * 0.5 * B["D0"])),
                    "pix": (xs, ys)})
    vr = [a for a in out if a["type"] == "vrille"]
    return (max(vr, key=lambda a: a["aire"]) if vr else None), out


def _dist_ellipse(p, e):
    (cx, cy), (w, h), a = e
    t = math.radians(a)
    x = (p[:, 0] - cx) * math.cos(t) + (p[:, 1] - cy) * math.sin(t)
    y = -(p[:, 0] - cx) * math.sin(t) + (p[:, 1] - cy) * math.cos(t)
    return np.abs(np.sqrt((x / (w / 2)) ** 2 + (y / (h / 2)) ** 2) - 1) * min(w, h) / 2


def ellipse_noeud(B, N, P):
    """Ellipse robuste (RANSAC + fitEllipseDirect) sur les bords SUB-PIXEL haut et bas, entre les 2 limites."""
    sv = B["s"]
    m = (sv >= N["s_g"]) & (sv <= N["s_d"]) & np.isfinite(B["haut"]) & np.isfinite(B["bas"])
    pts = np.vstack([np.column_stack([sv[m], B["haut"][m]]), np.column_stack([sv[m], B["bas"][m]])]).astype(np.float32)
    if len(pts) < 12:
        return None
    rng = np.random.default_rng(0)
    best, bn = None, 0
    tol = P["ransac_tol_px"]
    for _ in range(int(P["ransac_iter"])):
        s = pts[rng.choice(len(pts), 6, replace=False)]
        try:
            e = cv2.fitEllipseDirect(s)
        except cv2.error:
            continue
        if not np.all(np.isfinite(e[1])) or min(e[1]) < 3 or max(e[1]) > 6 * B["D0"]:
            continue
        k = int((_dist_ellipse(pts, e) < tol).sum())
        if k > bn:
            best, bn = e, k
    if best is None:
        return None
    inl = pts[_dist_ellipse(pts, best) < tol]
    E = cv2.fitEllipseDirect(inl) if len(inl) >= 6 else best              # ré-ajustement sur les bons points
    (cx, cy), (w, h), ang = E
    t = math.radians(ang)
    ax_w = np.array([math.cos(t), math.sin(t)]); ax_h = np.array([-math.sin(t), math.cos(t)])
    a, b, U, V = w / 2, h / 2, ax_w, ax_h
    if a < b:
        a, b, U, V = b, a, V, U                                           # U = grand axe, V = petit axe
    res = _dist_ellipse(inl, E)
    return {"E": E, "C": np.array([cx, cy]), "a": a, "b": b, "U": U, "V": V,
            "inliers": int(len(inl)), "total": int(len(pts)), "rms_px": float(np.sqrt(np.mean(res ** 2)))}


def perigee(B, N, ell, vrille, P):
    """
    Périgée = SOMMET DU PETIT AXE de l'ellipse du nœud (un POINT sur le bord du nœud).
    Il y a 2 sommets (C + b·V et C − b·V) : on choisit
      • côté vrille si une vrille est vue (règle par défaut, perigee_cote = « vrille »),
      • sinon côté mandrin (mandrin_cote = droite / gauche), ou haut / bas si demandé.
    """
    if ell is None:
        return None
    c1, c2 = ell["C"] + ell["b"] * ell["V"], ell["C"] - ell["b"] * ell["V"]
    regle = P["perigee_cote"]
    if regle == "vrille" and vrille is not None:
        base = np.array(vrille["base"])
        Pt, raison = (c1, "côté vrille") if np.linalg.norm(c1 - base) <= np.linalg.norm(c2 - base) else (c2, "côté vrille")
    elif regle in ("haut", "bas"):
        Pt = min((c1, c2), key=lambda p: p[1]) if regle == "haut" else max((c1, c2), key=lambda p: p[1])
        raison = f"côté {regle}"
    else:
        sg = 1 if P["mandrin_cote"] == "droite" else -1
        Pt = max((c1, c2), key=lambda p: sg * p[0] + 1e-3 * p[1])
        raison = "côté mandrin" + (" (vrille non vue)" if regle == "vrille" else "")
    return {"P": Pt, "autre": c2 if Pt is c1 else c1, "raison": raison}


# =====================================================================================================
# S10  ANALYSE COMPLÈTE
# =====================================================================================================
def analyser(img, P, calib, etape=4, fond=None, inverser=False):
    """Enchaîne les sections S3 → S9 jusqu'à l'étape demandée. Renvoie un dictionnaire de résultats."""
    t0 = time.perf_counter()
    img = calib.corriger(img)
    out = {"img": img, "ok": False, "msg": ""}
    masque, contour = segmenter(img, P, fond, inverser)
    out.update(masque=masque, contour=contour)
    if contour is None:
        out["msg"] = "Enleve la branche puis appuie sur b" if (P["methode"] == 4 and fond is None) else "Aucune branche trouvee"
        return out
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    R = redresser(gris, masque)
    if R is None:
        out["msg"] = "Branche trop petite"; return out
    B = profil(R, P)
    if B is None:
        out["msg"] = "Profil impossible (contraste ?)"; return out
    out.update(R=R, B=B)
    N = noeud(B, P)
    out["N"] = N
    if N is None:
        out["msg"] = "Branche trop courte pour chercher un noeud"; return out
    if not N["ok"]:
        out["msg"] = f"Pas de noeud net (renflement {N['renfl']:.1f} % < {P['renfl_min_pct']} %)"
    out["ok"] = N["ok"]
    if N["ok"] and etape >= 2:
        out["O"] = oblong(B, N)
    elif N["ok"] and etape >= 3:
        out["O"] = oblong(B, N)
    if N["ok"] and etape >= 3:
        out["Mil"] = point_milieu(B, N, out["O"]) if out.get("O") else None
    if N["ok"] and etape >= 4:
        vr, apps = appendices(R, B, N)
        ell = ellipse_noeud(B, N, P)
        out.update(vrille=vr, apps=apps, ell=ell, peri=perigee(B, N, ell, vr, P))
    out["ms"] = (time.perf_counter() - t0) * 1000
    return out


def mesures(res, calib, etape):
    """Les mesures numériques (en mm si calibré) — utilisées pour l'affichage, la moyenne et le CSV."""
    d = {}
    B, N = res.get("B"), res.get("N")
    if B is None or N is None:
        return d
    d["diametre_base"] = calib.val_diam(N["base"])
    d["bruit_bord"] = calib.val_long(N["bruit_px"]) if np.isfinite(N["bruit_px"]) else float("nan")
    if not N["ok"]:
        return d
    d["diametre_noeud_max"] = calib.val_diam(N["D_pic"])
    d["longueur_noeud"] = calib.val_long(N["longueur"])
    d["renflement_pct"] = N["renfl"]
    d["position_noeud"] = calib.val_long(N["s_pic"] - N["s_debut"])
    if etape >= 2 and res.get("O"):
        O = res["O"]
        d["grand_diametre"] = calib.val_diam(O["D_grand"])
        d["petit_diametre"] = calib.val_diam(O["D_petit"])
        d["oblong_longueur"] = calib.val_long(O["L"])
        d["oblong_hauteur"] = calib.val_diam(O["H"])
    if etape >= 3 and res.get("Mil"):
        M = res["Mil"]
        d["milieu_s"] = calib.val_long(M["M"][0] - N["s_debut"])
        d["milieu_ecart_pic"] = calib.val_long(M["ecart_pic"])
        d["diametre_au_milieu"] = calib.val_diam(M["D_milieu"])
    if etape >= 4 and res.get("ell"):
        E = res["ell"]
        d["ellipse_grand_axe"] = calib.val_diam(2 * E["a"])
        d["ellipse_petit_axe"] = calib.val_diam(2 * E["b"])
        d["ellipse_rms_px"] = E["rms_px"]
        if res.get("peri"):
            Pp = res["peri"]["P"]
            d["perigee_s"] = calib.val_long(Pp[0] - N["s_debut"])
            d["perigee_t"] = calib.val_long(Pp[1] - _interp(B["s"], lissage_gauss(B["centre"], B["sigma"]), Pp[0]))
    return d


# ----------------------------------------------------------------------------------------- dessin
def _poly(img, pts_img, couleur, ep, ferme=False):
    """Trace une polyligne avec une précision de 1/16 de pixel (shift = 4)."""
    p = np.round(np.asarray(pts_img) * 16).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], ferme, couleur, ep, cv2.LINE_AA, shift=4)


def _point(img, p, couleur, r=7):
    q = (int(round(p[0] * 16)), int(round(p[1] * 16)))
    cv2.circle(img, q, r * 16, couleur, -1, cv2.LINE_AA, shift=4)
    cv2.circle(img, q, r * 16, (0, 0, 0), 2, cv2.LINE_AA, shift=4)


def dessiner_base(res, montrer_bords=True):
    """Dessin commun : bords sub-pixel de la tige (vert) + zone et limites du nœud (rouge)."""
    out = res["img"].copy()
    R, B, N = res.get("R"), res.get("B"), res.get("N")
    if R is None or B is None:
        if res.get("contour") is not None:
            cv2.drawContours(out, [res["contour"]], -1, (0, 255, 0), 2)
        return out
    if montrer_bords:
        for cle in ("haut", "bas"):
            ok = np.isfinite(B[cle])
            if ok.sum() > 2:
                _poly(out, vers_image(R, np.column_stack([B["s"][ok], B[cle][ok]])), (0, 255, 0), 2)
    if N is not None and N["ok"]:
        sv = B["s"]
        for s in (N["s_g"], N["s_d"]):                                    # 2 limites parallèles
            th = _interp(sv, B["haut_lisse"], s); tb = _interp(sv, B["bas_lisse"], s)
            ext = 0.15 * B["D0"]
            _poly(out, vers_image(R, [(s, th - ext), (s, tb + ext)]), (0, 0, 255), 3)
    return out


def bandeau(img, lignes, y0=36):
    for i, l in enumerate(lignes):
        y = y0 + 34 * i
        cv2.putText(img, l, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, l, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def graphique(res, calib, w=1200, h=600):
    """Profil de diamètre (brut en gris, lissé en noir) + base, niveau, pic et limites."""
    g = np.full((h, w, 3), 255, np.uint8)
    B, N = res.get("B"), res.get("N")
    if B is None or N is None:
        cv2.putText(g, "Pas de profil", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return g
    sv, Db, L = B["s"], B["D"], B["D_lisse"]
    vals = L[np.isfinite(L)]
    if vals.size == 0:                                                     # profil vide → pas de graphique
        cv2.putText(g, "Pas de profil", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return g
    vmin, vmax = 0.0, float(vals.max()) * 1.12
    X = lambda s: 60 + (s - sv[0]) * (w - 90) / max(1, sv[-1] - sv[0])
    Y = lambda v: h - 50 - (v - vmin) * (h - 100) / max(1e-6, vmax - vmin)
    if N["ok"]:
        cv2.rectangle(g, (int(X(N["s_g"])), 50), (int(X(N["s_d"])), h - 50), (215, 215, 255), -1)
    ok = np.isfinite(Db)
    for s, v in zip(sv[ok][::2], Db[ok][::2]):
        cv2.circle(g, (int(X(s)), int(Y(v))), 1, (170, 170, 170), -1)
    p = np.column_stack([[X(s) for s in sv], [Y(v) for v in L]])
    _poly(g, p, (0, 0, 0), 2)
    for v, c, nom in ((N["base"], (120, 120, 120), "base"), (N["niveau"], (0, 140, 255), "niveau limites")):
        cv2.line(g, (60, int(Y(v))), (w - 30, int(Y(v))), c, 1, cv2.LINE_AA)
        cv2.putText(g, nom, (w - 200, int(Y(v)) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    if N["ok"]:
        for s in (N["s_g"], N["s_d"]):
            cv2.line(g, (int(X(s)), 50), (int(X(s)), h - 50), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(g, (int(X(N["s_pic"])), int(Y(N["D_pic"]))), 6, (255, 0, 255), -1)
    u = calib.unite()
    cv2.putText(g, f"Profil de diametre D(s) - gris: mesure brute sub-pixel, noir: lisse   (axe vertical: px ; "
                   f"{'mm = a*px+b' if calib.ok else 'non calibre'})", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(g, (60, h - 50), (w - 30, h - 50), (0, 0, 0), 1)
    cv2.line(g, (60, 50), (60, h - 50), (0, 0, 0), 1)
    return g


def zoom_noeud(res, img_dessin, taille=600):
    """Vue agrandie autour du nœud (pour vérifier les bords à l'œil)."""
    N, R = res.get("N"), res.get("R")
    if N is None or R is None:
        return np.full((taille, taille, 3), 255, np.uint8)
    B = res["B"]
    c = vers_image(R, [(N["s_pic"], _interp(B["s"], lissage_gauss(B["centre"], B["sigma"]), N["s_pic"]))])[0]
    demi = int(max(N["longueur"], B["D0"]) * 1.1)
    x0, y0 = int(c[0] - demi), int(c[1] - demi)
    H, W = img_dessin.shape[:2]
    x0, y0 = max(0, min(W - 2 * demi, x0)), max(0, min(H - 2 * demi, y0))
    crop = img_dessin[y0:y0 + 2 * demi, x0:x0 + 2 * demi]
    if crop.size == 0:
        return np.full((taille, taille, 3), 255, np.uint8)
    return cv2.resize(crop, (taille, taille), interpolation=cv2.INTER_NEAREST)


# ----------------------------------------------------------------------------------------- application
class Application:
    """
    Fenêtre commune aux 4 étapes : source d'images, curseurs, clavier, moyenne sur N images, enregistrement.
    dessiner(res, calib) et textes(res, calib) sont fournis par chaque script d'étape.
    """
    def __init__(self, etape, titre, dessiner, textes):
        self.etape, self.titre, self.dessiner, self.textes = etape, titre, dessiner, textes
        self.P = dict(DEFAUT)
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        cal = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--calib=")), "calibration.json")
        self.arg = args[0] if args else "0"
        self.calib = Calibration(cal)
        print(f"Calibration : {'OK, ' + f'{self.calib.a:.5f} mm/px' if self.calib.ok else 'ABSENTE → mesures en pixels'}"
              f"{' + distorsion corrigée' if self.calib.K is not None else ''}")
        self.src = ouvrir_source(self.arg, self.P)
        self.fond, self.inverser, self.vue = None, False, 0       # vue : 0 profil, 1 zoom nœud, 2 masque
        self.dossier = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.moy = None                                           # liste de mesures en cours de moyenne
        self.res_moy = None

    def _curseurs(self, fen):
        r = lambda v: None
        for nom, cle, mx in [("Seuil", "seuil", 255), ("Seuil fond", "seuil_fond", 255), ("Flou", "flou", 10),
                             ("Morpho", "morpho", 10), ("Aire min x100", "aire_min", 500),
                             ("Coupe app %", "coupe_app_pct", 95), ("Lissage %", "lissage_pct", 20),
                             ("Bord noeud %", "bord_pct", 90), ("Renfl min %", "renfl_min_pct", 40),
                             ("Contraste min", "contraste_min", 100)]:
            v = self.P[cle] // 100 if cle == "aire_min" else self.P[cle]
            cv2.createTrackbar(nom, fen, int(v), mx, r)
        self.noms = [("Seuil", "seuil"), ("Seuil fond", "seuil_fond"), ("Flou", "flou"), ("Morpho", "morpho"),
                     ("Aire min x100", "aire_min"), ("Coupe app %", "coupe_app_pct"), ("Lissage %", "lissage_pct"),
                     ("Bord noeud %", "bord_pct"), ("Renfl min %", "renfl_min_pct"), ("Contraste min", "contraste_min")]

    def _lire_curseurs(self, fen):
        for nom, cle in self.noms:
            v = cv2.getTrackbarPos(nom, fen)
            self.P[cle] = v * 100 if cle == "aire_min" else v
        self.P["coupe_app_pct"] = max(10, self.P["coupe_app_pct"])
        self.P["bord_pct"] = max(1, self.P["bord_pct"])

    def run(self):
        fen = self.titre + "  (q = quitter)"
        cv2.namedWindow(fen, cv2.WINDOW_NORMAL)
        att = np.full((300, 900, 3), 40, np.uint8)
        cv2.putText(att, "Demarrage...", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imshow(fen, att); cv2.waitKey(1)
        self._curseurs(fen)
        print("Touches : 1-4 méthode · b fond · i inverser · v vue (profil/zoom/masque) · a moyenne · s enregistrer · +/- expo · q quitter")
        try:
            while True:
                img = self.src.lire()
                if img is None:
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break
                    continue
                self._lire_curseurs(fen)
                try:                                                  # une image ratée ne doit JAMAIS arrêter le programme
                    self._traiter(img, fen)
                except Exception as e:
                    nom = f"{type(e).__name__}: {e}"
                    if nom not in self.__dict__.setdefault("_erreurs", set()):   # chaque erreur n'est écrite qu'une fois
                        self._erreurs.add(nom); print("[image ignorée]", nom)
                    aff = img.copy()
                    bandeau(aff, [f"Image ignoree (branche qui bouge / floue) : {type(e).__name__}"])
                    W = 900; Hh = int(aff.shape[0] * W / aff.shape[1])
                    cv2.imshow(fen, np.hstack([cv2.resize(aff, (W, Hh)), np.full((Hh, W, 3), 255, np.uint8)]))
                k = cv2.waitKey(1) & 0xFF
                if not self._touche(k, img):
                    break
        finally:
            self.src.fermer()
            cv2.destroyAllWindows()

    def _traiter(self, img, fen):
        """Analyse + affichage d'UNE image (appelée dans un try : une erreur n'arrête pas le programme)."""
        res = analyser(img, self.P, self.calib, self.etape, self.fond, self.inverser)
        mes = mesures(res, self.calib, self.etape)
        if self.moy is not None:                                      # moyenne en cours
            if res["ok"]:
                self.moy.append(mes)
            if len(self.moy) >= self.P["n_moyenne"]:
                self.res_moy = self._finir_moyenne()
                self.moy = None
        dessin = self.dessiner(res, self.calib)
        lignes = [f"[{self.P['methode']}] {NOMS_METHODES[self.P['methode']]}  {res.get('ms', 0):.0f} ms  "
                  f"{'mm (calibre)' if self.calib.ok else 'PIXELS (non calibre)'}"]
        lignes += self.textes(res, self.calib) if res.get("B") is not None else [res["msg"]]
        if res["msg"] and res.get("B") is not None:
            lignes.append(res["msg"])
        if self.moy is not None:
            lignes.append(f"Moyenne en cours : {len(self.moy)}/{self.P['n_moyenne']}")
        elif self.res_moy:
            lignes += self.res_moy[:4]
        bandeau(dessin, lignes)
        W = 900
        Hh = int(dessin.shape[0] * W / dessin.shape[1])
        if self.vue == 0:
            droite = graphique(res, self.calib, 1200, int(1200 * Hh / W))
        elif self.vue == 1:
            droite = zoom_noeud(res, dessin, 800)
        else:
            m = res.get("masque")
            droite = cv2.cvtColor(m if m is not None else np.zeros(img.shape[:2], np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imshow(fen, np.hstack([cv2.resize(dessin, (W, Hh)), cv2.resize(droite, (W, Hh))]))
        self._dernier = (res, dessin, mes)                            # pour la touche s

    def _touche(self, k, img):
        """Gère une touche. Renvoie False pour quitter."""
        if k == ord('q'):
            return False
        elif k in (ord('1'), ord('2'), ord('3'), ord('4')):
            self.P["methode"] = int(chr(k))
        elif k == ord('i'):
            self.inverser = not self.inverser
        elif k == ord('v'):
            self.vue = (self.vue + 1) % 3
        elif k == ord('b'):
            kk = 2 * int(self.P["flou"]) + 1
            self.fond = cv2.GaussianBlur(self.calib.corriger(img), (kk, kk), 0); self.P["methode"] = 4
            print("Fond mémorisé.")
        elif k in (ord('+'), ord('=')):
            self.src.exposition(1.25); print(f"Exposition {self.src.expo / 1000:.2f} ms")
        elif k == ord('-'):
            self.src.exposition(0.8); print(f"Exposition {self.src.expo / 1000:.2f} ms")
        elif k == ord('a'):
            self.moy, self.res_moy = [], None
            print(f"Moyenne sur {self.P['n_moyenne']} images : ne bouge pas la branche…")
        elif k == ord('s') and getattr(self, "_dernier", None) is not None:
            res, dessin, mes = self._dernier
            self._enregistrer(img, res, dessin, mes)
        return True

    def _finir_moyenne(self):
        cles = sorted(set().union(*[m.keys() for m in self.moy]))
        lignes, txt = [], []
        for c in cles:
            v = np.array([m[c] for m in self.moy if c in m and np.isfinite(m[c])])
            if len(v):
                txt.append(f"{c:22s} = {v.mean():9.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}  (n={len(v)})")
        print("\n=== MOYENNE (valeur ± écart-type = répétabilité) ===\n" + "\n".join(txt))
        for c in ("diametre_noeud_max", "grand_diametre", "petit_diametre", "longueur_noeud"):
            v = np.array([m[c] for m in self.moy if c in m])
            if len(v):
                lignes.append(f"MOY {c}: {v.mean():.3f} +/- {v.std(ddof=1) if len(v) > 1 else 0:.3f}")
        return lignes

    def _enregistrer(self, img, res, dessin, mes):
        st = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(self.dossier, f"etape{self.etape}_{st}")
        cv2.imwrite(base + "_image.png", img)
        cv2.imwrite(base + "_resultat.png", dessin)
        cv2.imwrite(base + "_profil.png", graphique(res, self.calib))
        f = os.path.join(self.dossier, f"mesures_etape{self.etape}.csv")
        neuf = not os.path.exists(f)
        with open(f, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter=";")
            if neuf:
                w.writerow(["date", "unite"] + COLONNES_CSV)
            w.writerow([st, self.calib.unite()] + [f"{mes[c]:.4f}" if c in mes else "" for c in COLONNES_CSV])
        print("Enregistré :", base + "_*.png", "et", f)