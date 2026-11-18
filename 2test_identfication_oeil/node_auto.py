# -*- coding: utf-8 -*-
"""
branche_noeud_auto.py — ÉTAPE 1 (version automatique, un seul fichier)
Détection de la BRANCHE et du NŒUD (limites précises au sous-pixel), SANS aucun curseur à régler :
tous les paramètres sont choisis automatiquement à partir de l'image de la branche.

Lancer (PC Linux, sans venv) :
    python3 branche_noeud_auto.py            → caméra Basler n° 0
    python3 branche_noeud_auto.py 1          → caméra n° 1
    python3 branche_noeud_auto.py photo.png  → image enregistrée (sans caméra)
Touches :
    a  moyenne sur 15 images (valeur ± écart-type dans le terminal)
    s  enregistrer image + résultat + profil (PNG) et une ligne dans mesures_branche_noeud.csv
    v  vue de droite : profil → zoom sur le nœud → masque
    p  afficher / cacher les paramètres choisis automatiquement
    e  refaire le réglage automatique de l'exposition
    q  quitter
Si un fichier calibration.json (créé par calibration_mm.py) est à côté du script : mesures en mm, sinon en pixels.

Ce qui est choisi AUTOMATIQUEMENT à chaque image (voir la fonction choisir_parametres) :
    • le canal d'image où la branche se voit le mieux (gris, bleu, vert, rouge ou « couleur »),
    • le sens (branche sombre sur fond clair ou l'inverse),
    • le flou (selon le bruit mesuré dans l'image),
    • le nettoyage du masque et la suppression de la vrille (selon le diamètre de la branche),
    • le contraste minimal accepté pour un bord (selon le contraste réel fond / branche),
    • le lissage du profil (selon le bruit des bords),
    • le seuil « c'est un nœud » (selon le bruit du profil : le renflement doit dépasser le bruit),
    • l'exposition de la caméra (au démarrage : fond clair sans saturation).
Seule constante de DÉFINITION (pas un réglage) : les limites du nœud sont placées à 20 % de la hauteur
du renflement au-dessus du diamètre de la tige (BORD_NOEUD_PCT), pour mesurer toutes les branches pareil.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORMTHEME", "none")       # évite un plantage Qt/GNOME sur Ubuntu (avant import cv2)
import sys, json, math, time, csv, warnings
import numpy as np
import cv2

# ---------------------------------------------------------------------------------------------------------
# Constantes de DÉFINITION (identiques pour toutes les branches — ce ne sont pas des réglages)
# ---------------------------------------------------------------------------------------------------------
BORD_NOEUD_PCT = 20.0      # limites du nœud : base + 20 % × (pic − base)
COUPE_APP_PCT = 60.0       # appendices plus fins que 60 % du diamètre (vrille, pétiole) ignorés pour trouver la tige
MARGE_BOUTS = 0.8          # bouts de la branche ignorés : 0,8 × diamètre à chaque bout (coupes)
N_MOYENNE = 15             # images moyennées avec la touche « a »
CIBLE_FOND = 205           # exposition auto : le fond doit être vers 205/255 (clair, jamais saturé)


# =========================================================================================================
# 1. CALIBRATION (lecture seule : calibration.json écrit par calibration_mm.py)
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

    def diam(self, px):
        return f"{self.a * px + self.b:.3f} mm" if self.ok else f"{px:.2f} px"

    def long(self, px):
        return f"{self.a * px:.3f} mm" if self.ok else f"{px:.2f} px"

    def vd(self, px):
        return self.a * px + self.b if self.ok else px

    def vl(self, px):
        return self.a * px if self.ok else px

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
        for nom, val in [("ExposureAuto", "Off"), ("GainAuto", "Off"), ("Gain", 0.0), ("ExposureTime", 5000.0),
                         ("AcquisitionFrameRateEnable", True), ("AcquisitionFrameRate", 20.0)]:
            self._regler(nom, val)                                    # tout en manuel : mesures répétables
        self._regler("BalanceWhiteAuto", "Once")
        self.conv = pylon.ImageFormatConverter()
        self.conv.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.expo = 5000.0

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

    def regler_expo(self, us):
        self.expo = float(np.clip(us, 30, 30000))
        self._regler("ExposureTime", self.expo)

    def fermer(self):
        try:
            self.cam.StopGrabbing(); self.cam.Close()
        except Exception:
            pass


class SourceImage:
    def __init__(self, chemin):
        self.img = cv2.imread(chemin, cv2.IMREAD_COLOR)
        if self.img is None:
            raise SystemExit("Image illisible : " + chemin)
        self.expo = None

    def lire(self):
        return self.img.copy()

    def regler_expo(self, us):
        pass

    def fermer(self):
        pass


def exposition_auto(src, fen):
    """
    Règle l'exposition UNE FOIS puis la bloque (une exposition qui change pendant la mesure fausse les bords) :
    on vise un fond clair (98e centile du gris ≈ CIBLE_FOND) sans pixels saturés.
    """
    if src.expo is None:
        return
    for it in range(12):
        img = None
        for _ in range(3):                                            # 3 images : les 2 premières (ancienne exposition) sont jetées
            img, t0 = None, time.time()
            while img is None and time.time() - t0 < 1.0:
                img = src.lire(); cv2.waitKey(1)
        if img is None:
            return
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p98 = float(np.percentile(g, 98))
        sat = float((g >= 254).mean())
        aff = img.copy(); bandeau(aff, [f"Reglage auto de l'exposition... {src.expo / 1000:.2f} ms  fond {p98:.0f}"])
        cv2.imshow(fen, cv2.resize(aff, (900, int(900 * aff.shape[0] / aff.shape[1])))); cv2.waitKey(1)
        if sat > 0.002:                                               # saturé → on divise par 2
            src.regler_expo(src.expo * 0.5); continue
        if abs(p98 - CIBLE_FOND) < 8:
            break
        src.regler_expo(src.expo * CIBLE_FOND / max(p98, 5.0))         # règle de trois (le capteur est linéaire)
    print(f"Exposition réglée et bloquée : {src.expo / 1000:.2f} ms")


# =========================================================================================================
# 3. CHOIX AUTOMATIQUE DES PARAMÈTRES (à partir de l'image de la branche)
# =========================================================================================================
def bruit_image(g):
    """Estimation du bruit (écart-type en niveaux de gris) par la méthode d'Immerkær : rapide et sans réglage."""
    noyau = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = cv2.filter2D(g.astype(np.float32), -1, noyau)
    h, w = g.shape
    return float(math.sqrt(math.pi / 2) * np.abs(r[1:-1, 1:-1]).sum() / (6 * (w - 2) * (h - 2)))


def canaux(img):
    """
    Les 5 « vues » de l'image où la branche peut ressortir : gris, bleu, vert, rouge, et « couleur ».
    « couleur » = projection de chaque pixel (B, G, R) sur la direction qui va de la couleur du fond à la couleur
    de la branche (trouvées par k-moyennes à 2 groupes sur une image réduite). C'est une combinaison LINÉAIRE
    des 3 canaux : le bord reste à mi-chemin entre fond et branche (la saturation HSV, elle, n'est pas linéaire
    et décalerait les bords d'environ 2 px).
    """
    b, g, r = cv2.split(img)
    C = {"gris": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), "bleu": b, "vert": g, "rouge": r}
    p = cv2.resize(img, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, ctr = cv2.kmeans(p, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)       # 2 couleurs dominantes
    d = ctr[1] - ctr[0]
    nd = float(np.linalg.norm(d))
    if nd > 1:
        w = d / nd                                                    # direction fond ↔ branche (longueur 1)
        proj = img.reshape(-1, 3).astype(np.float32) @ w              # projection linéaire
        lo, hi = np.percentile(proj, [0.5, 99.5])
        C["couleur"] = np.clip((proj - lo) * 255.0 / max(hi - lo, 1e-3), 0, 255).reshape(img.shape[:2]).astype(np.uint8)
    return C


def separation(c):
    """
    Seuil d'Otsu sur une version réduite + note de séparation de Fisher J = (m1 − m2)² / (v1 + v2).
    Plus J est grand, mieux la branche se distingue du fond dans ce canal.
    """
    p = cv2.resize(c, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    p = cv2.GaussianBlur(p, (3, 3), 0)
    t, _ = cv2.threshold(p, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    a, b = p[p <= t].astype(float), p[p > t].astype(float)
    if a.size < 50 or b.size < 50:
        return t, 0.0, 0.0, 0.0
    return t, (a.mean() - b.mean()) ** 2 / (a.var() + b.var() + 1e-6), a.mean(), b.mean()


class Auto:
    """Mémorise le canal choisi (avec hystérésis pour ne pas « sauter » d'un canal à l'autre à chaque image)."""
    def __init__(self):
        self.canal = None
        self.P = {}

    def choisir_parametres(self, img):
        C = canaux(img)
        notes = {k: separation(v) for k, v in C.items()}
        meilleur = max(notes, key=lambda k: notes[k][1])
        if self.canal is None or notes[meilleur][1] > 1.25 * notes[self.canal][1]:
            self.canal = meilleur                                     # on ne change que si c'est nettement mieux
        k = self.canal
        g = C[k]
        t, J, m_bas, m_haut = notes[k]
        # --- sens : le fond est ce qui touche le bord de l'image (la branche est au milieu)
        cadre = np.concatenate([g[:8].ravel(), g[-8:].ravel(), g[:, :8].ravel(), g[:, -8:].ravel()])
        fond_clair = np.median(cadre) > t
        # --- bruit → flou
        sig = bruit_image(g)
        flou = 3 if sig < 2 else (5 if sig < 6 else 7)
        P = {"canal": k, "J": J, "fond_clair": bool(fond_clair), "bruit": sig, "flou": flou,
             "contraste": abs(m_haut - m_bas), "contraste_min": max(6.0, 0.3 * abs(m_haut - m_bas))}
        self.P = P
        return g, P


# =========================================================================================================
# 4. SEGMENTATION (masque = simple GUIDE ; la mesure précise est faite sur les gris, partie 6)
# =========================================================================================================
def plus_grande_composante(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m, 0
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return np.where(lab == k, 255, 0).astype(np.uint8), int(st[k, cv2.CC_STAT_AREA])


def segmenter(g, P):
    """Otsu sur le canal choisi, dans le bon sens ; nettoyage proportionnel au diamètre de la branche."""
    f = cv2.GaussianBlur(g, (P["flou"], P["flou"]), 0)
    mode = cv2.THRESH_BINARY_INV if P["fond_clair"] else cv2.THRESH_BINARY      # branche = blanc dans le masque
    _, m = cv2.threshold(f, 0, 255, mode + cv2.THRESH_OTSU)
    m, aire = plus_grande_composante(m)
    if aire < 0.001 * g.size:
        return None, None
    # diamètre approximatif = aire / longueur (rectangle tourné) → taille du nettoyage
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), (w, h), _ = cv2.minAreaRect(max(cs, key=cv2.contourArea))
    D_est = aire / max(w, h, 1)
    t = int(np.clip(0.04 * D_est, 3, 15)) | 1
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, el), cv2.MORPH_CLOSE, el)
    m, aire = plus_grande_composante(m)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [c for c in cs if cv2.contourArea(c) > 0]
    if not cs:
        return None, None
    c = max(cs, key=cv2.contourArea)
    plein = np.zeros_like(m)
    cv2.drawContours(plein, [c], -1, 255, cv2.FILLED)                  # branche remplie (sans trous)
    P["morpho"], P["D_est"] = t, D_est
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


def bords_subpixel(R, t, contraste_min):
    """
    Pour chaque colonne : niveau local = moitié entre le gris du FOND et celui de la BRANCHE (médianes),
    bord = endroit où le gris traverse ce niveau, interpolé entre 2 pixels (≈ 0,05 px). Toutes les colonnes d'un coup.
    """
    g, H = R["g"], R["g"].shape[0]
    col = t > 0
    pres = np.nonzero(col.any(axis=0))[0]
    sub = col[:, pres]
    yt, yb = sub.argmax(axis=0), H - 1 - sub[::-1].argmax(axis=0)
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
    t, D0 = tige(R["m"])
    s, haut, bas = bords_subpixel(R, t, P["contraste_min"])
    if len(s) < 30:
        return None
    ok2 = np.isfinite(haut) & np.isfinite(bas)
    if ok2.sum() < max(30, 0.25 * len(s)):                            # image bougée/floue : trop peu de bords nets
        return None
    centre = 0.5 * (haut + bas)
    sel = ok2.copy()
    cl = np.where(ok2, centre, np.nanmean(centre))
    for _ in range(3):                                                # ligne centrale = parabole robuste
        if sel.sum() < 10:
            break
        pp = np.polyfit(s[sel], centre[sel], 2); cl = np.polyval(pp, s)
        res = centre - cl
        sel = ok2 & (np.abs(res) < max(3 * 1.4826 * np.median(np.abs(res[sel])), 0.5))
    fen = max(5, int(0.15 * D0))
    rh, rb = hampel(cl - haut, fen), hampel(bas - cl, fen)           # rayons haut et bas, nettoyés
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
            "centre": np.where(ok2, centre, cl), "D0": D0, "valides": 100 * np.isfinite(D).mean()}


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


def noeud(B, P):
    sv, L, D0 = B["s"], B["D_lisse"], B["D0"]
    zone = (sv >= sv[0] + MARGE_BOUTS * D0) & (sv <= sv[-1] - MARGE_BOUTS * D0)
    if zone.sum() < 10:
        return None
    iz = np.nonzero(zone)[0]
    i = iz[int(np.argmax(L[iz]))]
    if 0 < i < len(L) - 1:                                            # sommet de la parabole par 3 points
        y0, y1, y2 = L[i - 1], L[i], L[i + 1]
        den = y0 - 2 * y1 + y2
        dx = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if den != 0 else 0.0
        s_pic, D_pic = sv[i] + dx * (sv[i + 1] - sv[i]), y1 - 0.25 * (y0 - y2) * dx
    else:
        s_pic, D_pic = sv[i], L[i]
    base, s_g, s_d = float(np.median(L[iz])), s_pic, s_pic
    for _ in range(3):                                                # base recalculée sans la zone du nœud
        niv = base + BORD_NOEUD_PCT / 100 * (D_pic - base)
        s_g, s_d = _traverse(sv, L, i, niv, -1), _traverse(sv, L, i, niv, +1)
        tg = zone & ~((sv > s_g - 0.3 * D0) & (sv < s_d + 0.3 * D0))
        if tg.sum() > 10:
            base = float(np.median(L[tg]))
    niv = base + BORD_NOEUD_PCT / 100 * (D_pic - base)
    # --- seuil AUTOMATIQUE « c'est un nœud » : le renflement doit dépasser 6 × la variation naturelle de la tige
    tg = zone & ~((sv > s_g - 0.3 * D0) & (sv < s_d + 0.3 * D0))
    var_tige = float(1.4826 * np.median(np.abs(L[tg] - np.median(L[tg])))) if tg.sum() > 10 else 0.0
    seuil = max(0.03 * base, 6 * var_tige)
    P["seuil_noeud_px"] = seuil
    return {"ok": (D_pic - base) >= seuil, "s_pic": float(s_pic), "D_pic": float(D_pic), "base": base,
            "renfl": 100 * (D_pic / base - 1), "niveau": niv, "s_g": s_g, "s_d": s_d, "longueur": s_d - s_g,
            "s_debut": float(sv[0]), "a": float(sv[iz[0]]), "b": float(sv[iz[-1]])}


def analyser(img, auto, calib):
    """Toute la chaîne pour une image. Renvoie un dictionnaire de résultats."""
    t0 = time.perf_counter()
    img = calib.corriger(img)
    g, P = auto.choisir_parametres(img)
    res = {"img": img, "P": P, "ok": False, "msg": ""}
    masque, contour = segmenter(g, P)
    res.update(masque=masque, contour=contour)
    if masque is None:
        res["msg"] = "Aucune branche trouvee"; return res
    R = redresser(g, masque)
    B = profil(R, P) if R is not None else None
    if B is None:
        res["msg"] = "Branche trop petite / contraste trop faible"; return res
    N = noeud(B, P)
    res.update(R=R, B=B, N=N, ms=(time.perf_counter() - t0) * 1000)
    if N is None:
        res["msg"] = "Branche trop courte"
    elif not N["ok"]:
        res["msg"] = f"Pas de noeud net (renflement {N['renfl']:.1f} % = dans le bruit)"
    res["ok"] = N is not None and N["ok"]
    return res


# =========================================================================================================
# 8. AFFICHAGE
# =========================================================================================================
def ligne(img, pts, couleur, ep, ferme=False):
    p = np.round(np.asarray(pts) * 16).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], ferme, couleur, ep, cv2.LINE_AA, shift=4)      # précision 1/16 px


def bandeau(img, lignes):
    for i, l in enumerate(lignes):
        y = 36 + 34 * i
        cv2.putText(img, l, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, l, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


def dessiner(res, calib, montrer_P):
    out = res["img"].copy()
    R, B, N, P = res.get("R"), res.get("B"), res.get("N"), res["P"]
    L = []
    if B is None:
        if res.get("contour") is not None:
            cv2.drawContours(out, [res["contour"]], -1, (0, 255, 0), 2)
        L.append(res["msg"])
    else:
        for cle in ("haut", "bas"):                                   # bords sub-pixel (vert)
            ok = np.isfinite(B[cle])
            ligne(out, vers_image(R, np.column_stack([B["s"][ok], B[cle][ok]])), (0, 255, 0), 2)
        L.append(f"Diametre de la tige : {calib.diam(N['base']) if N else '-'}")
        if N is not None and N["ok"]:
            sv = B["s"]
            for s in (N["s_g"], N["s_d"]):                             # 2 limites du nœud (rouge)
                th, tb = np.interp(s, sv, B["haut_lisse"]), np.interp(s, sv, B["bas_lisse"])
                ligne(out, vers_image(R, [(s, th - 0.15 * B["D0"]), (s, tb + 0.15 * B["D0"])]), (0, 0, 255), 3)
            th, tb = np.interp(N["s_pic"], sv, B["haut_lisse"]), np.interp(N["s_pic"], sv, B["bas_lisse"])
            ligne(out, vers_image(R, [(N["s_pic"], th), (N["s_pic"], tb)]), (255, 0, 255), 2)   # diamètre max (violet)
            L += [f"NOEUD diametre max : {calib.diam(N['D_pic'])}  (+{N['renfl']:.1f} %)",
                  f"NOEUD longueur : {calib.long(N['longueur'])}",
                  f"Centre du noeud a {calib.long(N['s_pic'] - N['s_debut'])} du bout gauche"]
        elif res["msg"]:
            L.append(res["msg"])
        L.append(f"Qualite : bords valides {B['valides']:.0f} %, bruit {P.get('bruit_bord_px', 0):.2f} px  ({res.get('ms', 0):.0f} ms)")
    if montrer_P:
        L.append(f"[auto] canal {P['canal']} (J={P['J']:.1f}), fond {'clair' if P['fond_clair'] else 'sombre'}, "
                 f"bruit image {P['bruit']:.1f}, flou {P['flou']}, morpho {P.get('morpho', '-')}")
        L.append(f"[auto] contraste {P['contraste']:.0f} (min {P['contraste_min']:.0f}), lissage {P.get('lissage_px', 0):.1f} px, "
                 f"seuil noeud {P.get('seuil_noeud_px', 0):.2f} px")
    L.append("Mesures en mm (calibre)" if calib.ok else "Mesures en PIXELS (pas de calibration.json)")
    bandeau(out, L)
    return out


def graphique(res, w=1200, h=600):
    g = np.full((h, w, 3), 255, np.uint8)
    B, N = res.get("B"), res.get("N")
    if B is None or N is None:
        return g
    sv, L = B["s"], B["D_lisse"]
    if not np.isfinite(L).any():                                     # profil vide → graphique blanc
        return g
    vmax = float(np.nanmax(L)) * 1.12
    X = lambda s: 60 + (s - sv[0]) * (w - 90) / max(1, sv[-1] - sv[0])
    Y = lambda v: h - 50 - v * (h - 100) / vmax
    if N["ok"]:
        cv2.rectangle(g, (int(X(N["s_g"])), 50), (int(X(N["s_d"])), h - 50), (215, 215, 255), -1)
    ok = np.isfinite(B["D"])
    for s, v in zip(sv[ok][::2], B["D"][ok][::2]):
        cv2.circle(g, (int(X(s)), int(Y(v))), 1, (170, 170, 170), -1)
    ligne(g, np.column_stack([[X(s) for s in sv], [Y(v) for v in L]]), (0, 0, 0), 2)
    for v, c in ((N["base"], (120, 120, 120)), (N["niveau"], (0, 140, 255))):
        cv2.line(g, (60, int(Y(v))), (w - 30, int(Y(v))), c, 1, cv2.LINE_AA)
    if N["ok"]:
        for s in (N["s_g"], N["s_d"]):
            cv2.line(g, (int(X(s)), 50), (int(X(s)), h - 50), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(g, "Profil de diametre : gris = mesure sub-pixel, noir = lisse, gris clair = base, orange = niveau des limites",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return g


def zoom(res, dessin, taille=800):
    N, R, B = res.get("N"), res.get("R"), res.get("B")
    if N is None:
        return np.full((taille, taille, 3), 255, np.uint8)
    c = vers_image(R, [(N["s_pic"], np.interp(N["s_pic"], B["s"], B["centre"]))])[0]
    d = int(max(N["longueur"], B["D0"]) * 1.1)
    H, W = dessin.shape[:2]
    x0, y0 = int(np.clip(c[0] - d, 0, max(0, W - 2 * d))), int(np.clip(c[1] - d, 0, max(0, H - 2 * d)))
    crop = dessin[y0:y0 + 2 * d, x0:x0 + 2 * d]
    return cv2.resize(crop, (taille, taille), interpolation=cv2.INTER_NEAREST) if crop.size else np.full((taille, taille, 3), 255, np.uint8)


# =========================================================================================================
# 9. PROGRAMME PRINCIPAL
# =========================================================================================================
def main():
    arg = next((a for a in sys.argv[1:] if not a.startswith("--")), "0")
    dossier = os.path.dirname(os.path.abspath(__file__))
    calib = Calibration(os.path.join(dossier, "calibration.json"))
    print("Calibration :", f"OK ({calib.a:.5f} mm/px)" if calib.ok else "absente → mesures en pixels")
    src = SourceBasler(int(arg)) if arg.isdigit() else SourceImage(arg)
    fen = "Branche + noeud (automatique)  -  q quitter"
    cv2.namedWindow(fen, cv2.WINDOW_NORMAL)
    exposition_auto(src, fen)
    auto, vue, montrer_P, moy, erreurs = Auto(), 0, False, None, set()
    print("Touches : a moyenne · s enregistrer · v vue · p paramètres auto · e exposition auto · q quitter")
    try:
        while True:
            img = src.lire()
            if img is None:
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                continue
            try:                                                  # une image ratée (branche qui bouge vite, flou…)
                                                                  # ne doit JAMAIS arrêter le programme
                res = analyser(img, auto, calib)
                if moy is not None and res["ok"]:                         # moyenne en cours (touche a)
                    N = res["N"]
                    moy.append((calib.vd(N["base"]), calib.vd(N["D_pic"]), calib.vl(N["longueur"]), calib.vl(N["s_pic"] - N["s_debut"])))
                    if len(moy) >= N_MOYENNE:
                        a = np.array(moy); sd = a.std(axis=0, ddof=1)
                        u = "mm" if calib.ok else "px"
                        print(f"\n=== MOYENNE sur {len(a)} images ({u}) : valeur ± écart-type (répétabilité) ===")
                        for nom, m_, s_ in zip(("diametre tige", "diametre max noeud", "longueur noeud", "position noeud"), a.mean(0), sd):
                            print(f"  {nom:20s} = {m_:10.4f} ± {s_:.4f}")
                        moy = None
                dessin = dessiner(res, calib, montrer_P)
                if moy is not None:
                    bandeau(dessin, [""] * 9 + [f"Moyenne : {len(moy)}/{N_MOYENNE} - ne bouge pas la branche"])
                W = 900; H = int(dessin.shape[0] * W / dessin.shape[1])
                droite = (graphique(res, 1200, int(1200 * H / W)) if vue == 0 else zoom(res, dessin) if vue == 1
                          else cv2.cvtColor(res["masque"] if res["masque"] is not None else np.zeros(img.shape[:2], np.uint8), cv2.COLOR_GRAY2BGR))
                cv2.imshow(fen, np.hstack([cv2.resize(dessin, (W, H)), cv2.resize(droite, (W, H))]))
            except Exception as e:
                nom = f"{type(e).__name__}: {e}"
                if nom not in erreurs:                            # chaque erreur n'est écrite qu'une fois
                    erreurs.add(nom); print("[image ignorée]", nom)
                aff = img.copy(); bandeau(aff, ["Image ignoree (branche qui bouge / floue)"])
                W = 900; H = int(aff.shape[0] * W / aff.shape[1])
                cv2.imshow(fen, np.hstack([cv2.resize(aff, (W, H)), np.full((H, W, 3), 255, np.uint8)]))
                res, dessin = None, aff
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('v'):
                vue = (vue + 1) % 3
            elif k == ord('p'):
                montrer_P = not montrer_P
            elif k == ord('e'):
                exposition_auto(src, fen)
            elif k == ord('a'):
                moy = []; print(f"Moyenne sur {N_MOYENNE} images…")
            elif k == ord('s'):
                st = time.strftime("%Y%m%d_%H%M%S"); base = os.path.join(dossier, f"branche_noeud_{st}")
                cv2.imwrite(base + "_image.png", img); cv2.imwrite(base + "_resultat.png", dessin)
                cv2.imwrite(base + "_profil.png", graphique(res) if res else dessin)
                f = os.path.join(dossier, "mesures_branche_noeud.csv"); neuf = not os.path.exists(f)
                with open(f, "a", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh, delimiter=";")
                    if neuf:
                        w.writerow(["date", "unite", "diametre_tige", "diametre_max_noeud", "renflement_pct", "longueur_noeud", "position_noeud", "noeud_ok"])
                    N = res.get("N") if res else None
                    if N:
                        w.writerow([st, "mm" if calib.ok else "px", f"{calib.vd(N['base']):.4f}", f"{calib.vd(N['D_pic']):.4f}",
                                    f"{N['renfl']:.2f}", f"{calib.vl(N['longueur']):.4f}", f"{calib.vl(N['s_pic'] - N['s_debut']):.4f}", N["ok"]])
                print("Enregistré :", base + "_*.png  +", f)
    finally:
        src.fermer()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()