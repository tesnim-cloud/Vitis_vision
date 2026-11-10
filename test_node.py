# -*- coding: utf-8 -*-

import os
os.environ.setdefault("QT_QPA_PLATFORMTHEME", "none")   # évite le plantage Qt/GNOME vu sur Ubuntu (à mettre AVANT import cv2)
import sys                                             # lire l'argument (numéro de caméra ou image)
import time                                            # horodatage des fichiers enregistrés
import numpy as np                                     # calcul sur les images (tableaux)
import cv2                                             # traitement d'image

# ------------------------------------------------------------------ réglages
EXPO_US = 8000.0          # exposition de départ (µs) : à ajuster avec + / -
MM_PAR_PX = None          # échelle mm/pixel si connue (ex. 0.047 pour la caméra 1) → affiche les mesures en mm
LARGEUR_AFFICHAGE = 900   # largeur de chaque vue à l'écran (px)
NOMS = {1: "Otsu", 2: "Seuil manuel", 3: "Couleur (saturation)", 4: "Fond appris"}


# ================================================================== SEGMENTATION
def segmenter(img_bgr, methode, p, fond=None, inverser=False):
    """
    Renvoie (masque, contour) :
      masque  = image noir/blanc, blanc (255) = branche
      contour = liste de points du contour extérieur de la plus grosse forme (ou None)
    p = dictionnaire des réglages venant des curseurs (seuil, flou, morpho, aire_min)
    """
    k = p["flou"] * 2 + 1                                       # taille du flou (toujours impaire : 1, 3, 5…)
    flou = cv2.GaussianBlur(img_bgr, (k, k), 0)                 # lisse l'image : enlève le bruit et la texture de l'écorce
    gris = cv2.cvtColor(flou, cv2.COLOR_BGR2GRAY)               # passage en niveaux de gris (0 noir … 255 blanc)

    if methode == 1:                                            # --- Otsu : trouve tout seul le meilleur seuil
        _, m = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # INV : branche sombre → blanc
    elif methode == 2:                                          # --- seuil manuel (curseur)
        _, m = cv2.threshold(gris, p["seuil"], 255, cv2.THRESH_BINARY_INV)
    elif methode == 3:                                          # --- couleur : saturation HSV
        hsv = cv2.cvtColor(flou, cv2.COLOR_BGR2HSV)             # H = teinte, S = saturation, V = luminosité
        s = hsv[:, :, 1]                                        # fond blanc/gris → S faible ; écorce → S plus forte
        _, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:                                                       # --- fond appris : différence avec l'image du fond
        if fond is None:                                        # pas encore de fond mémorisé
            return np.zeros(gris.shape, np.uint8), None
        diff = cv2.absdiff(flou, fond)                          # différence pixel par pixel avec le fond
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)           # une seule valeur de différence par pixel
        _, m = cv2.threshold(diff, p["seuil_fond"], 255, cv2.THRESH_BINARY)

    if inverser:                                                # si la branche ressort en noir → on inverse
        m = cv2.bitwise_not(m)

    # --- nettoyage du masque (morphologie)
    t = p["morpho"] * 2 + 1                                     # taille de l'élément de nettoyage
    noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, noyau)              # OUVERTURE : supprime les petits points blancs (poussière)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, noyau)             # FERMETURE : bouche les petits trous dans la branche

    # --- on garde seulement la plus grosse forme = la branche
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # EXTERNAL = contour extérieur seulement
    contours = [c for c in contours if cv2.contourArea(c) >= p["aire_min"]]      # ignore les formes trop petites
    if not contours:
        return m, None
    c = max(contours, key=cv2.contourArea)                      # la plus grande surface
    propre = np.zeros_like(m)
    cv2.drawContours(propre, [c], -1, 255, thickness=cv2.FILLED)   # masque final : la branche remplie, sans trous
    return propre, c


def mesurer(masque, contour):
    """Mesures simples à partir du contour : longueur, diamètre max, diamètre moyen, aire, angle."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(contour)            # plus petit rectangle tourné qui contient la branche
    longueur = max(w, h)                                        # grand côté = longueur de la branche
    dist = cv2.distanceTransform(masque, cv2.DIST_L2, 5)        # pour chaque pixel : distance au bord le plus proche
    diam_max = 2 * dist.max()                                   # plus gros cercle inscrit × 2 = épaisseur max (souvent le nœud)
    aire = cv2.contourArea(contour)
    diam_moy = aire / longueur if longueur > 0 else 0           # aire / longueur ≈ diamètre moyen d'un cylindre vu de côté
    return {"longueur": longueur, "diam_max": diam_max, "diam_moy": diam_moy, "aire": aire,
            "angle": ang, "rect": ((cx, cy), (w, h), ang)}


def dessiner(img, masque, contour, mes, texte):
    """Image résultat : branche colorée en transparence + contour vert + rectangle + mesures."""
    out = img.copy()
    if contour is not None:
        couleur = np.zeros_like(img); couleur[masque > 0] = (0, 180, 255)   # orange sur la branche
        out = cv2.addWeighted(out, 1.0, couleur, 0.35, 0)                   # mélange transparent
        cv2.drawContours(out, [contour], -1, (0, 255, 0), 3)               # CONTOUR EXTÉRIEUR en vert
        boite = cv2.boxPoints(mes["rect"]).astype(np.int32)                 # 4 coins du rectangle tourné
        cv2.drawContours(out, [boite], -1, (255, 0, 0), 2)                 # rectangle en bleu
    y = 40
    for ligne in texte:                                                     # texte en haut à gauche
        cv2.putText(out, ligne, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(out, ligne, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        y += 40
    return out


def unite(v_px):
    """Affiche une longueur en px, et en mm si l'échelle est connue."""
    return f"{v_px:.0f} px" + (f" = {v_px * MM_PAR_PX:.2f} mm" if MM_PAR_PX else "")


# ================================================================== SOURCE D'IMAGES
class SourceBasler:
    """Petite classe pour lire une caméra Basler avec pypylon (exposition manuelle, couleur ou mono)."""
    def __init__(self, index):
        from pypylon import pylon                                  # importé ici : inutile si on teste une image
        self.pylon = pylon
        infos = pylon.TlFactory.GetInstance().EnumerateDevices()   # liste des caméras
        if index >= len(infos):
            raise SystemExit(f"Caméra n° {index} absente ({len(infos)} trouvée(s)).")
        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(infos[index]))
        self.cam.Open()                                            # ouvre la caméra (ferme pylon Viewer avant !)
        print("Caméra :", self.cam.GetDeviceInfo().GetModelName(), "S/N", self.cam.GetDeviceInfo().GetSerialNumber())
        for nom, val in [("ExposureAuto", "Off"), ("GainAuto", "Off"), ("Gain", 0.0), ("ExposureTime", EXPO_US),
                         ("AcquisitionFrameRateEnable", True), ("AcquisitionFrameRate", 20.0)]:
            try: getattr(self.cam, nom).Value = val                # réglages manuels (mesures répétables)
            except Exception: pass                                 # paramètre absent sur ce modèle → on ignore
        try: self.cam.BalanceWhiteAuto.Value = "Once"              # couleur : balance des blancs une fois
        except Exception: pass
        self.conv = pylon.ImageFormatConverter()                   # conversion vers BGR pour OpenCV
        self.conv.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) # toujours l'image la plus récente
        self.expo = EXPO_US

    def lire(self):
        r = self.cam.RetrieveResult(100, self.pylon.TimeoutHandling_Return)    # attend 100 ms max (la fenêtre reste réactive)
        img = None
        if r is not None and r.IsValid():
            if r.GrabSucceeded():
                img = self.conv.Convert(r).Array.copy()            # image BGR (copie : le buffer est rendu juste après)
            r.Release()
        return img

    def exposition(self, facteur):
        self.expo = float(np.clip(self.expo * facteur, 50, 200000))
        try: self.cam.ExposureTime.Value = self.expo
        except Exception: pass

    def fermer(self):
        self.cam.StopGrabbing(); self.cam.Close()


class SourceImage:
    """Même interface, mais avec une image fixe : pour tester sans caméra."""
    def __init__(self, chemin):
        self.img = cv2.imread(chemin)
        if self.img is None:
            raise SystemExit("Image illisible : " + chemin)
        self.expo = 0
    def lire(self): return self.img.copy()
    def exposition(self, f): pass
    def fermer(self): pass



# ================================================================== NŒUD : profil de largeur
def redresser(masque):
    """Tourne le masque pour mettre l'axe de la branche à l'horizontale.
    Renvoie (masque_redressé, M_inverse) ; M_inverse ramène un point redressé dans l'image d'origine."""
    ys, xs = np.nonzero(masque)                                   # coordonnées de tous les pixels de la branche
    pts = np.column_stack([xs, ys]).astype(np.float32)
    centre = pts.mean(axis=0)                                     # centre de la branche
    cov = np.cov((pts - centre).T)                                # matrice de covariance des points
    vals, vecs = np.linalg.eigh(cov)                              # ACP : vecteur propre de plus grande valeur = axe
    axe = vecs[:, np.argmax(vals)]
    angle = np.degrees(np.arctan2(axe[1], axe[0]))                # angle de l'axe par rapport à l'horizontale
    h, w = masque.shape
    diag = int(np.hypot(h, w)) + 4                                # grande toile : rien n'est coupé en tournant
    M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), angle, 1.0)   # rotation autour du centre
    M[0, 2] += diag / 2 - centre[0]                               # recentre la branche au milieu de la toile
    M[1, 2] += diag / 2 - centre[1]
    rot = cv2.warpAffine(masque, M, (diag, diag), flags=cv2.INTER_NEAREST)
    Minv = cv2.invertAffineTransform(M)                           # transformation inverse (toile → image)
    return rot, Minv


def profil_largeur(rot):
    """Pour chaque colonne x : largeur (nb de pixels entre bord haut et bord bas), bord haut, bord bas."""
    col = rot > 0
    presents = col.any(axis=0)                                    # colonnes qui contiennent de la branche
    haut = np.where(presents, col.argmax(axis=0), 0)              # 1er pixel blanc en partant du haut
    bas = np.where(presents, rot.shape[0] - 1 - col[::-1].argmax(axis=0), 0)   # dernier pixel blanc
    larg = np.where(presents, bas - haut + 1, 0).astype(np.float32)
    return larg, haut, bas


def lisser(v, n):
    """Moyenne glissante sur n points (enlève le « bruit » du bord de l'écorce)."""
    n = max(1, int(n) | 1)                                        # n impair
    return np.convolve(v, np.ones(n) / n, mode="same")


def _trouver_noeud_core(masque, p):
    """
    Cherche le nœud. Renvoie un dictionnaire avec :
      base, pic, x_pic, x_g, x_d (limites dans l'image redressée), renflement (%),
      lignes : 2 segments [(x1,y1),(x2,y2)] = limites du nœud dans l'image d'ORIGINE,
      profil (pour le graphique), ok (True si un nœud net est trouvé)
    """
    rot, Minv = redresser(masque)
    larg0, _, _ = profil_largeur(rot)
    xs = np.nonzero(larg0)[0]
    if len(xs) < 20:
        return None
    D0 = float(np.median(larg0[xs]))                              # diamètre approximatif de la tige

    # enlève les appendices fins (vrille, petits bourgeons) : ouverture avec un disque < diamètre
    k = max(3, int(D0 * p["coupe_app"] / 100) | 1)
    disque = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tige = cv2.morphologyEx(rot, cv2.MORPH_OPEN, disque)
    larg, haut, bas = profil_largeur(tige)
    xs = np.nonzero(larg)[0]
    if len(xs) < 20:
        return None
    x0, x1 = xs[0], xs[-1]                                        # extrémités de la branche
    marge = int(0.8 * D0)                                         # on ignore les bouts (coupes arrondies)
    a, b = x0 + marge, x1 - marge
    if b - a < 10:
        return None

    L = lisser(larg, 0.3 * D0)                                    # profil lissé
    base = float(np.median(L[a:b]))                               # diamètre de base de la tige (médiane)
    x_pic = a + int(np.argmax(L[a:b]))                            # position du plus gros renflement
    pic = float(L[x_pic])
    renfl = 100 * (pic / base - 1)                                # renflement en % du diamètre de base

    niveau = base + p["bord"] / 100 * (pic - base)                # niveau des limites (ex. 30 % de la hauteur)
    x_g = x_pic
    while x_g > a and L[x_g] > niveau:                            # descend vers la gauche jusqu'au niveau
        x_g -= 1
    x_d = x_pic
    while x_d < b and L[x_d] > niveau:                            # descend vers la droite jusqu'au niveau
        x_d += 1

    def vers_origine(x, y):                                       # point redressé → point dans l'image d'origine
        q = Minv @ np.array([x, y, 1.0])
        return int(round(q[0])), int(round(q[1]))
    lignes = [(vers_origine(x, haut[x] - 15), vers_origine(x, bas[x] + 15)) for x in (x_g, x_d)]  # traits de part en part
    pic_seg = (vers_origine(x_pic, haut[x_pic]), vers_origine(x_pic, bas[x_pic]))

    return {"ok": renfl >= p["renfl_min"], "base": base, "pic": pic, "renfl": renfl,
            "x_pic": x_pic, "x_g": x_g, "x_d": x_d, "x0": x0, "x1": x1, "a": a, "b": b,
            "niveau": niveau, "profil": L, "lignes": lignes, "pic_seg": pic_seg,
            "poly": [vers_origine(x_g, haut[x_g]), vers_origine(x_d, haut[x_d]),
                     vers_origine(x_d, bas[x_d]), vers_origine(x_g, bas[x_g])]}


MAX_COTE = 700   # la recherche du nœud se fait sur une image réduite (côté max 700 px) : ~10× plus rapide


def trouver_noeud(masque, p):
    """Version rapide : découpe la zone de la branche, la réduit, cherche le nœud, puis remet tout à l'échelle."""
    x, y, w, h = cv2.boundingRect(masque)                          # rectangle autour de la branche
    if w < 10 or h < 10:
        return None
    m = 20                                                         # petite marge autour
    x0, y0 = max(0, x - m), max(0, y - m)
    sub = masque[y0:y + h + m, x0:x + w + m]                       # on ne garde que la zone utile
    f = min(1.0, MAX_COTE / max(sub.shape))                        # facteur de réduction
    petit = cv2.resize(sub, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) if f < 1 else sub
    n = _trouver_noeud_core(petit, p)
    if n is None:
        return None
    back = lambda pt: (int(round(pt[0] / f + x0)), int(round(pt[1] / f + y0)))   # point réduit → image d'origine
    n["lignes"] = [(back(a), back(b)) for a, b in n["lignes"]]
    n["pic_seg"] = (back(n["pic_seg"][0]), back(n["pic_seg"][1]))
    n["poly"] = [back(q) for q in n["poly"]]
    for k in ("base", "pic", "niveau"):                            # diamètres → pixels de l'image d'origine
        n[k] = n[k] / f
    n["profil"] = n["profil"] / f
    n["px"] = 1 / f                                                # 1 pas du profil = 1/f pixel d'origine
    return n


def dessiner_noeud(out, n):
    """Ajoute sur l'image : zone du nœud en rouge transparent, 2 traits de limite, trait du diamètre max."""
    if n is None or not n["ok"]:
        return out
    calque = out.copy()
    cv2.fillPoly(calque, [np.array(n["poly"], np.int32)], (0, 0, 255))     # zone du nœud
    out = cv2.addWeighted(out, 0.7, calque, 0.3, 0)
    for (p1, p2) in n["lignes"]:
        cv2.line(out, p1, p2, (0, 0, 255), 4)                              # limites du nœud (rouge)
    cv2.line(out, n["pic_seg"][0], n["pic_seg"][1], (255, 0, 255), 2)      # diamètre max au nœud (violet)
    cv2.putText(out, "NOEUD", (n["lignes"][1][0][0] + 10, n["lignes"][1][0][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return out


def graphique_profil(n, w, h):
    """Dessine le profil de largeur D(x) : base (gris), niveau (orange), limites (rouge)."""
    g = np.full((h, w, 3), 255, np.uint8)
    if n is None:
        cv2.putText(g, "Pas de profil", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return g
    L = n["profil"][n["x0"]:n["x1"] + 1]
    if len(L) < 2:
        return g
    vmax = max(L.max(), 1) * 1.15
    X = lambda x: int(30 + (x - n["x0"]) * (w - 60) / max(1, len(L) - 1))   # abscisse → pixel du graphique
    Y = lambda v: int(h - 40 - v * (h - 80) / vmax)                         # largeur → pixel du graphique
    cv2.rectangle(g, (X(n["a"]), 40), (X(n["b"]), h - 40), (235, 235, 235), -1)      # zone analysée
    if n["ok"]:
        cv2.rectangle(g, (X(n["x_g"]), 40), (X(n["x_d"]), h - 40), (210, 210, 255), -1)  # zone du nœud
    pts = np.array([[X(n["x0"] + i), Y(v)] for i, v in enumerate(L)], np.int32)
    cv2.polylines(g, [pts], False, (0, 0, 0), 2)                                      # le profil
    cv2.line(g, (30, Y(n["base"])), (w - 30, Y(n["base"])), (120, 120, 120), 1)       # diamètre de base
    cv2.line(g, (30, Y(n["niveau"])), (w - 30, Y(n["niveau"])), (0, 140, 255), 1)     # niveau des limites
    for x in (n["x_g"], n["x_d"]):
        cv2.line(g, (X(x), 40), (X(x), h - 40), (0, 0, 255), 2)                        # limites
    cv2.putText(g, "Profil de largeur D(x)  -  gris: base  orange: niveau  rouge: limites",
                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return g


# ================================================================== PROGRAMME PRINCIPAL
def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    src = SourceImage(arg) if not arg.isdigit() else SourceBasler(int(arg))   # chiffre = caméra, sinon = fichier

    fen = "Noeud de la branche (q pour quitter)"
    cv2.namedWindow(fen, cv2.WINDOW_NORMAL)
    attente = np.full((300, 900, 3), 40, np.uint8)                 # image d'attente affichée tout de suite
    cv2.putText(attente, "Demarrage camera...", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imshow(fen, attente); cv2.waitKey(1)                       # la fenêtre répond dès le lancement
    rien = lambda v: None
    cv2.createTrackbar("Seuil", fen, 100, 255, rien)          # méthode 2
    cv2.createTrackbar("Seuil fond", fen, 30, 255, rien)      # méthode 4
    cv2.createTrackbar("Flou", fen, 3, 15, rien)              # lissage de l'image
    cv2.createTrackbar("Morpho", fen, 3, 20, rien)            # nettoyage du masque
    cv2.createTrackbar("Aire min x100", fen, 50, 500, rien)   # taille minimale gardée
    cv2.createTrackbar("Bord noeud %", fen, 30, 90, rien)     # niveau des limites : % de la hauteur du renflement
    cv2.createTrackbar("Renfl. min %", fen, 6, 50, rien)      # renflement minimal pour dire « c'est un nœud »
    cv2.createTrackbar("Coupe app. %", fen, 60, 95, rien)     # taille (en % du diamètre) des appendices supprimés

    methode, inverser, fond, voir_masque = 1, False, None, False
    dossier = os.path.dirname(os.path.abspath(__file__))
    print("Touches : 1-4 méthode · b fond · i inverser · m masque/profil · s enregistrer · +/- expo · q quitter")

    try:
        while True:
            img = src.lire()
            if img is None:
                if (cv2.waitKey(1) & 0xFF) == ord('q'):        # garde la fenêtre réactive même sans image
                    break
                continue
            p = {"seuil": cv2.getTrackbarPos("Seuil", fen), "seuil_fond": cv2.getTrackbarPos("Seuil fond", fen),
                 "flou": cv2.getTrackbarPos("Flou", fen), "morpho": cv2.getTrackbarPos("Morpho", fen),
                 "aire_min": cv2.getTrackbarPos("Aire min x100", fen) * 100,
                 "bord": cv2.getTrackbarPos("Bord noeud %", fen),
                 "renfl_min": cv2.getTrackbarPos("Renfl. min %", fen),
                 "coupe_app": max(10, cv2.getTrackbarPos("Coupe app. %", fen))}

            t0 = time.time()
            masque, contour = segmenter(img, methode, p, fond, inverser)          # 1. segmentation
            n = trouver_noeud(masque, p) if contour is not None else None         # 2-6. nœud et limites
            dt = (time.time() - t0) * 1000

            texte = [f"[{methode}] {NOMS[methode]}  {dt:.0f} ms" + ("  (inverse)" if inverser else "")]
            if methode == 4 and fond is None:
                texte.append("Enleve la branche puis appuie sur b")
            elif contour is None or n is None:
                texte.append("Aucune branche trouvee")
            else:
                texte.append(f"Diametre de base : {unite(n['base'])}")
                if n["ok"]:
                    texte += [f"NOEUD : longueur {unite((n['x_d'] - n['x_g']) * n['px'])}",
                              f"Diametre au noeud : {unite(n['pic'])}  (+{n['renfl']:.0f} %)",
                              f"Centre du noeud a {unite((n['x_pic'] - n['x0']) * n['px'])} d'un bout"]
                else:
                    texte.append(f"Pas de noeud net (renflement {n['renfl']:.0f} % < seuil)")
            res = dessiner(img, masque, contour, None, texte) if contour is None else \
                  dessiner(img, masque, contour, mesurer(masque, contour), texte)
            res = dessiner_noeud(res, n)

            W = LARGEUR_AFFICHAGE
            H = int(img.shape[0] * W / img.shape[1])
            droite = cv2.cvtColor(masque, cv2.COLOR_GRAY2BGR) if voir_masque else graphique_profil(n, W * 2, H * 2)
            cv2.imshow(fen, np.hstack([cv2.resize(res, (W, H)), cv2.resize(droite, (W, H))]))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
                methode = int(chr(key))
            elif key == ord('i'):
                inverser = not inverser
            elif key == ord('m'):
                voir_masque = not voir_masque
            elif key == ord('b'):
                k = p["flou"] * 2 + 1
                fond = cv2.GaussianBlur(img, (k, k), 0); methode = 4
                print("Fond mémorisé.")
            elif key in (ord('+'), ord('=')):
                src.exposition(1.5); print(f"Exposition {src.expo/1000:.2f} ms")
            elif key == ord('-'):
                src.exposition(1 / 1.5); print(f"Exposition {src.expo/1000:.2f} ms")
            elif key == ord('s'):
                st = time.strftime("%Y%m%d_%H%M%S")
                for nom, im in [("image", img), ("masque", masque), ("resultat", res),
                                ("profil", graphique_profil(n, 1200, 500))]:
                    f = os.path.join(dossier, f"noeud_{st}_{nom}.png")
                    cv2.imwrite(f, im); print("Enregistré :", f)
    finally:
        src.fermer()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()