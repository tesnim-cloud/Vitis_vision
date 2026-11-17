import os
os.environ.setdefault("QT_QPA_PLATFORMTHEME", "none")  # évite le plantage Qt/GNOME vu sur Ubuntu (à mettre AVANT import cv2)
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
    elif methode == 3:                                           # --- couleur : saturation HSV
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
        r = self.cam.RetrieveResult(2000, self.pylon.TimeoutHandling_Return)   # attend une image (2 s max)
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


# ================================================================== PROGRAMME PRINCIPAL
def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    src = SourceImage(arg) if not arg.isdigit() else SourceBasler(int(arg))   # chiffre = caméra, sinon = fichier

    fen = "Segmentation branche (q pour quitter)"
    cv2.namedWindow(fen, cv2.WINDOW_NORMAL)
    rien = lambda v: None
    cv2.createTrackbar("Seuil", fen, 100, 255, rien)          # méthode 2
    cv2.createTrackbar("Seuil fond", fen, 30, 255, rien)      # méthode 4 : différence minimale avec le fond
    cv2.createTrackbar("Flou", fen, 3, 15, rien)              # lissage
    cv2.createTrackbar("Morpho", fen, 3, 20, rien)            # nettoyage du masque
    cv2.createTrackbar("Aire min x100", fen, 50, 500, rien)   # taille minimale d'une forme gardée (×100 px²)

    methode, inverser, fond = 1, False, None
    dossier = os.path.dirname(os.path.abspath(__file__))
    print("Touches : 1-4 méthode · b apprendre le fond · i inverser · s enregistrer · +/- expo · q quitter")

    try:
        while True:
            img = src.lire()
            if img is None:
                continue
            p = {"seuil": cv2.getTrackbarPos("Seuil", fen), "seuil_fond": cv2.getTrackbarPos("Seuil fond", fen),
                 "flou": cv2.getTrackbarPos("Flou", fen), "morpho": cv2.getTrackbarPos("Morpho", fen),
                 "aire_min": cv2.getTrackbarPos("Aire min x100", fen) * 100}

            t0 = time.time()
            masque, contour = segmenter(img, methode, p, fond, inverser)          # ← le cœur : la segmentation
            dt = (time.time() - t0) * 1000                                        # temps de calcul en ms

            texte = [f"[{methode}] {NOMS[methode]}  {dt:.0f} ms" + ("  (inverse)" if inverser else "")]
            mes = None
            if methode == 4 and fond is None:
                texte.append("Enleve la branche puis appuie sur b")
            elif contour is None:
                texte.append("Aucune branche trouvee")
            else:
                mes = mesurer(masque, contour)
                texte += [f"Longueur : {unite(mes['longueur'])}",
                          f"Diametre max : {unite(mes['diam_max'])}",
                          f"Diametre moyen : {unite(mes['diam_moy'])}",
                          f"Points du contour : {len(contour)}"]
            res = dessiner(img, masque, contour, mes, texte)

            # affichage : résultat à gauche,  masque à droite
            h = int(img.shape[0] * LARGEUR_AFFICHAGE / img.shape[1])
            vue = np.hstack([cv2.resize(res, (LARGEUR_AFFICHAGE, h)),
                             cv2.resize(cv2.cvtColor(masque, cv2.COLOR_GRAY2BGR), (LARGEUR_AFFICHAGE, h))])
            cv2.imshow(fen, vue)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
                methode = int(chr(key))
            elif key == ord('i'):
                inverser = not inverser
            elif key == ord('b'):                                  # mémorise le fond (sans branche !)
                k = p["flou"] * 2 + 1
                fond = cv2.GaussianBlur(img, (k, k), 0)
                methode = 4
                print("Fond mémorisé.")
            elif key in (ord('+'), ord('=')):
                src.exposition(1.5); print(f"Exposition {src.expo/1000:.2f} ms")
            elif key == ord('-'):
                src.exposition(1 / 1.5); print(f"Exposition {src.expo/1000:.2f} ms")
            elif key == ord('s'):                                  # enregistre image brute + masque + résultat
                st = time.strftime("%Y%m%d_%H%M%S")
                for nom, im in [("image", img), ("masque", masque), ("resultat", res)]:
                    f = os.path.join(dossier, f"branche_{st}_{nom}.png")
                    cv2.imwrite(f, im); print("Enregistré :", f)
    finally:
        src.fermer()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()