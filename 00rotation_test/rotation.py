#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rotation_360_test.py — Vitis Optima, test du balayage 360° + retour à l'angle utile
=================================================================================
Idée :
  1) On tourne la branche sur 360° (à la main pour l'instant) pendant que la caméra 1 prend des photos.
  2) Pour chaque photo on mesure "à quel point le bourgeon / la vrille dépasse" de la tige (profil).
  3) Le meilleur angle = celui où le bourgeon dépasse le plus, du côté voulu (haut ou bas de l'image).
  4) On ramène la branche à : meilleur angle + OFFSET_VERIN_DEG (angle entre l'axe caméra et le vérin de coupe).
     En mode manuel : la caméra te guide en live jusqu'au pic (STOP en vert), puis tu ajoutes l'offset à la main.
     Avec le moteur (plus tard) : retour = nombre de pas calculé, pas besoin de guidage.

Sources possibles :  --source basler (défaut) | webcam | dossier (images déjà prises, nom = angle, ex. 045.png)
Modes de balayage :  --mode pas (Entrée à chaque pas, défaut 30°) | continu (tu tournes 1 tour régulier, Entrée pour stop)

Exemples :
  python3 rotation_360_test.py                       # Basler, pas de 30°
  python3 rotation_360_test.py --pas 15              # pas de 15° (24 photos)
  python3 rotation_360_test.py --mode continu        # tour continu à la main
  python3 rotation_360_test.py --source dossier --dossier ./images_test --sans-retour
"""
import argparse                       # lecture des options de la ligne de commande
import csv                            # sauvegarde des résultats en CSV
import os                             # chemins / dossiers
import sys                            # sortie propre
import threading                      # thread pour détecter "Entrée" en mode continu
import time                           # chrono (mode continu)

import cv2                            # traitement d'image
import numpy as np                    # calcul

# ---------------------------------------------------------------------------------------------
# PARAMÈTRES (à ajuster pendant les essais)
# ---------------------------------------------------------------------------------------------
OFFSET_VERIN_DEG = 0.0      # angle (°) entre "bourgeon vu de profil" et "bourgeon face au vérin" -> à mesurer sur la machine
COTE_VOULU = "haut"         # le bourgeon doit dépasser vers le "haut" ou le "bas" de l'image (lève l'ambiguïté θ / θ+180°)
BRANCHE_SOMBRE = True       # True = branche plus sombre que le fond (rétro-éclairage) ; False = branche claire sur fond noir
SEUIL_PIC = 0.95            # retour manuel : STOP quand le bourgeon dépasse à >= 95 % du max vu pendant le balayage
MARGE_BORDS = 0.05          # on ignore 5 % de la longueur de la branche à chaque bout (bords de l'image, coupe)
DOSSIER_SORTIE = "00rotation_test/resultats_360"  # où on écrit CSV + images annotées


# =============================================================================================
# 1) CAMÉRA : une seule interface .grab() -> image BGR, quelle que soit la source
# ====================================
# =========================================================
class CameraBasler:
    """Caméra Basler via pypylon (couleur ou mono)."""
    def __init__(self, exposition_us=None):
        from pypylon import pylon                                          # import ici : pas obligatoire pour les autres sources
        self.pylon = pylon                                                 # garde le module
        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())  # 1re caméra trouvée
        self.cam.Open()                                                    # ouvre la caméra
        if exposition_us is not None:                                      # exposition imposée ?
            self.cam.ExposureTime.Value = float(exposition_us)             # en µs
        self.conv = pylon.ImageFormatConverter()                           # convertisseur -> BGR OpenCV
        self.conv.OutputPixelFormat = pylon.PixelType_BGR8packed           # sortie BGR 8 bits
        self.conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)         # on garde toujours la dernière image

    def grab(self):
        res = self.cam.RetrieveResult(2000, self.pylon.TimeoutHandling_ThrowException)  # attend max 2 s
        try:
            if not res.GrabSucceeded():                                    # image ratée
                return None
            return self.conv.Convert(res).GetArray().copy()                # numpy BGR
        finally:
            res.Release()                                                  # libère le buffer pylon

    def close(self):
        self.cam.StopGrabbing(); self.cam.Close()                          # arrêt propre


class CameraWebcam:
    """Webcam USB (pour tester sans Basler)."""
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)                                 # ouvre la webcam
        if not self.cap.isOpened():
            raise RuntimeError("Webcam introuvable")

    def grab(self):
        ok, img = self.cap.read()                                          # lit une image
        return img if ok else None

    def close(self):
        self.cap.release()


class CameraDossier:
    """Images déjà prises ; le nom du fichier = angle (ex. 000.png, 009.png, 018.png...)."""
    def __init__(self, dossier):
        ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        fichiers = [f for f in os.listdir(dossier) if f.lower().endswith(ext)]
        self.liste = sorted(((float(os.path.splitext(f)[0]), os.path.join(dossier, f)) for f in fichiers))  # (angle, chemin)
        if not self.liste:
            raise RuntimeError(f"Aucune image dans {dossier}")
        self.i = 0                                                         # index courant

    def grab(self):
        if self.i >= len(self.liste):                                      # plus d'images
            return None
        img = cv2.imread(self.liste[self.i][1]); self.i += 1               # lit l'image suivante
        return img

    def close(self):
        pass


# =============================================================================================
# 2) MESURE sur une image : combien le bourgeon dépasse, et de quel côté
# =============================================================================================
def mesurer_branche(img):
    """
    Retourne dict : depassement_px (>=0), cote ('haut'/'bas'), diametre_tige_px, x_noeud (px, image redressée),
                    angle_axe_deg, ok (bool)
    Méthode simple et robuste pour un 1er test (pas la précision sub-pixel de vitis_core) :
      Otsu -> plus grand blob = branche -> axe par ACP -> image redressée (branche horizontale)
      -> pour chaque colonne : bord haut, bord bas -> écarts au bord "normal" de la tige (médiane)
      -> le plus gros écart = bourgeon/vrille en profil.
    """
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img     # en niveaux de gris
    gris = cv2.GaussianBlur(gris, (5, 5), 0)                                   # enlève le bruit
    mode = cv2.THRESH_BINARY_INV if BRANCHE_SOMBRE else cv2.THRESH_BINARY      # branche = blanc dans le masque
    _, masque = cv2.threshold(gris, 0, 255, mode + cv2.THRESH_OTSU)            # seuil automatique Otsu
    masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # enlève les poussières

    n, lab, stats, _ = cv2.connectedComponentsWithStats(masque)                # blobs
    if n < 2:                                                                  # rien trouvé
        return {"ok": False}
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))                        # plus grand blob (0 = fond)
    masque = np.where(lab == k, 255, 0).astype(np.uint8)                       # garde seulement la branche

    ys, xs = np.nonzero(masque)                                                # pixels de la branche
    pts = np.column_stack([xs, ys]).astype(np.float64)                         # (x, y)
    centre = pts.mean(axis=0)                                                  # centre de gravité
    _, vecs = np.linalg.eigh(np.cov((pts - centre).T))                         # ACP : vecteurs propres
    axe = vecs[:, 1]                                                           # plus grande valeur propre = axe de la branche
    if axe[0] < 0:                                                             # le signe d'un vecteur propre est arbitraire :
        axe = -axe                                                             # on force l'axe vers la droite, sinon haut/bas s'inversent
    angle_axe = np.degrees(np.arctan2(axe[1], axe[0]))                         # angle de l'axe dans l'image

    H, W = masque.shape                                                        # taille image
    M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), angle_axe, 1.0)  # rotation -> branche horizontale
    diag = int(np.hypot(H, W))                                                 # grande toile pour ne rien couper
    M[0, 2] += diag / 2 - centre[0]; M[1, 2] += diag / 2 - centre[1]           # recentre dans la toile
    redresse = cv2.warpAffine(masque, M, (diag, diag), flags=cv2.INTER_NEAREST)  # masque redressé

    cols = np.nonzero(redresse.any(axis=0))[0]                                 # colonnes qui contiennent la branche
    if cols.size < 20:
        return {"ok": False}
    marge = int(MARGE_BORDS * cols.size)                                       # on évite les bouts
    cols = cols[marge: cols.size - marge]                                      # colonnes utiles
    haut = np.array([np.argmax(redresse[:, c] > 0) for c in cols], float)      # 1er pixel blanc depuis le haut
    bas = np.array([diag - 1 - np.argmax(redresse[::-1, c] > 0) for c in cols], float)  # 1er depuis le bas

    haut_ref = np.median(haut); bas_ref = np.median(bas)                       # bords "normaux" de la tige
    d_haut = haut_ref - haut                                                   # >0 = ça dépasse vers le haut
    d_bas = bas - bas_ref                                                      # >0 = ça dépasse vers le bas
    d_haut = np.convolve(d_haut, np.ones(5) / 5, mode="same")                  # petit lissage
    d_bas = np.convolve(d_bas, np.ones(5) / 5, mode="same")

    ih, ib = int(np.argmax(d_haut)), int(np.argmax(d_bas))                     # pics haut et bas
    if d_haut[ih] >= d_bas[ib]:                                                # le plus gros dépassement gagne
        dep, cote, xn = d_haut[ih], "haut", cols[ih]
    else:
        dep, cote, xn = d_bas[ib], "bas", cols[ib]

    return {"ok": True,
            "depassement_px": float(max(dep, 0.0)),
            "dep_haut_px": float(max(d_haut[ih], 0.0)),                        # utile pour le choix du côté
            "dep_bas_px": float(max(d_bas[ib], 0.0)),
            "cote": cote,
            "diametre_tige_px": float(bas_ref - haut_ref),
            "x_noeud": int(xn),
            "angle_axe_deg": float(angle_axe)}


# =============================================================================================
# 3) BALAYAGE 360°
# =============================================================================================
def balayage_pas(cam, pas_deg):
    """Tu tournes de pas_deg à la main, Entrée, photo. Retourne [(angle, image, mesure)]."""
    res = []
    n = int(round(360 / pas_deg))                                              # nombre de photos
    if isinstance(cam, CameraDossier):                                         # dossier : on lit TOUTES les images
        n = len(cam.liste)
    for i in range(n):
        a = i * pas_deg                                                        # angle théorique
        if not isinstance(cam, CameraDossier):                                 # en mode dossier pas d'attente
            input(f"  -> Mets la branche à {a:6.1f}°  puis Entrée ")
        img = cam.grab()
        if img is None:
            print("  image ratée / plus d'images, arrêt du balayage"); break
        if isinstance(cam, CameraDossier):
            a = cam.liste[cam.i - 1][0]                                        # angle lu dans le nom du fichier
        m = mesurer_branche(img)
        res.append((a, img, m))
        print(f"     {a:6.1f}°  dépassement={m.get('depassement_px', float('nan')):6.1f} px  côté={m.get('cote', '?')}")
    return res


def balayage_continu(cam):
    """Tu fais 1 tour à vitesse régulière ; angle = temps écoulé / temps total × 360."""
    input("  Prêt ? Entrée = je commence à tourner (fais UN tour régulier) ")
    stop = threading.Event()                                                   # drapeau d'arrêt
    threading.Thread(target=lambda: (input("  ... Entrée quand le tour est fini "), stop.set()), daemon=True).start()
    t0 = time.time(); brut = []                                                # (temps, image)
    while not stop.is_set():
        img = cam.grab()
        if img is not None:
            brut.append((time.time() - t0, img))                               # garde l'heure de chaque image
    T = time.time() - t0                                                       # durée du tour
    res = []
    for t, img in brut:
        a = 360.0 * t / T                                                      # angle estimé (vitesse supposée constante)
        res.append((a, img, mesurer_branche(img)))
    print(f"  {len(res)} images en {T:.1f} s")
    return res


def trouver_meilleur_angle(res):
    """Angle où le bourgeon dépasse le plus du COTE_VOULU, affiné par parabole (sous le pas)."""
    angles = np.array([a for a, _, m in res if m["ok"]])                       # angles valides
    cle = "dep_haut_px" if COTE_VOULU == "haut" else "dep_bas_px"              # on note seulement le côté voulu
    scores = np.array([m[cle] for _, _, m in res if m["ok"]])
    if scores.size < 3:
        raise RuntimeError("Pas assez d'images valides")
    i = int(np.argmax(scores))                                                 # meilleur échantillon
    n = len(scores)
    y0, y1, y2 = scores[(i - 1) % n], scores[i], scores[(i + 1) % n]           # voisins (circulaire : 360° = 0°)
    pas = np.median(np.diff(np.sort(angles))) if n > 1 else 0.0                # pas moyen
    den = (y0 - 2 * y1 + y2)
    decal = 0.5 * (y0 - y2) / den if den != 0 else 0.0                         # sommet de la parabole (-0.5..0.5)
    decal = float(np.clip(decal, -0.5, 0.5))
    meilleur = (angles[i] + decal * pas) % 360.0                               # angle affiné
    return meilleur, angles, scores


# =============================================================================================
# 4) RETOUR à l'angle cible (manuel guidé par la caméra, ou moteur plus tard)
# =============================================================================================
def retour_manuel(cam, res, meilleur, angle_cible, affichage=True):
    """
    Retour en 2 temps (test à la main) :
      a) on te guide en live jusqu'au PIC (bourgeon de profil, côté voulu) : repère visuel net
      b) puis tu ajoutes OFFSET_VERIN_DEG à la main (rapporteur) -> bourgeon face au vérin
    Pourquoi pas directement la cible ? Si la cible tombe là où le bourgeon est caché (devant/derrière la tige),
    toutes ces positions se ressemblent -> la caméra ne peut pas guider. Avec le moteur, le (b) sera juste un nombre de pas.
    """
    cle = "dep_haut_px" if COTE_VOULU == "haut" else "dep_bas_px"              # côté qui compte
    autre = "dep_bas_px" if COTE_VOULU == "haut" else "dep_haut_px"
    vmax = max(m[cle] for _, _, m in res if m["ok"])                           # dépassement max vu au balayage
    delta = ((meilleur + 180.0) % 360.0) - 180.0                               # plus court chemin 0° -> pic
    sens = "dans le MÊME sens que le balayage" if delta >= 0 else "dans le sens INVERSE"
    print(f"\n  a) Tourne d'environ {abs(delta):.0f}° {sens} -> pic attendu à {meilleur:.1f}° "
          f"(bourgeon {COTE_VOULU}, max {vmax:.1f} px)")
    print("     Tourne doucement, STOP quand c'est vert. 'q' (ou Ctrl+C) quand tu es dessus.")

    try:
        while True:
            img = cam.grab()
            if img is None:
                break
            m = mesurer_branche(img)                                           # mesure live
            v = m.get(cle, 0.0) if m["ok"] else 0.0                            # dépassement côté voulu
            ok = m["ok"] and v >= SEUIL_PIC * vmax and v > m.get(autre, 0.0)   # proche du max ET bon côté
            txt = f"{COTE_VOULU} {v:5.1f}/{vmax:.1f} px ({100 * v / max(vmax, 1e-6):3.0f} %)  " + \
                  ("STOP !" if ok else "tourne...")
            if affichage:
                vue = img.copy()
                coul = (0, 200, 0) if ok else (0, 0, 255)                      # vert = STOP, rouge = continue
                cv2.putText(vue, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, coul, 3)
                cv2.imshow("retour", cv2.resize(vue, None, fx=0.5, fy=0.5))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            else:
                print("\r  " + txt + "   ", end="", flush=True)
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    if affichage:
        cv2.destroyAllWindows()
    if OFFSET_VERIN_DEG != 0:
        print(f"\n  b) Maintenant tourne de {OFFSET_VERIN_DEG:+.1f}° (même sens que le balayage si +) "
              f"-> bourgeon face au vérin ({angle_cible:.1f}°)")
    else:
        print("\n  b) Offset = 0 -> tu es déjà en face du vérin.")


class MoteurPi:
    """PLUS TARD : envoie la rotation au Raspberry Pi (TCP) qui pilote le DM860H. Pas utilisé en test manuel."""
    def __init__(self, ip="192.168.1.50", port=5000):
        self.ip, self.port = ip, port

    def tourner(self, deg):
        import socket
        with socket.create_connection((self.ip, self.port), timeout=5) as s:   # connexion au Pi
            s.sendall(f"ROT {deg:.2f}\n".encode())                             # protocole à définir avec le code Pi
            return s.recv(64).decode().strip()                                 # réponse "OK" attendue


# =============================================================================================
# 5) SAUVEGARDE
# =============================================================================================
def sauvegarder(res, meilleur, angle_cible):
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    with open(os.path.join(DOSSIER_SORTIE, "balayage.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["angle_deg", "ok", "dep_haut_px", "dep_bas_px", "cote", "diametre_tige_px"])
        for a, _, m in res:
            w.writerow([f"{a:.2f}", m["ok"], m.get("dep_haut_px", ""), m.get("dep_bas_px", ""),
                        m.get("cote", ""), m.get("diametre_tige_px", "")])
    i = int(np.argmin([abs(((a - meilleur + 180) % 360) - 180) for a, _, _ in res]))  # photo la plus proche du meilleur
    vue = res[i][1].copy()
    cv2.putText(vue, f"meilleur {meilleur:.1f} deg -> cible verin {angle_cible:.1f} deg", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
    cv2.imwrite(os.path.join(DOSSIER_SORTIE, "meilleur_angle.png"), vue)
    try:                                                                       # courbe si matplotlib est là
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        a = [r[0] for r in res if r[2]["ok"]]
        plt.figure(figsize=(8, 4))
        plt.plot(a, [r[2]["dep_haut_px"] for r in res if r[2]["ok"]], "o-", label="dépasse en haut")
        plt.plot(a, [r[2]["dep_bas_px"] for r in res if r[2]["ok"]], "s--", label="dépasse en bas")
        plt.axvline(meilleur, color="g", label=f"meilleur {meilleur:.1f}°")
        plt.xlabel("angle (°)"); plt.ylabel("dépassement (px)"); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(DOSSIER_SORTIE, "courbe_360.png"), dpi=120)
    except ImportError:
        pass
    print(f"  Résultats dans ./{DOSSIER_SORTIE}/")


# =============================================================================================
# MAIN
# =============================================================================================
def main():
    global OFFSET_VERIN_DEG, COTE_VOULU, BRANCHE_SOMBRE
    p = argparse.ArgumentParser(description="Test balayage 360° + retour à l'angle utile")
    p.add_argument("--source", choices=["basler", "webcam", "dossier"], default="basler")
    p.add_argument("--dossier", default="images_test", help="pour --source dossier")
    p.add_argument("--mode", choices=["pas", "continu"], default="pas")
    p.add_argument("--pas", type=float, default=30.0, help="pas en degrés (mode pas)")
    p.add_argument("--offset", type=float, default=OFFSET_VERIN_DEG, help="angle caméra -> vérin (°)")
    p.add_argument("--cote", choices=["haut", "bas"], default=COTE_VOULU)
    p.add_argument("--branche-claire", action="store_true", help="branche claire sur fond sombre")
    p.add_argument("--exposition", type=float, default=None, help="exposition Basler en µs")
    p.add_argument("--sans-retour", action="store_true", help="juste le balayage + calcul")
    p.add_argument("--sans-affichage", action="store_true")
    o = p.parse_args()
    OFFSET_VERIN_DEG, COTE_VOULU, BRANCHE_SOMBRE = o.offset, o.cote, not o.branche_claire

    if o.source == "basler":
        cam = CameraBasler(o.exposition)
    elif o.source == "webcam":
        cam = CameraWebcam()
    else:
        cam = CameraDossier(o.dossier)

    try:
        print("\n=== 1) BALAYAGE 360° ===")
        res = balayage_continu(cam) if (o.mode == "continu" and o.source != "dossier") else balayage_pas(cam, o.pas)

        print("\n=== 2) MEILLEUR ANGLE ===")
        meilleur, _, _ = trouver_meilleur_angle(res)
        cible = (meilleur + OFFSET_VERIN_DEG) % 360.0                          # là où le bourgeon fait face au vérin
        print(f"  Bourgeon de profil ({COTE_VOULU}) à {meilleur:.1f}°  ->  cible vérin = {cible:.1f}°")
        sauvegarder(res, meilleur, cible)

        if not o.sans_retour and o.source != "dossier":
            print("\n=== 3) RETOUR ===")
            input("  Termine le tour (branche revenue à 0°) puis Entrée ")
            retour_manuel(cam, res, meilleur, cible, affichage=not o.sans_affichage)
    finally:
        cam.close()


if __name__ == "__main__":
    main()