# -*- coding: utf-8 -*-

import time                                   # pour mesurer les images/s et horodater les fichiers
import os                                     # pour construire le chemin des images enregistrées
import numpy as np                            # tableaux d'images
import cv2                                    # affichage et enregistrement des images
from pypylon import pylon                     # SDK officiel Basler pour Python

EXPO_US = 5000.0                              # exposition de départ en µs (5 ms) : à baisser si l'image est trop claire
FPS_TEST = 30.0                               # cadence limitée pour le test (évite de saturer l'USB avec 2 caméras)
DOSSIER = os.path.dirname(os.path.abspath(__file__))   # les images sont enregistrées à côté du script


def lire(cam, nom, defaut="?"):
    """Lit un paramètre de la caméra sans planter s'il n'existe pas sur ce modèle."""
    try:
        return getattr(cam, nom).Value        # ex. cam.DeviceModelName.Value
    except Exception:
        return defaut                         # paramètre absent → valeur par défaut


def regler(cam, nom, valeur):
    """Écrit un paramètre sans planter s'il n'existe pas ou n'est pas modifiable."""
    try:
        getattr(cam, nom).Value = valeur      # ex. cam.ExposureTime.Value = 5000
        return True
    except Exception as e:
        print(f"   (réglage {nom} ignoré : {e.__class__.__name__})")
        return False


#  1. trouver les caméras
tl = pylon.TlFactory.GetInstance()            # « usine » qui connaît toutes les interfaces (USB, GigE…)
infos = tl.EnumerateDevices()                 # liste des caméras Basler branchées
print(f"\n{len(infos)} caméra(s) Basler trouvée(s) :")
for i, d in enumerate(infos):                 # affiche modèle, numéro de série, interface
    print(f"  [{i}] {d.GetModelName()}  S/N {d.GetSerialNumber()}  ({d.GetDeviceClass()})")
if len(infos) == 0:                           # aucune caméra : on arrête avec un message clair
    raise SystemExit("Aucune caméra. Vérifie le câble USB 3, le port (bleu / SS) et que pylon Viewer la voit.")

n = min(len(infos), 2)                        # on teste au maximum 2 caméras
cams = pylon.InstantCameraArray(n)            # tableau de caméras pour les piloter ensemble
for i in range(n):
    try:
        cams[i].Attach(tl.CreateDevice(infos[i]))  # associe chaque objet à une caméra physique
    except Exception as e:
        if "exclusively opened" in str(e):         # la caméra est déjà ouverte par un autre programme
            raise SystemExit("\nLa caméra %s est déjà utilisée par un autre programme.\n"
                             "→ Ferme pylon Viewer (et pylon IP Configurator), ferme les autres scripts Python,\n"
                             "  ou débranche / rebranche la caméra, puis relance." % infos[i].GetSerialNumber())
        raise

# 2. ouvrir et régler
cams.Open()                                   # ouvre la connexion avec toutes les caméras
for i in range(n):
    c = cams[i]
    print(f"\nCaméra {i} : {lire(c, 'DeviceModelName')}  S/N {lire(c, 'DeviceSerialNumber')}")
    print(f"   Vitesse USB : {lire(c, 'BslUSBSpeedMode')}  (il faut « SuperSpeed », sinon câble/port USB 2)")
    print(f"   Capteur : {lire(c, 'SensorWidth')} x {lire(c, 'SensorHeight')} px")
    couleur = "c" in str(lire(c, "DeviceModelName"))[-6:]   # a2A1920-160ucBAS → « uc » = caméra couleur
    print(f"   Caméra {'COULEUR' if couleur else 'MONO'}")
    if not couleur:
        regler(c, "PixelFormat", "Mono8")     # mono : niveaux de gris 8 bits
    else:
        regler(c, "PixelFormat", "BayerRG8")  # couleur : image brute Bayer, convertie en couleur par pylon
        regler(c, "BalanceWhiteAuto", "Once") # balance des blancs automatique une fois
    regler(c, "ExposureAuto", "Off")          # exposition manuelle : obligatoire pour des mesures répétables
    regler(c, "GainAuto", "Off")              # gain manuel
    regler(c, "Gain", 0.0)                    # gain à 0 dB (le moins de bruit)
    regler(c, "ExposureTime", EXPO_US)        # exposition de départ
    regler(c, "AcquisitionFrameRateEnable", True)   # active la limite de cadence
    regler(c, "AcquisitionFrameRate", FPS_TEST)     # 30 i/s pour le test
    regler(c, "TriggerSelector", "FrameStart")      # le déclenchement porte sur chaque image
    regler(c, "TriggerMode", "Off")                 # départ en mode continu

conv = pylon.ImageFormatConverter()           # convertit toute image (Bayer, Mono…) en BGR pour OpenCV
conv.OutputPixelFormat = pylon.PixelType_BGR8packed

mode_trigger = False                          # False = flux continu, True = 1 image par appui sur ESPACE
cams.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)   # garde seulement l'image la plus récente (pas de retard)
derniere = [None] * n                         # dernière image reçue de chaque caméra
compte = [0] * n                              # compteur d'images pour calculer les i/s
t0 = time.time()
fps = [0.0] * n
expo = EXPO_US
print("\nTouches : q quitter · s enregistrer · +/- exposition · t mode déclenchement · ESPACE photo\n")

# 3. boucle d'acquisition
try:
    while cams.IsGrabbing():
        try:
            res = cams.RetrieveResult(100, pylon.TimeoutHandling_Return)   # attend 100 ms max une image
        except pylon.RuntimeException:
            res = None
        if res is not None and res.IsValid() and res.GrabSucceeded():
            k = res.GetCameraContext()        # numéro de la caméra qui a envoyé cette image
            img = conv.Convert(res).Array     # image couleur BGR (ou gris recopié sur 3 canaux)
            derniere[k] = img.copy()          # copie, car le buffer est rendu à la caméra juste après
            compte[k] += 1
        elif res is not None and res.IsValid() and not res.GrabSucceeded():
            print("Image perdue :", res.GetErrorDescription())   # utile pour voir un problème de bande passante USB
        if res is not None and res.IsValid():
            res.Release()                     # rend le buffer à pylon

        if time.time() - t0 >= 1.0:           # toutes les secondes : calcul des images/s
            fps = [c / (time.time() - t0) for c in compte]
            compte = [0] * n
            t0 = time.time()

        # --- affichage côte à côte (réduit pour tenir à l'écran)
        vues = []
        for k in range(n):
            if derniere[k] is None:
                vues.append(np.zeros((400, 640, 3), np.uint8))       # écran noir tant que rien n'est reçu
                continue
            v = cv2.resize(derniere[k], (640, 400))                   # 1920x1200 → 640x400
            txt = f"Cam {k}  {fps[k]:.0f} i/s  expo {expo/1000:.1f} ms" + ("  [TRIGGER]" if mode_trigger else "")
            cv2.putText(v, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            vues.append(v)
        h = max(v.shape[0] for v in vues)
        vues = [cv2.copyMakeBorder(v, 0, h - v.shape[0], 0, 0, cv2.BORDER_CONSTANT) for v in vues]
        cv2.imshow("Test Basler (q pour quitter)", np.hstack(vues))

        # --- clavier
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):                 # enregistre l'image pleine résolution de chaque caméra
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for k in range(n):
                if derniere[k] is not None:
                    f = os.path.join(DOSSIER, f"cam{k}_{stamp}.png")
                    cv2.imwrite(f, derniere[k]); print("Enregistré :", f)
        elif key in (ord('+'), ord('-')):     # change l'exposition des 2 caméras
            expo = expo * 1.5 if key == ord('+') else expo / 1.5
            expo = float(np.clip(expo, 50, 100000))    # entre 50 µs et 100 ms
            for k in range(n):
                regler(cams[k], "ExposureTime", expo)
        elif key == ord('t'):                 # bascule continu ↔ déclenchement logiciel
            mode_trigger = not mode_trigger
            cams.StopGrabbing()               # il faut arrêter l'acquisition pour changer le mode
            for k in range(n):
                regler(cams[k], "TriggerMode", "On" if mode_trigger else "Off")
                regler(cams[k], "TriggerSource", "Software")
            cams.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print("Mode déclenchement logiciel" if mode_trigger else "Mode continu")
        elif key == ord(' ') and mode_trigger:  # 1 photo sur chaque caméra (comme dans la machine)
            for k in range(n):
                try:
                    cams[k].ExecuteSoftwareTrigger()
                except Exception as e:
                    print("Trigger impossible :", e)
finally:
    cams.StopGrabbing()                       # arrêt propre même en cas d'erreur
    cams.Close()
    cv2.destroyAllWindows()
    print("Caméras fermées.")