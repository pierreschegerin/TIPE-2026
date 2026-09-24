import cv2
import numpy as np
import serial
import time

# Parametres

port = 'COM8' # port de communication 
camera = 1 # index de la lunette de guidage (Camera SVBony SV105)

T = 0.5 # periode d'echantillonnage

Kp = 80 # gain proportionnel
Ki = 30 # gain integral

integraleMax = 500 # Anti-windup : saturation de la somme integrale
freqMax = 2000 # frequence maximale d'impulsion moteur admissible (Hz)

sensMoteur = +1 # +1 ou -1 selon que le moteur compense vers la droite ou la gauche de l'écran


# Connexion carte

try:
    carte = serial.Serial(port, 115200, timeout=1)
    time.sleep(2)
    connexion = True

except Exception as erreur:
    print(f"Mode simulation : {erreur}")
    connexion = False





# Camera

cap = cv2.VideoCapture(camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def envoyerCommande(freq):
    if not connexion:
        return
    message = f"SET_RA:{int(freq)}\n"
    carte.write(message.encode())
    carte.readline()

# Detection de l'etoile

def trouverBarycentre(image):

    # Passage en niveaux de gris
    b, g, r = cv2.split(image.astype(np.float32))
    gris = 0.229*r + 0.587*g + 0.114*b
    gris = np.clip(gris, 0, 255).astype(np.uint8)

    # Reduction du bruit
    grisFiltre = cv2.GaussianBlur(gris, (5, 5), 0)

    # Calcul du seuil
    moyenne = np.mean(grisFiltre)
    ecartType = np.std(grisFiltre)
    seuil = moyenne + 3*ecartType
    _, masque = cv2.threshold(grisFiltre,seuil,255,cv2.THRESH_BINARY)

    # Recherche des contours
    contours, _ = cv2.findContours(masque,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    contour = max(contours, key=len)
    if len(contour) < 5:
        return None

    # Calcul du barycentre
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    x = M["m10"]/M["m00"]
    y = M["m01"]/M["m00"]
    return x, y




# Asservissement

xRef = None
yRef = None
integrale = 0
dernierEchantillon = time.time()

print("Système prêt")
print("R : verrouiller l'étoile")
print("Q : quitter")

while True:
    ret, image = cap.read()
    if not ret:
        continue

    tempsActuel = time.time()
    barycentre = trouverBarycentre(image)
    affichage = image.copy()

    if barycentre is not None:
        x, y = barycentre

        # Affichage du repere de l'astre suivi (Vert)
        cv2.circle(affichage,(int(x), int(y)),12, (0,255,0),2)
        cv2.drawMarker(affichage, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)

        if xRef is not None:
            # Affichage de la cible (Rouge)
            cv2.drawMarker(affichage, (int(xRef), int(yRef)), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)

            if (tempsActuel - dernierEchantillon) >= T:

                erreurPx = xRef - x # calcul de l'erreur

                integrale += erreurPx*T # correcteur PI
                integrale = np.clip(integrale, -integraleMax,integraleMax) #saturation de l'intégrale

                commande = Kp*erreurPx + Ki*integrale

                freq = int(np.clip(sensMoteur*commande,-freqMax,freqMax))
                envoyerCommande(freq) #envoi de la commande en fréquence au driver

                dernierEchantillon = tempsActuel

                print(f"Erreur = {erreurPx:+.2f} px | "f"Commande = {freq} Hz")

    cv2.imshow("Guidage", affichage)

    touche = cv2.waitKey(10) & 0xFF

    if touche == ord('q'):
        break

    elif touche == ord('r') and barycentre is not None:
        xRef = x
        yRef = y
        integrale = 0
        print(f"Reference : " f"({xRef:.1f} ; {yRef:.1f})")

envoyerCommande(0)
cap.release()
cv2.destroyAllWindows()
if connexion:
    carte.close()