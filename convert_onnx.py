import onnx
import os

# Le chemin exact où la voiture cherche son fichier
chemin = "models/v18/best.onnx"

if not os.path.exists(chemin):
    print(f"ERREUR : Le fichier {chemin} est introuvable !")
    exit(1)

print(f"Chargement du modèle : {chemin}")
model = onnx.load(chemin)

print(f"-> Ancienne IR version : {model.ir_version}")

# On force l'en-tête pour la Jetson
model.ir_version = 8

# On force aussi la version des opérations (opset) au cas où ton PC aurait forcé la 17
for imp in model.opset_import:
    if imp.version > 11:
        print(f"-> Baisse de l'opset de {imp.version} à 11")
        imp.version = 11

# On écrase directement le fichier de la voiture
onnx.save(model, chemin)

# On relit le fichier pour PROUVER que ça a marché
verif_model = onnx.load(chemin)
print(f"\n[SUCCÈS] Nouvelle IR version enregistrée : {verif_model.ir_version}")
print("Tu peux lancer la voiture !")