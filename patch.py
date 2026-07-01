import onnx

chemin = "models/v18/best.onnx"
model = onnx.load(chemin)
model.ir_version = 8  # On force l'enveloppe à 8
onnx.save(model, chemin)

print("Patch appliqué avec succès !")