"""
Interface graphique pour dessiner et tester le réseau de neurones

Permet de:
- Dessiner des chiffres à la souris
- Prédire en temps réel
- Voir les probabilités de chaque classe
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import pickle
from pathlib import Path


class DrawingInterface:
    """Interface de dessin pour tester le réseau"""

    def __init__(self, model_path=None):
        """
        Initialise l'interface

        Args:
            model_path: chemin vers le modèle sauvegardé
        """
        self.window = tk.Tk()
        self.window.title("<¨ Reconnaissance de Chiffres")
        self.window.geometry("900x600")

        # Charger le modèle
        self.model = self.load_model(model_path)

        # Canvas pour dessiner (280x280 pixels)
        self.canvas_size = 280
        self.canvas = tk.Canvas(
            self.window,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='black',
            cursor='cross'
        )
        self.canvas.grid(row=0, column=0, padx=20, pady=20, rowspan=5)

        # Image PIL pour stocker le dessin
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)

        # Variables pour le dessin
        self.pen_width = 20
        self.last_x = None
        self.last_y = None

        # Bind mouse events
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

        # Boutons
        btn_frame = ttk.Frame(self.window)
        btn_frame.grid(row=0, column=1, padx=10, pady=10, sticky='n')

        ttk.Button(btn_frame, text="=Ñ Effacer", command=self.clear_canvas).pack(pady=5, fill='x')
        ttk.Button(btn_frame, text="= Prédire", command=self.predict).pack(pady=5, fill='x')
        ttk.Button(btn_frame, text="L Quitter", command=self.window.quit).pack(pady=5, fill='x')

        # Label pour la prédiction
        self.result_label = ttk.Label(
            self.window,
            text="Dessinez un chiffre",
            font=('Arial', 24, 'bold')
        )
        self.result_label.grid(row=1, column=1, padx=10, pady=10)

        # Frame pour les barres de probabilité
        self.prob_frame = ttk.Frame(self.window)
        self.prob_frame.grid(row=2, column=1, padx=10, pady=10, sticky='nsew')

        # Créer les barres de probabilité
        self.prob_bars = {}
        for i in range(10):
            frame = ttk.Frame(self.prob_frame)
            frame.pack(fill='x', pady=2)

            ttk.Label(frame, text=f"{i}:", width=3).pack(side='left')

            canvas = tk.Canvas(frame, height=20, bg='lightgray')
            canvas.pack(side='left', fill='x', expand=True)
            self.prob_bars[i] = canvas

        print(" Interface créée!")

    def load_model(self, model_path):
        """Charge le modèle sauvegardé"""
        if model_path is None:
            model_path = Path(__file__).parent / 'models' / 'mnist_network.pkl'

        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)

            # Recréer le modèle
            from src.network import NeuralNetwork
            model = NeuralNetwork(data['layer_dims'], data['learning_rate'])
            model.parameters = data['parameters']

            print(f" Modèle chargé depuis: {model_path}")
            return model
        except Exception as e:
            print(f"  Erreur de chargement du modèle: {e}")
            print("   Créez un modèle d'abord avec le notebook 04!")
            return None

    def paint(self, event):
        """Dessine sur le canvas"""
        x, y = event.x, event.y

        if self.last_x and self.last_y:
            # Dessiner sur le canvas Tkinter
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=self.pen_width,
                fill='white',
                capstyle=tk.ROUND,
                smooth=True
            )

            # Dessiner sur l'image PIL
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=255,
                width=self.pen_width
            )

        self.last_x = x
        self.last_y = y

    def reset(self, event):
        """Reset la position de la souris"""
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        """Efface le canvas"""
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Dessinez un chiffre")

        # Réinitialiser les barres
        for canvas in self.prob_bars.values():
            canvas.delete('all')

    def preprocess_image(self):
        """
        Prétraite l'image pour le réseau
        280x280 ’ 28x28 ’ vecteur 784
        """
        # Resize à 28x28
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)

        # Convertir en array numpy
        img_array = np.array(img_resized, dtype=np.float32)

        # Normaliser [0, 255] ’ [0, 1]
        img_array = img_array / 255.0

        # Aplatir en vecteur
        img_vector = img_array.flatten().reshape(1, -1)

        return img_vector, img_array

    def predict(self):
        """Fait une prédiction"""
        if self.model is None:
            self.result_label.config(text="L Pas de modèle!")
            return

        # Prétraiter l'image
        img_vector, img_28x28 = self.preprocess_image()

        # Prédiction
        predictions, _ = self.model.forward(img_vector)
        predicted_digit = np.argmax(predictions)
        confidence = predictions[0, predicted_digit]

        # Afficher le résultat
        self.result_label.config(
            text=f"Prédiction: {predicted_digit}\n({confidence:.1%})"
        )

        # Mettre à jour les barres de probabilité
        for digit in range(10):
            prob = predictions[0, digit]
            canvas = self.prob_bars[digit]
            canvas.delete('all')

            # Couleur selon la probabilité
            if digit == predicted_digit:
                color = 'green'
            else:
                color = 'lightblue'

            # Largeur de la barre proportionnelle à la probabilité
            width = int(prob * canvas.winfo_width())
            canvas.create_rectangle(0, 0, width, 20, fill=color, outline='')
            canvas.create_text(5, 10, text=f"{prob:.1%}", anchor='w')

    def run(self):
        """Lance l'interface"""
        print("\n" + "="*60)
        print("<¨ Interface de Dessin")
        print("="*60)
        print("\nInstructions:")
        print("  1. Dessinez un chiffre avec la souris")
        print("  2. Cliquez sur 'Prédire' pour voir le résultat")
        print("  3. Utilisez 'Effacer' pour recommencer")
        print("\n" + "="*60 + "\n")

        self.window.mainloop()


def main():
    """Point d'entrée principal"""
    # Chercher un modèle
    model_path = Path(__file__).parent / 'models' / 'mnist_network.pkl'

    if not model_path.exists():
        print("\n  Aucun modèle trouvé!")
        print("   Entraînez d'abord un modèle avec le notebook 04_building_complete_network.ipynb")
        print(f"   Le modèle sera sauvegardé dans: {model_path}")
        return

    # Créer et lancer l'interface
    app = DrawingInterface(model_path)
    app.run()


if __name__ == '__main__':
    main()
