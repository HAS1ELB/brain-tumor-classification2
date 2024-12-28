import os
import gdown

# Chemin du dossier
folder_path = 'data/processed'

# Créer le dossier s'il n'existe pas
os.makedirs(folder_path, exist_ok=True)

def download_processed_data_from_drive(url, output_path):
    """
    Télécharge un fichier partagé depuis Google Drive.
    Args:
        url (str): Lien partagé de Google Drive.
        output_path (str): Chemin local où sauvegarder le fichier.
    """
    if not os.path.exists(output_path):
        print(f"Téléchargement du fichier depuis Google Drive : {output_path}")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"Fichier déjà existant : {output_path}")


# Liste des données traitées à télécharger
processed_data_to_download = {
    "training_data": ("https://drive.google.com/uc?id=1tB6E-XBVVb_q0BHKj0RnzrdPbmg7li3C", "data/processed/training_data.npz"),
    "testing_data": ("https://drive.google.com/uc?id=1NyCUTavtHihPUzBrqN2jRdT-rk1tW2Nk", "data/processed/testing_data.npz")
}

# Téléchargement des données traitées
for data_name, (url, path) in processed_data_to_download.items():
    if not os.path.exists(path):
        print(f"Le fichier {data_name} n'existe pas. Téléchargement en cours...")
        download_processed_data_from_drive(url, path)
        print(f"Le fichier {data_name} a été téléchargé et sauvegardé à {path}.")
    else:
        print(f"Le fichier {data_name} existe déjà à l'emplacement {path}.")
