import re

# Charger la liste des poses à ignorer depuis ignore_list.csv
with open('ignore_list.csv', 'r', encoding='utf-8') as f:
    ignore_set = {line.strip() for line in f if line.strip()}

# Lire la liste complète des fichiers depuis aist.csv
with open('aist.csv', 'r', encoding='utf-8') as f:
    files = [line.strip() for line in f if line.strip()]

valid_files = []

# Pour chaque fichier, on retire l'extension et on remplace le tag de caméra par 'cAll'
for filename in files:
    # Suppression de l'extension .mp4
    if filename.endswith('.mp4'):
        basename = filename[:-4]
    else:
        basename = filename

    # Remplacer le numéro de caméra par 'cAll'
    # On suppose que la partie caméra a le format "_cXX" où XX sont deux chiffres
    key = re.sub(r'_c\d{2}', '_cAll', basename)
    
    # Si la pose n'est pas dans la liste d'ignore, on garde le fichier
    if key not in ignore_set:
        valid_files.append(filename)

# Écrire les fichiers valides dans un nouveau CSV
with open('aist_valid.csv', 'w', encoding='utf-8') as f:
    for valid in valid_files:
        f.write(valid + '\n')

print(f"{len(files) - len(valid_files)} fichiers supprimés, {len(valid_files)} fichiers valides conservés.")
