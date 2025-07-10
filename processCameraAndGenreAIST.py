import sys

from premiere.functionsAIST import getDanceGenres, getDanceGenreVideos


if len(sys.argv) != 5:
    print("Usage: processCameraAndGenreAIST.py <aistFileList> <aistFileMergeDirectory> <genre> <camera>")
    sys.exit(1)

aistFileList = sys.argv[1]
aistFileMergeDirectory = sys.argv[2]
genre = sys.argv[3]
camera = sys.argv[4]

# Lire la liste complète des fichiers depuis aist.csv
with open(aistFileList, 'r', encoding='utf-8') as f:
    allVideoFileNames = [line.strip() for line in f if line.strip()]

allDanceGenres = getDanceGenres()

for dances in allDanceGenres:
    genreVideos = getDanceGenreVideos(allVideoFileNames, camera, dances[0])
    print (len(genreVideos), "videos found for", dances[1])
    print (genreVideos)