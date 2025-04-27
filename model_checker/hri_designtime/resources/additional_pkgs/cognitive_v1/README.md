Per lanciare attualemente il main serve la seguente riga di comando:
/.main argv1.csv argv2.xml argv3.xml argv4.csv argv5.xml argv6.xml argv7.xml DPA XML main.xml
dove: 
- argv1.csv : è il nome del file di configuarazione messo nella MISSION FOLDER(DPA)
- argv2.xml : è il percorso relativo del file "argv2.xml" rispetto alla cartella in cui il "main.cpp" è inserito, in questo caso non necessito di mettere tipo "cartella\\argv2.csv", perchè è nella stessa folder del "main.cpp" 
- argv3.xml : stessa cosa fatta per l'"argv2.xml", ma per il file "argv3.xml"
- argv4.csv : stessa cosa fatta per l'"argv2.xml", ma per il file "argv4.csv"
- argv5.xml : stessa cosa fatta per l'"argv2.xml", ma per il file "argv5.xml"
- argv6.xml : stessa cosa fatta per l'"argv2.xml", ma per il file "argv6.xml"
- argv7.xml : stessa cosa fatta per l'"argv2.xml", ma per il file "argv7.xml" 
- DPA : nome della mission folder/percorso relativo dalla cartella del main per la mission folder
- XML : nome della xml folder/percorso relativo dalla cartella del main per la xml folder (folder in cui viene generato il file "main.xml")
- main.xml : nome del file main/percorso relativo dalla cartella xml (se in qualche sotto cartella ) 
