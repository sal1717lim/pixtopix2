phase 1: apprentissage du discriminateur
y' = generateur(x)
verdictFaux = discriminatuer(x,y')
ErreurFaux = BCE(verdictFaux, allZeros)
verdictReel = discriminatuer(x,y)
ErreurReel = BCE(verdictReel, allones)
ErreurDiscriminateur  =   (ErreurFaux + ErreurReel) / 2

phase 2: apprentissage du discriminateur
y' = generateur(x)
verdictFaux = discriminatuer(x,y')
ErreurFaux = BCE(verdictFaux, allZeros)
ErruerL1 = Lambda * L1(y,y') 
ErreurGenerateur =  ErreurFaux + Erreur L1