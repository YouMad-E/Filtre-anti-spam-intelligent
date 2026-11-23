import numpy as np
import os
import math
import re
import pickle

def lireMail(fichier, dictionnaire):
	""" 
	Lire un fichier et retourner un vecteur de booléens en fonctions du dictionnaire
	"""
	f = open(fichier, "r",encoding="ascii", errors="surrogateescape")
	mots = re.findall(r'[A-Z]{3,}', f.read().upper())
	f.close()
	
	x = [False] * len(dictionnaire)
	
	for i in range(len(x)):
		x[i] = dictionnaire[i] in mots
	
	return x

def charge_dico(fichier):
	f = open(fichier, "r")
	mots = f.read().split("\n")
	print("Chargé " + str(len(mots)) + " mots dans le dictionnaire")
	f.close()
	
	def exclureNomMoins3Lettres(x):
		return len(x) > 2
	numbers = list(filter(exclureNomMoins3Lettres, mots))
	print("Filtré " + str(len(numbers)) + " mots dans le dictionnaire")

	return numbers

def apprendBinomial(dossier, fichiers, dictionnaire):
	"""
	Fonction d'apprentissage d'une loi binomiale a partir des fichiers d'un dossier
	Retourne un vecteur b de paramètres 
		
	"""
	b = np.zeros(len(dictionnaire))
	m = len(fichiers)
	epsilon = 1 #pour eviter les proba a 0

	for fichier in fichiers:
		x = lireMail(dossier + "/" + fichier, dictionnaire)
		b += x

	b = (b + epsilon) / (m + 2 * epsilon)

	return b


def prediction(x, Pspam, Pham, bspam, bham):
	"""
		Prédit si un mail représenté par un vecteur booléen x est un spam
		à partir du modèle de paramètres Pspam, Pham, bspam, bham.
		Retourne True ou False.
		
	"""
	x_np = np.array(x, dtype=bool) #convertire la liste x en tableu bool 

	score_spam = math.log(Pspam) \
	+ np.sum(np.log(bspam[x_np])) \
	+ np.sum(np.log(1 - bspam[~x_np]))

	score_ham = math.log(Pham) \
	+ np.sum(np.log(bham[x_np])) \
	+ np.sum(np.log(1 - bham[~x_np]))

	max_score = max(score_spam, score_ham) #pour éviter les dépassements avec math.exp()

	#Calcul des probabilités exactes (a posteriori)
	exp_spam = math.exp(score_spam - max_score)
	exp_ham = math.exp(score_ham - max_score)
	somme_exp = exp_spam + exp_ham

	Pspam_x = exp_spam / somme_exp
	Pham_x = exp_ham / somme_exp

	isSpam = score_spam > score_ham #la décision final
	
	return isSpam, Pspam_x, Pham_x
	
def test(dossier, isSpam, Pspam, Pham, bspam, bham):
	"""
		Test le classifieur de paramètres Pspam, Pham, bspam, bham 
		sur tous les fichiers d'un dossier étiquetés 
		comme SPAM si isSpam et HAM sinon
		
		Retourne le taux d'erreur 
	"""
	fichiers = os.listdir(dossier)
	errors = 0

	for fichier in fichiers:
		x = lireMail(dossier + "/" + fichier, dictionnaire)
		pred, Pspam_x, Pham_x = prediction(x, Pspam, Pham, bspam, bham)

		isError = isSpam != pred
		errors += isError

		print((isSpam and "SPAM" or "HAM") + " " + fichier + ": " +
			"	P(Y=SPAM | X=x) = " + str(round(Pspam_x * 100, 1)) + "%," + 
			"	P(Y=HAM | X=x) = " + str(round(Pham_x * 100, 1)) + "%" + 
			"	=> identifié comme un " + (pred and "SPAM" or "HAM") + (isError and "	*** erreur ***" or ""))

	return errors / len(fichiers)


############ programme principal ############

def testClassifieur(dossier, isSpam, classifieur, dictionnaire):
    """
    Test le classifieur à partir d'une structure dictionnaire
    """
    #pour utiliser notre fonction test() existante
    return test(dossier, isSpam, classifieur['Pspam'], classifieur['Pham'], classifieur['bspam'], classifieur['bham'])


def apprentissageEnLigne(chemin_fichier, isSpam, classifieur, dictionnaire):
    """
    Met à jour un classifieur existant avec un seul nouveau message (sans tout réapprendre)
    """
    #On lit le nouveau mail
    x = lireMail(chemin_fichier, dictionnaire)
    x_np = np.array(x, dtype=float)
    epsilon = 1
    
    if isSpam:
        #On inverse la formule du lissage pour retrouver combien 
        #de fois on avait vu chaque mot
        n_spam = classifieur['bspam'] * (classifieur['mSpam'] + 2 * epsilon) - epsilon
        
        #On ajoute le nouveau mail au compteur
        classifieur['mSpam'] += 1
        n_spam += x_np #On ajoute les nouveaux mots trouvés
        
        classifieur['bspam'] = (n_spam + epsilon) / (classifieur['mSpam'] + 2 * epsilon)
    else:
        #On fait exactement pareil si le nouveau mail est un HAM
        n_ham = classifieur['bham'] * (classifieur['mHam'] + 2 * epsilon) - epsilon
        classifieur['mHam'] += 1
        n_ham += x_np
        classifieur['bham'] = (n_ham + epsilon) / (classifieur['mHam'] + 2 * epsilon)
        
    #on met à jour les probabilités générales (Pspam et Pham)
    mTotal = classifieur['mSpam'] + classifieur['mHam']
    classifieur['Pspam'] = classifieur['mSpam'] / mTotal
    classifieur['Pham'] = classifieur['mHam'] / mTotal
    
    print("Modèle mis à jour avec succès grâce au mail : " + chemin_fichier)
    return classifieur

dossier_spams = "basetest/spam"
dossier_hams = "basetest/ham"

fichiersspams = os.listdir(dossier_spams)
fichiershams = os.listdir(dossier_hams)

mSpam = len(fichiersspams)
mHam = len(fichiershams)

# Chargement du dictionnaire:
dictionnaire = charge_dico("dictionnaire1000en.txt")
print(dictionnaire)

# Apprentissage des bspam et bham:
print("apprentissage de bspam...")
bspam = apprendBinomial(dossier_spams, fichiersspams, dictionnaire)
print("apprentissage de bham...")
bham = apprendBinomial(dossier_hams, fichiershams, dictionnaire)

# Calcul des probabilités a priori Pspam et Pham:
Pspam = mSpam / (mSpam + mHam)
Pham = mHam / (mSpam + mHam)

print("\n--- CREATION ET SAUVEGARDE DU CLASSIFIEUR ---")
#On regroupe tout dans une belle structure de données
classifieur = {
    'Pspam': Pspam,
    'Pham': Pham,
    'bspam': bspam,
    'bham': bham,
    'mSpam': mSpam,
    'mHam': mHam
}

#Sauvegarde du modèle dans un fichier physique
fichier_modele = "mon_modele_spam.pkl"
f_save = open(fichier_modele, 'wb')
pickle.dump(classifieur, f_save)
f_save.close()
print("Classifieur sauvegardé dans " + fichier_modele)

print("\n--- TESTS AVEC LA NOUVELLE STRUCTURE ---")
#On utilise testClassifieur au lieu de test()
erreurSpam = testClassifieur(dossier_spams, True, classifieur, dictionnaire)
erreurHam = testClassifieur(dossier_hams, False, classifieur, dictionnaire)

print("Erreur de test sur les SPAM: " + str(round(erreurSpam * 100, 1)) + "%")
print("Erreur de test sur les HAM: " + str(round(erreurHam * 100, 1)) + "%")
print("Erreur de test globale: " + str(round((erreurSpam * mSpam + erreurHam * mHam) / (mSpam + mHam) * 100, 1)) + "%")


print("\n--- TEST APPRENTISSAGE EN LIGNE ---")
#On simule l'arrivée d'un nouveau spam (on prend le premier du dossier test pour l'exemple)
nouveau_mail_test = dossier_spams + "/" + fichiersspams[0]
classifieur = apprentissageEnLigne(nouveau_mail_test, True, classifieur, dictionnaire)
