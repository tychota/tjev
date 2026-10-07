# French samples for review (mix-v5)

The French generator templates introduced in mix-v5 ([fr-data.md §5](fr-data.md#5-mix-v5-as-built))
and the `jev_fr_pilot` translations, as rendered for review. The held-out templates are
unchanged since; mix-v5b changed return_window's windows and added hint lines to some
training rows. The machine-translated `jev_fr_mt` rows of mix-v7.5 are not sampled here.

French rows for a native speaker to review: please flag anything unnatural, wrong or ambiguous (the expected answer must hold under any reasonable reading).

- **1–21: generator training templates**, two per generator (one for the long policy).
- **22–30: held-out templates**, which training never sees.
- **31–40: translated contender rows** (`jev_fr_pilot`).

Generator labels are computed by code; a template only changes the wording (`tests/data/test_generators.py::test_a_template_never_changes_the_label`).

### 1. gen_return_window (entraînement)

> Politique du magasin : retour possible dans les 14 jours suivant l'achat.  
> Date d'achat : 16 novembre 2024. Aujourd'hui : 22 novembre 2024.

- **Question :** Le retour est-il encore dans le délai autorisé ?
- **Options :** Non; Oui
- **Réponse attendue :** oui

### 2. gen_return_window (entraînement)

> Échange ou remboursement possible jusqu'à J+14, le jour J étant celui de l'achat.  
> Achat : lundi 1er juin 2026  
> Demande : vendredi 19 juin 2026

- **Question :** La demande arrive-t-elle avant l'expiration du délai ?
- **Options :** Non; Oui
- **Réponse attendue :** non

### 3. gen_weekday (entraînement)

> Départ : mercredi 17/10/2029.  
> Retour : 67 jours après le départ.

- **Question :** Quel jour de la semaine le retour est-il prévu ?
- **Options :** lundi; mardi; mercredi; jeudi; vendredi; samedi; dimanche
- **Réponse attendue :** dimanche

### 4. gen_weekday (entraînement)

> La commande a été passée le jeudi 25 janvier 2024 ; l'expédition intervient 169 jours après.

- **Question :** Quel jour de la semaine la commande est-elle expédiée ?
- **Options :** lundi; mardi; mercredi; jeudi; vendredi; samedi; dimanche
- **Réponse attendue :** vendredi

### 5. gen_invoice_total (entraînement)

> Lignes de facture :  
> - chaises : 10 × 295,99 €  
> - licences : 5 × 74,00 €  
> Remise : 15 % sur toute la commande.  
> Total indiqué : 2 840,41 €

- **Question :** Le total indiqué est-il correct (au centime près) ?
- **Options :** Non; Oui
- **Réponse attendue :** non

### 6. gen_invoice_total (entraînement)

> Récapitulatif d'achat :  
> - licences, 4 × 139,99 €  
> - câbles, 5 × 71,50 €  
> - bureaux, 10 × 52,50 €  
> Réduction : 15 % sur le total.  
> À régler : 1 236,08 €

- **Question :** La somme à régler est-elle bien calculée ?
- **Options :** Non; Oui
- **Réponse attendue :** non

### 7. gen_base_rate (entraînement)

> Chiffres connus : les positifs représentent 0,72 % de la population ; sensibilité 79,6 % ; taux de faux positifs 1,07 %.  
> Résultat pour ce cas : positif.

- **Question :** Faut-il en conclure que l'e-mail est une tentative d'hameçonnage ?
- **Options :** Non; Oui
- **Réponse attendue :** probabilité de « oui » 0.35

### 8. gen_base_rate (entraînement)

> Sur l'ensemble des dossiers, 0,21 % sont positifs. Le système d'alerte repère 96,5 % des dossiers positifs, mais signale aussi à tort 6,52 % des dossiers négatifs. Ce dossier a déclenché une alerte.

- **Question :** Compte tenu de cette alerte, est-ce que la transaction est frauduleuse ?
- **Options :** Non; Oui
- **Réponse attendue :** probabilité de « oui » 0.03

### 9. gen_schedule_conflict (entraînement)

> Créneaux déjà pris :  
> - de 9 h 00 à 10 h 30  
> - de 9 h 30 à 11 h 00  
> On me propose un rendez-vous de 14 h 00 à 14 h 30.

- **Question :** Ce rendez-vous empiète-t-il sur un créneau déjà pris ?
- **Options :** Non; Oui
- **Réponse attendue :** non

### 10. gen_schedule_conflict (entraînement)

> Agenda (occupé) :  
> - 12h30–13h00  
> - 15h30–17h00  
> Réunion proposée : 9h45–10h45.

- **Question :** La réunion proposée chevauche-t-elle un créneau existant ?
- **Options :** Non; Oui
- **Réponse attendue :** non

### 11. gen_table_argmax (entraînement)

> Classement non trié, heures saisies :  
> - Emma : 165  
> - Chloé : 318  
> - Bruno : 393  
> - Alice : 239  
> - Farid : 477

- **Question :** Qui arrive en tête pour les heures saisies ?
- **Options :** Emma; Chloé; Bruno; Alice; Farid
- **Réponse attendue :** Farid

### 12. gen_table_argmax (entraînement)

> | collaborateur | heures saisies |  
> |---|---|  
> | Alice | 231 |  
> | Hugo | 497 |  
> | Bruno | 131 |

- **Question :** Quel collaborateur affiche le chiffre le plus élevé en heures saisies ?
- **Options :** Alice; Hugo; Bruno
- **Réponse attendue :** Hugo

### 13. gen_refund_policy (entraînement)

> Politique de remboursement (document interne) :  
> - Les produits en « vente finale » ne peuvent pas être remboursés.  
> - Nous remboursons les produits de la catégorie électronique si la demande arrive sous 30 jours après la livraison.  
> - Nous ne remboursons pas les logiciels ouverts.  
> - Au-dessus de 500,00 €, la demande doit être validée par un superviseur.  
> - Les titulaires de la carte de fidélité disposent de 15 jours supplémentaires.  
>   
> Le client (titulaire de la carte de fidélité) demande le remboursement de 398,00 € ; le colis est arrivé il y a 14 jours et l'article est ouvert.

- **Question :** Quelle est la bonne issue pour cette demande de remboursement ?
- **Options :** `approve` Rembourser le client; `deny` Ne pas donner suite; `escalate` Transmettre à un superviseur
- **Réponse attendue :** approve

### 14. gen_refund_policy (entraînement)

> Règles de remboursement :  
> - Les articles vendus en « vente finale » sont exclus des remboursements.  
> - Un superviseur doit valider tout remboursement de plus de 1 000,00 €.  
> - Les membres du club ont droit à 30 jours de plus.  
> - Les achats de la catégorie épicerie sont remboursables jusqu'à 14 jours après la livraison.  
> - Un logiciel dont l'emballage a été ouvert n'est pas remboursable.  
>   
> Client : membre du club. Livré il y a 35 jours. Article intact. Montant : 912,00 €.

- **Question :** Comment l'agent doit-il traiter cette demande ?
- **Options :** `approve` Accepter le remboursement; `deny` Le refuser; `escalate` Faire valider par un superviseur
- **Réponse attendue :** approve

### 15. gen_long_refund_policy (entraînement)

> Politique :  
> Article 1. Force majeure. Aucune partie n'est responsable des retards dus à des événements imprévisibles. Le vendeur informe les clients des retards importants dans les meilleurs délais. Ces événements comprennent les inondations, les grèves, les pandémies et les pannes des réseaux publics. Si l'événement dure plus de 7 semaines, chaque partie peut annuler la commande concernée. Les obligations sont suspe […] nt d'un produit de la catégorie papeterie. Certains articles de la catégorie papeterie sont vendus exclusivement en ligne. Les photos de la catégorie papeterie sont non contractuelles ; les couleurs peuvent légèrement varier.  
>   
> Nouvelle demande : un client membre réclame 1 601,00 € ; catégorie électronique, article ouvert, livré il y a 41 jours (article marqué « vente finale »).

- **Question :** Que faut-il faire de cette demande ?
- **Options :** `approve` Accorder le remboursement; `deny` Le refuser; `escalate` Renvoyer vers un superviseur
- **Réponse attendue :** deny

### 16. gen_ticket_triage (entraînement)

> Bonsoir, je n'arrive pas à me connecter à mon compte pour Premium. Réf. 652674.

- **Question :** Quelle file de support doit traiter ce message ?
- **Options :** `billing` Demandes liées à la facturation et aux paiements; `technical` Demandes liées aux problèmes techniques; `account` Demandes liées à l'accès et aux réglages du compte; `shipping` Demandes liées aux livraisons; `other` Pas une demande pour ces files
- **Réponse attendue :** account

### 17. gen_ticket_triage (entraînement)

> Salut. Comment ajouter un second utilisateur sur mon compte pour CloudDrive ?

- **Question :** Orientez ce message client vers le bon service.
- **Options :** `billing` Service facturation : débits, factures, remboursements de paiements; `technical` Support technique : plantages, erreurs, appareils en panne; `account` Gestion des comptes : connexion, mots de passe, paramètres; `shipping` Logistique : commandes en cours, livraisons, colis; `other` Hors sujet : aucun service concerné
- **Réponse attendue :** account

### 18. gen_spelling_error (entraînement)

> À relire : « Paul enverra le contrat révisé à l'équipe avant jeudi. Nous prévoyons de terminer le budget avant la fin du mois. »

- **Question :** Ce texte doit-il être corrigé (orthographe ou grammaire) ?
- **Options :** Non; Oui
- **Réponse attendue :** non

### 19. gen_spelling_error (entraînement)

> Texte : Merci de dire à Tom si le nouvau planning convient pour vendredi.

- **Question :** Ce texte comporte-t-il au moins une faute ?
- **Options :** Non; Oui
- **Réponse attendue :** oui

### 20. gen_register (entraînement)

> Madame, Monsieur, veuillez trouver ci-joint le compte rendu demandé. Je vous prie d'agréer mes salutations distinguées.

- **Question :** Le ton du message est-il plutôt relâché ou formel ?
- **Options :** 0: relâché < 1: neutre < 2: formel
- **Réponse attendue :** 2

### 21. gen_register (entraînement)

> Bonjour tout le monde, j'ai mis à jour le compte rendu, dis-moi s'il manque quelque chose. Merci !

- **Question :** Quel est le registre de ce message ?
- **Options :** 0: familier, comme entre amis < 1: courant, comme entre collègues < 2: soutenu, comme dans un courrier officiel
- **Réponse attendue :** 1

### 22. gen_return_window (held-out)

> Garantie commerciale : toute plainte doit nous parvenir au plus tard 7 jours après la remise du colis.  
> Colis remis le mercredi 15 avril 2026. Plainte reçue le mardi 28 avril 2026.

- **Question :** La plainte est-elle recevable dans les délais ?
- **Options :** Non; Oui
- **Réponse attendue :** non

### 23. gen_weekday (held-out)

> Le vol aller part le vendredi 25/07/2031 ; le vol retour décolle 473 jours après.

- **Question :** Quel jour de la semaine décolle le vol retour ?
- **Options :** lundi; mardi; mercredi; jeudi; vendredi; samedi; dimanche
- **Réponse attendue :** mardi

### 24. gen_invoice_total (held-out)

> Devis accepté par le client :  
> 1 × vannes — 281,00 € par unité  
> 10 × casques — 141,00 € par unité  
> Réduction négociée sur l'ensemble du devis : 15 %.  
> Total général facturé : 1 437,35 €

- **Question :** Le total général est-il cohérent avec le devis ?
- **Options :** Non; Oui
- **Réponse attendue :** oui

### 25. gen_schedule_conflict (held-out)

> Tournée du technicien, interventions déjà planifiées :  
> * 15 h 15 jusqu'à 16 h 30  
> * 17 h 00 jusqu'à 19 h 00  
> * 18 h 00 jusqu'à 18 h 50  
> Dépannage urgent à caser : 17 h 00 jusqu'à 17 h 45.

- **Question :** Le dépannage urgent tomberait-il en même temps qu'une intervention planifiée ?
- **Options :** Non; Oui
- **Réponse attendue :** oui

### 26. gen_refund_policy (held-out)

> Charte après-vente :  
> - Un dispositif médical dont le scellé est rompu ne peut pas être repris contre remboursement.  
> - Les lots de déstockage vendus comme définitifs ne sont pas couverts.  
> - Les demandes d'un montant supérieur à 200,00 € sont soumises à l'accord d'un responsable.  
> - Les abonnés au programme de fidélité bénéficient de 30 jours de plus.  
> - Les demandes de remboursement pour la gamme dispositifs médicaux sont honorées pendant 45 jours après la remise.  
>   
> Fiche : acheteur abonné au programme de fidélité ; remise il y a 71 jours ; scellé intact ; demande de 113,00 €.

- **Question :** Quelle suite donner à cette demande ?
- **Options :** `approve` Honorer la demande; `deny` Rejeter la demande; `escalate` La soumettre à l'accord d'un responsable
- **Réponse attendue :** approve

### 27. gen_base_rate (held-out)

> Un modèle d'audit a marqué cet enregistrement. Lors des audits précédents, 1,54 % des enregistrements posaient réellement problème ; le modèle en marque 88,1 %, ainsi que 18,15 % des enregistrements sains.

- **Question :** Après ce marquage, doit-on considérer que la pièce est défectueuse ?
- **Options :** Non; Oui
- **Réponse attendue :** probabilité de « oui » 0.07

### 28. gen_table_argmax (held-out)

> | agent | défauts trouvés |  
> |---|---|  
> | Kenji | 46 |  
> | Oscar | 206 |  
> | Jonas | 208 |  
> | Mateo | 137 |  
> | Priya | 50 |  
> | Nadia | 302 |

- **Question :** Quel agent compte le moins de défauts trouvés ?
- **Options :** Kenji; Oscar; Jonas; Mateo; Priya; Nadia
- **Réponse attendue :** Kenji

### 29. gen_ticket_triage (held-out)

> Rebonjour. Quelqu'un d'autre semble connecté avec mes identifiants pour MailPro. C'est urgent.

- **Question :** Attribuez le message entrant à un seul département.
- **Options :** `billing` Recouvrement et encaissements; `technical` Ingénierie produit d'astreinte; `account` Identité et authentification; `shipping` Relations transporteurs; `privacy` Délégué à la protection des données; `cancellation` Fidélisation et fins de contrat; `other` Personne : simple bavardage ou question sans rapport
- **Réponse attendue :** account

### 30. gen_register (held-out)

> yo la team mardi ça va être chaud mdrr biz

- **Question :** Quel est le degré de formalité de ce message ?
- **Options :** 0: relâché / très familier < 1: standard < 2: très soutenu / officiel
- **Réponse attendue :** 0

### 31. jev_fr_pilot (traduction de certo-eval-1062-0)

> L'article est usagé. Le retour a été demandé 16 jours après l'achat.

- **Question :** Sélectionnez l'action qu'impose la politique de retour fournie. Règle : accepter un article non utilisé si le retour a été demandé dans un délai de 14 jours (inclus). Refuser lorsqu'une condition énoncée n'est pas remplie. Ne demander des informations que lorsque l'éligibilité ne peut pas être déterminée à partir des faits énoncés.
- **Options :** `a` Accepter le retour conformément à la politique.; `b` Refuser le retour conformément à la politique.; `c` Demander les informations nécessaires pour déterminer l'éligibilité.
- **Réponse attendue :** b

### 32. jev_fr_pilot (traduction de certo-train-10267-0)

> Une dépense de 375 $ est classée dans la catégorie « personnel ». Le plafond par dépense est de 50 $.

- **Question :** La dépense est-elle remboursable ? Elle doit relever d'une catégorie professionnelle et respecter le plafond.
- **Options :** la dépense n'est pas remboursable; la dépense est remboursable
- **Réponse attendue :** non

### 33. jev_fr_pilot (traduction de certo-train-5311-0)

> Une demande de remboursement est faite 55 jours après l'achat. Aucun justificatif d'achat n'est joint.

- **Question :** Appliquez la politique de remboursement : les remboursements sont acceptés dans un délai de 30 jours sur présentation d'un justificatif d'achat.
- **Options :** `a` Approuver le remboursement.; `b` Refuser le remboursement.; `c` Demander le justificatif d'achat manquant.
- **Réponse attendue :** c

### 34. jev_fr_pilot (traduction de decider_teacher-eval-12532-0)

> Mon permis de stationnement a expiré et je dois le renouveler.

- **Question :** Quel est le sujet principal de votre message ?
- **Options :** `parking_permit` ; `waste_collection` ; `property_tax` ; `general_support` ; `out_of_scope` 
- **Réponse attendue :** parking_permit

### 35. jev_fr_pilot (traduction de decider_teacher-train-10099-0)

> Bonjour, vous vendez des ordinateurs portables pas chers ou du matériel pour développeurs ?

- **Question :** Cette demande liée à une pull request porte-t-elle sur une approbation de revue de code ou sur un conflit de fusion ?
- **Options :** `needs_code_approval` ; `merge_conflict_help` ; `general_pr_request` ; `none of the above` 
- **Réponse attendue :** none of the above

### 36. jev_fr_pilot (traduction de decider_teacher-train-22996-0)

> Le sous-marin navigue sans à-coups à 40 mètres de profondeur, avec 85 % de batterie restante. Le sonar ne détecte aucun contact dans un rayon de 300 mètres, et le cône du sonar avant est dégagé de toute mine ou obstacle. Le chronomètre de mission indique 45 minutes restantes et l'objectif se trouve à 12 kilomètres sur le cap actuel.

- **Question :** Quelle est la meilleure ligne de conduite à l'instant présent ?
- **Options :** `option_1` Plonger immédiatement à 150 mètres pour se cacher; `option_2` Couper les moteurs et dériver en silence; `option_3` Larguer une bouée sonar leurre; `option_4` Maintenir la vitesse et le cap actuels; `option_5` Virer de 90 degrés pour éviter une menace potentielle; `option_6` Faire surface pour consulter le radar
- **Réponse attendue :** option_4

### 37. jev_fr_pilot (traduction de mghafiri_scenarios-eval-118-3)

> {  
>  "email": {  
>   "from": "p.ambrose@example.org",  
>   "to": "publicworks@example.org",  
>   "subject": "Votre chasse-neige a ENCORE démoli ma boîte aux lettres",  
>   "body": "Votre chasse-neige a arraché ma boîte aux lettres de son poteau mardi matin, le poteau est cassé net à la base. C'est la troisième fois en 4 ans. Je veux que la ville en paie une neuve. Et votre conducteur roulait beaucoup trop vite pour une rue résiden […] et recul) et (2) une lame ou un véhicule de déneigement de la ville l'a physiquement heurtée. Les dommages causés par la neige projetée ou poussée par un chasse-neige ne sont pas remboursables. Les résidents doivent envoyer le formulaire de réclamation CL-2 avec des photos. Les plaintes concernant le comportement des conducteurs sont transmises au superviseur des opérations."  
> }

- **Question :** `email.body` se plaint-il de la façon dont conduisait le conducteur du chasse-neige ?
- **Options :** ; 
- **Réponse attendue :** probabilité de « oui » 0.96

### 38. jev_fr_pilot (traduction de mghafiri_scenarios-train-156-1)

> {  
>  "listing_a": {  
>   "id": "LST-201",  
>   "seller": "SEL-3310",  
>   "title": "Gourde isotherme Aquaflow 500 ml - Acier inoxydable - Noir mat",  
>   "attributes": {  
>    "capacity": "500 ml",  
>    "material": "Acier inoxydable 18/8",  
>    "lid": "Bouchon rabattable avec paille",  
>    "color": "Noir mat"  
>   },  
>   "upc": "0-12345-67890-5"  
>  },  
>  "listing_b": {  
>   "id": "LST-877",  
>   "seller": "SEL-3310",  
>   "title": "Bouteille isotherme Aquaflow 16,9 oz, noire, bouchon paille",  
>   "attributes": {  
>    "capacity": "16,9 fl oz",  
>    "material": "Acier inoxydable",  
>    "lid": "Bouchon paille",  
>    "color": "Noir"  
>   },  
>   "upc": "non fourni"  
>  },  
>  "policy": "Les vendeurs ne peuvent pas créer d'annonces en double pour un même produit. Les variantes (taille, couleur) doivent être regroupées sous une seule annonce."  
> }

- **Question :** Les deux annonces proviennent-elles du même vendeur ?
- **Options :** ; 
- **Réponse attendue :** probabilité de « oui » 0.98

### 39. jev_fr_pilot (traduction de mghafiri_scenarios-train-282-2)

> {  
>  "clause": "Le présent Contrat sera renouvelé automatiquement pour des périodes successives d'un an, sauf si l'une ou l'autre des parties notifie son intention de ne pas le renouveler au moins soixante (60) jours avant la fin de la période en cours. Le Prestataire pourra augmenter les tarifs pour toute période de renouvellement dans la limite de 5 %, sous réserve d'en informer le Client au moins quatre-vingt-dix (90) jours avant le renouvellement."  
> }

- **Question :** Quelle partie `clause` avantage-t-elle ?
À considérer : qui peut empêcher le renouvellement ; qui peut modifier le prix
- **Options :** 0: Avantage le Prestataire < 1: Équilibrée < 2: Avantage le Client
- **Réponse attendue :** 0 (0.53)

### 40. jev_fr_pilot (traduction de openjev-eval-6446-0)

> Client Harbor-b1a63d8326 : L'export d'un rapport échoue, mais je peux télécharger le CSV et terminer dans l'application de bureau. C'est la demande que je veux que vous traitiez en priorité.  
> Agent : Pourriez-vous ajouter des précisions ?  
> Client : Ça arrive de temps en temps, mais je ne me souviens pas de l'enchaînement qui y a mené. C'est frustrant, mais j'apprécie votre aide et je vais rester constructif. Je vais continuer à utiliser le service et renouveler mon abonnement.

- **Question :** L'utilisateur demande explicitement un remboursement ou un avoir pour le problème actuel. Une question sur des débits, un refus explicite de se faire rembourser ou la mention d'un remboursement déjà effectué ne constitue pas une demande en cours.
- **Options :** ; 
- **Réponse attendue :** non
